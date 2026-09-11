"""Independently score saved states on CPU and verify the resource gate."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, tensor_hash
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import math
import statistics as st

ROOT = Path("results/fused_timing_confirmation_v1")
p = read(ROOT / "protocol.json")
r = read(ROOT / "result.json")
assert not (ROOT / "audit.json").exists()
assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
hashes(p["sources"])
hashes(p["maintained_files"])
inputs = {
    f.as_posix(): sha(f)
    for f in ROOT.iterdir()
    if f.is_file() and f.suffix in (".py", ".pt", ".json")
}
write_json(
    ROOT / "audit_protocol.json",
    dict(
        inputs=inputs,
        tolerance=1e-5,
        scoring="CPU FP64, independent explicit formula, batches257",
        backward_calls=0,
    ),
)
assert len(r["cases"]) == 36 and len(r["boundaries"]) == 37
assert r["training_updates"] == r["backwards"] == 1080
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])
data = {}
for batch in p["batches"]:
    for seed in p["seeds"]:
        saved = torch.load(ROOT / f"data_{batch}_{seed}.pt", weights_only=True)
        x = torch.randn(batch, 384, generator=torch.Generator().manual_seed(50000 + batch + seed))
        q = torch.linalg.qr(
            torch.randn(384, 384, generator=torch.Generator().manual_seed(60000 + seed))
        ).Q
        assert torch.equal(saved["x"], x) and torch.equal(saved["y"], x @ q)
        data[batch, seed] = saved


def score(state, arm, sample):
    w = {k: v.double() for k, v in state.items()}
    kind = arm.split(":")[0]
    total = 0.0
    for start in range(0, len(sample["x"]), 257):
        x = sample["x"][start : start + 257].double()
        for j in range(8):
            if "q" in w:
                x = x @ w["q"][j].T + w["bias"][j]
                a = 0.25 * torch.tanh(w["theta"][j])
                x = x * (1 + a) if kind == "affine" else x + a * (x / (1 + torch.abs(x)))
            else:
                v = {
                    k.removeprefix(f"blocks.{j}."): t
                    for k, t in w.items()
                    if k.startswith(f"blocks.{j}.")
                }
                u = x @ v["up.weight"].T + v["up.bias"]
                z = (
                    u * torch.sigmoid(u) * (x @ v["gate.weight"].T + v["gate.bias"])
                    if kind == "full_swiglu"
                    else 0.5 * u * (1 + torch.erf(u / math.sqrt(2)))
                )
                x = x + (z @ v["down.weight"].T + v["down.bias"]) / math.sqrt(8)
        total += ((x - sample["y"][start : start + 257].double()) ** 2).sum().item()
    return total / sample["y"].numel()


fixed = {}
for seed in p["seeds"]:
    gen = torch.Generator().manual_seed(seed + 8)
    q = torch.linalg.qr(torch.randn(8, 384, 384, generator=gen, dtype=torch.float64)).Q.float()
    theta = torch.empty(8, 384, dtype=torch.float64).uniform_(-1.5, 1.5, generator=gen).float()
    fixed[seed] = (q, theta)
checks = []
seen = set()
for index, c in enumerate(r["cases"]):
    assert c == read(ROOT / f"case{index:02d}.json") and c["index"] == index
    key = (c["batch"], c["seed"], c["arm"], c["repeat"])
    assert key not in seen
    seen.add(key)
    assert sha(c["state_path"]) == c["state_sha256"]
    source = next(
        v
        for v in p["source_cases"]
        if (v["batch"], v["seed"], v["arm"]) == (c["batch"], c["seed"], c["arm"])
    )
    assert c["source"] == source["state_path"] and c["source_sha256"] == source[
        "state_sha256"
    ] == sha(c["source"])
    assert c["restored_exact"] and c["source_step"] == 12 and c["setup_ms"] > 0
    s = torch.load(c["state_path"], weights_only=True)
    if "q" in s["model"]:
        assert torch.equal(s["model"]["q"], fixed[c["seed"]][0])
        if c["arm"] == "fixed:reconstruct":
            assert torch.equal(s["model"]["theta"], fixed[c["seed"]][1])
    params = {
        k: v
        for k, v in s["model"].items()
        if k != "q" and not (c["arm"] == "fixed:reconstruct" and k == "theta")
    }
    buffers = {k: v for k, v in s["model"].items() if k not in params}
    n = sum(v.numel() for v in params.values())
    assert n == c["parameters"] == p["counts"][c["arm"]]["parameters"]
    assert (
        c["parameter_bytes"] == 4 * n
        and c["buffer_bytes"]
        == sum(v.numel() * v.element_size() for v in buffers.values())
        == p["counts"][c["arm"]]["buffer_bytes"]
    )
    assert c["macs"] == p["counts"][c["arm"]]["macs"]
    opt = s["optimizer"]["state"]
    assert len(opt) == len(params)
    assert c["optimizer_bytes"] == sum(
        v.numel() * v.element_size()
        for t in opt.values()
        for v in t.values()
        if isinstance(v, torch.Tensor)
    )
    assert all(
        float(t["step"]) == 42
        and all(bool(torch.isfinite(t[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
        for t in opt.values()
    )
    assert all(bool(torch.isfinite(v).all()) for v in s["model"].values())
    sample = data[c["batch"], c["seed"]]
    assert (
        tensor_hash(sample["x"]) == c["input_hash"] and tensor_hash(sample["y"]) == c["target_hash"]
    )
    assert [h["step"] for h in c["history"]] == list(range(1, 31))
    assert all(math.isfinite(v) for h in c["history"] for v in h.values())
    for metric in ("event_ms", "wall_ms"):
        values = [h[metric] for h in c["history"]][10:]
        assert min(values) > 0
        assert c["timing"][metric] == dict(
            mean=st.mean(values), median=st.median(values), sample_variance=st.variance(values)
        )
        halves = [st.median(values[:10]), st.median(values[10:])]
        assert c["stability"][metric] == max(halves) / min(halves)
    assert [d["step"] for d in c["diagnostics"]] == [1, 30]
    for d in c["diagnostics"]:
        assert math.isfinite(d["input_gradient_norm"]) and d["input_gradient_norm"] > 0
        assert set(d["postclip_parameter_norms"]) == set(params)
        assert all(math.isfinite(v) for v in d["postclip_parameter_norms"].values())
        assert sum(v * v for v in d["postclip_parameter_norms"].values()) <= 1.00001
    loss = score(s["model"], c["arm"], sample)
    error = abs(loss - c["final_loss"]) / max(abs(loss), 1e-12)
    assert error <= 1e-5
    checks.append(dict(index=index, independent_loss=loss, relative_error=error))
assert seen == {
    (b, s, a, rep) for b in p["batches"] for s in p["seeds"] for a in p["arms"] for rep in (0, 1)
}
aggregates = []
comparisons = []
repeat_states = []
for fixture, (batch, seed) in enumerate((b, s) for b in p["batches"] for s in p["seeds"]):
    shift = fixture % 3
    order = p["arms"][shift:] + p["arms"][:shift]
    assert [c["arm"] for c in r["cases"][6 * fixture : 6 * fixture + 6]] == order + list(
        reversed(order)
    )
    groups = {
        arm: [c for c in r["cases"] if (c["batch"], c["seed"], c["arm"]) == (batch, seed, arm)]
        for arm in p["arms"]
    }
    stats = {}
    for arm, cs in groups.items():
        assert len(cs) == 2
        timing = {
            metric: dict(
                mean=st.mean(c["history"][j][metric] for c in cs for j in range(10, 30)),
                median=st.median(c["history"][j][metric] for c in cs for j in range(10, 30)),
                sample_variance=st.variance(
                    c["history"][j][metric] for c in cs for j in range(10, 30)
                ),
            )
            for metric in ("event_ms", "wall_ms")
        }
        repeat_ratio = {
            metric: max(c["timing"][metric]["median"] for c in cs)
            / min(c["timing"][metric]["median"] for c in cs)
            for metric in timing
        }
        stats[arm] = dict(
            parameters=cs[0]["parameters"],
            peak_bytes=max(c["peak_bytes"] for c in cs),
            timing=timing,
            repeat_ratio=repeat_ratio,
            stable=max(*repeat_ratio.values(), *(v for c in cs for v in c["stability"].values()))
            <= 1.15,
        )
        aggregates.append(dict(batch=batch, seed=seed, arm=arm, **stats[arm]))
        s0, s1 = [torch.load(c["state_path"], weights_only=True)["model"] for c in cs]
        names = [k for k in s0 if k != "q"]
        numerator = sum((s0[k].double() - s1[k].double()).square().sum().item() for k in names)
        denominator = sum(s0[k].double().square().sum().item() for k in names)
        repeat_states.append(
            dict(
                batch=batch,
                seed=seed,
                arm=arm,
                parameter_relative_error=math.sqrt(numerator / denominator),
                loss_relative_error=abs(cs[0]["final_loss"] - cs[1]["final_loss"])
                / max(cs[0]["final_loss"], 1e-30),
            )
        )
    candidate = stats["learned:fused"]
    for control in ("full_gelu:checkpoint", "full_swiglu:checkpoint"):
        ref = stats[control]
        ratios = dict(
            parameters=candidate["parameters"] / ref["parameters"],
            peak=candidate["peak_bytes"] / ref["peak_bytes"],
            **{
                m: candidate["timing"][m]["median"] / ref["timing"][m]["median"]
                for m in ("event_ms", "wall_ms")
            },
        )
        gates = dict(
            parameters=ratios["parameters"] <= 0.8,
            peak=ratios["peak"] <= 0.9,
            timing=max(ratios["event_ms"], ratios["wall_ms"]) <= 1.15,
            stability=candidate["stable"] and ref["stable"],
        )
        comparisons.append(
            dict(
                batch=batch,
                seed=seed,
                control=control,
                ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
            )
        )
hashes(inputs)
hashes(p["maintained_files"])
assert not torch.cuda.is_initialized()
write_json(
    ROOT / "audit.json",
    dict(
        status="VERIFIED",
        checks=checks,
        aggregates=aggregates,
        repeat_states=repeat_states,
        comparisons=comparisons,
        passed=all(c["passed"] for c in comparisons),
        backwards=0,
        cuda_initialized=False,
    ),
)
print("36 saved-state scores verified; confirmation gate:", all(c["passed"] for c in comparisons))
