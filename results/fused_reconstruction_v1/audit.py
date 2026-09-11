"""Independently score saved states on CPU and verify the resource gate."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, tensor_hash
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import math
import statistics as st

ROOT = Path("results/fused_reconstruction_v1")
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
assert len(r["cases"]) == 60 and len(r["boundaries"]) == 61
assert r["training_updates"] == r["backwards"] == 720
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
    key = (c["batch"], c["seed"], c["arm"])
    assert key not in seen
    seen.add(key)
    assert sha(c["state_path"]) == c["state_sha256"]
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
        float(t["step"]) == 12
        and all(bool(torch.isfinite(t[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
        for t in opt.values()
    )
    assert all(bool(torch.isfinite(v).all()) for v in s["model"].values())
    sample = data[c["batch"], c["seed"]]
    assert (
        tensor_hash(sample["x"]) == c["input_hash"] and tensor_hash(sample["y"]) == c["target_hash"]
    )
    assert [h["step"] for h in c["history"]] == list(range(1, 13))
    assert all(math.isfinite(v) for h in c["history"] for v in h.values())
    for metric in ("event_ms", "wall_ms"):
        values = [h[metric] for h in c["history"]][4:]
        assert min(values) > 0
        assert c["timing"][metric] == dict(
            mean=st.mean(values), median=st.median(values), sample_variance=st.variance(values)
        )
        halves = [st.median(values[:4]), st.median(values[4:])]
        assert c["stability"][metric] == max(halves) / min(halves)
    assert [d["step"] for d in c["diagnostics"]] == [1, 12]
    for d in c["diagnostics"]:
        assert math.isfinite(d["input_gradient_norm"]) and d["input_gradient_norm"] > 0
        assert set(d["postclip_parameter_norms"]) == set(params)
        assert all(math.isfinite(v) for v in d["postclip_parameter_norms"].values())
        assert sum(v * v for v in d["postclip_parameter_norms"].values()) <= 1.00001
    loss = score(s["model"], c["arm"], sample)
    error = abs(loss - c["final_loss"]) / max(abs(loss), 1e-12)
    assert error <= 1e-5
    checks.append(dict(index=index, independent_loss=loss, relative_error=error))
assert seen == {(b, s, a) for b in p["batches"] for s in p["seeds"] for a in p["arms"]}
comparisons = []
for c in r["cases"]:
    if c["arm"] not in ("learned:fused",):
        continue
    for base_control in ("full_gelu", "full_swiglu"):
        control = base_control + ":checkpoint"
        ref = next(
            v
            for v in r["cases"]
            if (v["batch"], v["seed"], v["arm"]) == (c["batch"], c["seed"], control)
        )
        ratios = dict(
            parameters=c["parameters"] / ref["parameters"],
            peak=c["peak_bytes"] / ref["peak_bytes"],
            **{
                m: c["timing"][m]["median"] / ref["timing"][m]["median"]
                for m in ("event_ms", "wall_ms")
            },
        )
        gates = dict(
            parameters=ratios["parameters"] <= 0.8,
            peak=ratios["peak"] <= 0.9,
            timing=max(ratios["event_ms"], ratios["wall_ms"]) <= 1.15,
            stability=max(*c["stability"].values(), *ref["stability"].values()) <= 1.15,
        )
        comparisons.append(
            dict(
                arm=c["arm"],
                batch=c["batch"],
                seed=c["seed"],
                control=control,
                ratios=ratios,
                gates=gates,
                passed=all(gates.values()),
                control_stability=ref["stability"],
            )
        )
aggregates = []
for b in p["batches"]:
    for arm in p["arms"]:
        rows = [c for c in r["cases"] if c["batch"] == b and c["arm"] == arm]
        fields = {
            "peak_mib": [c["peak_bytes"] / 2**20 for c in rows],
            "event_ms": [c["timing"]["event_ms"]["median"] for c in rows],
            "wall_ms": [c["timing"]["wall_ms"]["median"] for c in rows],
            "final_loss": [c["final_loss"] for c in rows],
        }
        aggregates.append(
            dict(
                batch=b,
                arm=arm,
                statistics={
                    k: dict(mean=st.mean(v), median=st.median(v), sample_variance=st.variance(v))
                    for k, v in fields.items()
                },
            )
        )
execution_pairs = []
for batch in p["batches"]:
    for seed in p["seeds"]:
        eager = next(
            c
            for c in r["cases"]
            if (c["batch"], c["seed"], c["arm"]) == (batch, seed, "learned:reconstruct")
        )
        ref = torch.load(eager["state_path"], weights_only=True)["model"]
        for mode in ("checkpoint", "fused"):
            c = next(
                c
                for c in r["cases"]
                if (c["batch"], c["seed"], c["arm"]) == (batch, seed, "learned:" + mode)
            )
            state = torch.load(c["state_path"], weights_only=True)["model"]
            delta = torch.cat(
                [(state[k].double() - ref[k].double()).flatten() for k in ("theta", "bias")]
            )
            denom = torch.cat([ref[k].double().flatten() for k in ("theta", "bias")]).norm()
            execution_pairs.append(
                dict(
                    batch=batch,
                    seed=seed,
                    mode=mode,
                    parameter_relative_error=(delta.norm() / denom).item(),
                    loss_relative_error=abs(c["final_loss"] - eager["final_loss"])
                    / eager["final_loss"],
                )
            )
write_json(ROOT / "execution_pairs.json", execution_pairs)
hashes(inputs)
hashes(p["maintained_files"])
assert not torch.cuda.is_initialized()
write_json(
    ROOT / "audit.json",
    dict(
        status="VERIFIED",
        checks=checks,
        comparisons=comparisons,
        aggregates=aggregates,
        candidates={
            arm: all(c["passed"] for c in comparisons if c["arm"] == arm)
            for arm in ("learned:fused",)
        },
        torch_version=torch.__version__,
        cuda_build=torch.version.cuda,
        cuda_initialized=False,
        backward_calls=0,
    ),
)
print(
    "60 independent FP64 scores verified; resource decisions:",
    {arm: all(c["passed"] for c in comparisons if c["arm"] == arm) for arm in ("learned:fused",)},
)
