"""Independent explicit-formula CPU scoring and validation-only selection."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.narrow_coupling_v1.data import make_data
from results.narrow_coupling_v1.model import Model
from pathlib import Path
import math
import statistics as st

ROOT = Path("results/narrow_coupling_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
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
    dict(inputs=inputs, scorer="CPU FP64 explicit formula, batches257", relative_tolerance=1e-5),
)
assert (
    len(r["cases"]) == 120
    and len(r["boundaries"]) == 121
    and r["training_updates"] == r["backwards"] == 72000
)
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])
data = {}
for task in p["tasks"]:
    for seed in p["seeds"]:
        d = torch.load(ROOT / f"data_{task}_{seed}.pt", weights_only=True)
        expected = make_data(task, seed)
        assert all(torch.equal(v, expected[k]) for k, v in d.items())
        data[task, seed] = d


def predict(s, x, arm):
    if "coupling" in arm:
        for j in range(4):
            ab = x[:, s["permutations"][j]]
            a, b = ab[:, :16], ab[:, 16:]
            z = b @ s[f"up.{j}.weight"].T + s[f"up.{j}.bias"]
            if arm != "affine_coupling":
                z = 0.5 * z * (1 + torch.erf(z / math.sqrt(2)))
            v = (z @ s[f"down.{j}.weight"].T + s[f"down.{j}.bias"]) / 2
            x = torch.cat((a + v, b), dim=1)[:, s["inverse_permutations"][j]]
        return x @ s["readout.weight"].T + s["readout.bias"]
    for j in range(4):
        v = {
            k.removeprefix(f"blocks.{j}."): t for k, t in s.items() if k.startswith(f"blocks.{j}.")
        }
        z = x @ v["up.weight"].T + v["up.bias"]
        if arm == "relu":
            z = z.clamp_min(0)
        elif arm in ("leaky_relu", "prelu"):
            z = torch.where(z >= 0, z, z * (0.01 if arm == "leaky_relu" else v["act.weight"]))
        elif arm in ("silu", "swiglu"):
            z = z * torch.sigmoid(z)
        else:
            z = 0.5 * z * (1 + torch.erf(z / math.sqrt(2)))
        if arm == "swiglu":
            z = z * (x @ v["gate.weight"].T + v["gate.bias"])
        x = x + (z @ v["down.weight"].T + v["down.bias"]) / 2
    return x


def score(s, x, y, arm):
    total = 0.0
    for start in range(0, len(x), 257):
        total += (
            (predict(s, x[start : start + 257].double(), arm) - y[start : start + 257].double())
            .square()
            .sum()
            .item()
        )
    return total / y.numel()


checks = []
seen = set()
for i, c in enumerate(r["cases"]):
    assert c == read(ROOT / f"case{i:03d}.json") and c["index"] == i and c["steps"] == 600
    key = (c["task"], c["seed"], c["arm"], c["rate"])
    assert key not in seen
    seen.add(key)
    assert sha(c["state_path"]) == c["state_sha256"]
    state = torch.load(c["state_path"], weights_only=True)
    s = state["model"]
    initial = Model(c["arm"], c["seed"])
    buffers = dict(initial.named_buffers())
    assert all(torch.equal(s[k], v) for k, v in buffers.items())
    params = {k: v for k, v in s.items() if k not in buffers}
    assert (
        sum(v.numel() for v in params.values())
        == c["parameters"]
        == p["counts"][c["arm"]]["parameters"]
    )
    assert (
        sum(v.numel() * v.element_size() for v in buffers.values())
        == c["buffer_bytes"]
        == p["counts"][c["arm"]]["buffer_bytes"]
    )
    assert all(bool(torch.isfinite(v).all()) for v in s.values())
    for group in state["optimizer"]["param_groups"]:
        assert (
            group["lr"] == c["rate"]
            and tuple(group["betas"]) == (0.9, 0.95)
            and group["weight_decay"] == 0
        )
    moments = state["optimizer"]["state"]
    assert len(moments) == len(params) and all(
        float(v["step"]) == 600
        and all(bool(torch.isfinite(v[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
        for v in moments.values()
    )
    assert [h["step"] for h in c["history"]] == list(range(50, 601, 50))
    assert all(math.isfinite(v) for h in c["history"] for v in h.values())
    d = data[c["task"], c["seed"]]
    assert math.isclose(
        c["zero_reporting_mse"], d["y"][6144:].double().square().mean().item(), rel_tol=1e-12
    )
    s = {k: (v.double() if v.is_floating_point() else v) for k, v in s.items()}
    errors = {}
    for name, lo, hi in [("validation_mse", 4096, 6144), ("reporting_mse", 6144, 8192)]:
        val = score(s, d["x"][lo:hi], d["y"][lo:hi], c["arm"])
        error = abs(val - c[name]) / max(val, 1e-30)
        assert error <= 1e-5, (i, name, error)
        errors[name] = error
    checks.append(dict(index=i, relative_errors=errors))
    if i % 20 == 0:
        print("audited", i, flush=True)
assert seen == {
    (t, s, a, lr) for t in p["tasks"] for s in p["seeds"] for a in p["arms"] for lr in p["rates"]
}
selected = []
lookup = {}
for task in p["tasks"]:
    for arm in p["arms"]:
        rates = {
            lr: st.mean(
                c["validation_mse"]
                for c in r["cases"]
                if (c["task"], c["arm"], c["rate"]) == (task, arm, lr)
            )
            for lr in p["rates"]
        }
        rate = min(p["rates"], key=lambda lr: (rates[lr], lr))
        cases = [c for c in r["cases"] if (c["task"], c["arm"], c["rate"]) == (task, arm, rate)]
        values = [c["reporting_mse"] for c in cases]
        selected.append(
            dict(
                task=task,
                arm=arm,
                rate=rate,
                validation_means=rates,
                reporting=dict(
                    mean=st.mean(values),
                    median=st.median(values),
                    sample_variance=st.variance(values),
                ),
                case_indices=[c["index"] for c in cases],
            )
        )
        for c in cases:
            lookup[task, c["seed"], arm] = c
comparisons = []
tasks = {}
for task in p["tasks"]:
    qualified = all(
        lookup[task, s, a]["reporting_mse"] <= 0.8 * lookup[task, s, a]["zero_reporting_mse"]
        for s in p["seeds"]
        for a in ("gelu", "swiglu")
    )
    passes = []
    for seed in p["seeds"]:
        c = lookup[task, seed, "coupling"]
        ratios = {
            a: c["reporting_mse"] / lookup[task, seed, a]["reporting_mse"]
            for a in ("gelu", "swiglu", "budget_gelu", "fixed_coupling", "affine_coupling")
        }
        passed = all(ratios[a] <= 1.01 for a in ("gelu", "swiglu", "budget_gelu")) and all(
            ratios[a] <= 0.95 for a in ("fixed_coupling", "affine_coupling")
        )
        passes.append(passed)
        comparisons.append(dict(task=task, seed=seed, ratios=ratios, passed=passed))
    tasks[task] = dict(qualified=qualified, passed=qualified and all(passes))
assert not torch.cuda.is_initialized()
hashes(inputs)
write_json(
    ROOT / "audit.json",
    dict(
        status="EVIDENCE_VERIFIED",
        checks=checks,
        selected=selected,
        comparisons=comparisons,
        tasks=tasks,
        passed=all(v["passed"] for v in tasks.values()),
        scored_states=len(checks),
        score_count=2 * len(checks),
    ),
)
print(tasks, flush=True)
