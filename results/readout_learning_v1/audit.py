"""Independent explicit-formula scoring, validation-only selection and paired gates."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.readout_learning_v1.data import make_data
from pathlib import Path
import math
import statistics as st

ROOT = Path("results/readout_learning_v1")
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
    len(r["cases"]) == 198
    and len(r["boundaries"]) == 199
    and r["training_updates"] == r["backwards"] == 59400
)
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])
data = {}
for task in p["tasks"]:
    for seed in p["seeds"]:
        s = torch.load(ROOT / f"data_{task}_{seed}.pt", weights_only=True)
        expected = make_data(task, seed)
        assert all(torch.equal(v, expected[k]) for k, v in s.items())
        data[task, seed] = s


def predict(s, x, arm):
    if arm == "linear":
        return x @ s["weight"].T + s["bias"]
    if arm in ("learned", "fixed", "affine"):
        for j in range(4):
            x = x @ s["core.q"][j].T + s["core.bias"][j]
            a = 0.5 * s["core.theta"][j].tanh()
            x = x * (1 + a) if arm == "affine" else x + a * x / (1 + x.abs())
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


fixed = {}
for seed in p["seeds"]:
    gen = torch.Generator().manual_seed(seed + 4)
    q = torch.linalg.qr(torch.randn(4, 32, 32, generator=gen, dtype=torch.float64)).Q.float()
    theta = torch.empty(4, 32, dtype=torch.float64).uniform_(-1.5, 1.5, generator=gen).float()
    fixed[seed] = (q, theta)
checks = []
seen = set()
for i, c in enumerate(r["cases"]):
    assert c == read(ROOT / f"case{i:03d}.json") and c["index"] == i and c["steps"] == 300
    key = (c["task"], c["seed"], c["arm"], c["rate"])
    assert key not in seen
    seen.add(key)
    assert sha(c["state_path"]) == c["state_sha256"]
    state = torch.load(c["state_path"], weights_only=True)
    s = state["model"]
    if "core.q" in s:
        assert torch.equal(s["core.q"], fixed[c["seed"]][0])
        if c["arm"] == "fixed":
            assert torch.equal(s["core.theta"], fixed[c["seed"]][1])
    for group in state["optimizer"]["param_groups"]:
        assert (
            group["lr"] == c["rate"]
            and tuple(group["betas"]) == (0.9, 0.95)
            and group["weight_decay"] == 0
        )
    buffers = {
        k: v for k, v in s.items() if k == "core.q" or (c["arm"] == "fixed" and k == "core.theta")
    }
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
    moments = state["optimizer"]["state"]
    assert len(moments) == len(params) and all(
        float(v["step"]) == 300
        and all(bool(torch.isfinite(v[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
        for v in moments.values()
    )
    assert [h["step"] for h in c["history"]] == list(range(25, 301, 25))
    assert all(math.isfinite(v) for h in c["history"] for v in h.values())
    d = data[c["task"], c["seed"]]
    assert math.isclose(
        c["zero_reporting_mse"], d["y"][6144:].double().square().mean().item(), rel_tol=1e-12
    )
    s = {k: v.double() for k, v in s.items()}
    errors = {}
    for name, lo, hi in [("validation_mse", 4096, 6144), ("reporting_mse", 6144, 8192)]:
        value = score(s, d["x"][lo:hi], d["y"][lo:hi], c["arm"])
        error = abs(value - c[name]) / max(value, 1e-30)
        assert error <= 1e-5, (i, name, error, value, c[name])
        errors[name] = error
    checks.append(dict(index=i, relative_errors=errors))
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
                parameters=cases[0]["parameters"],
                buffer_bytes=cases[0]["buffer_bytes"],
                cases=[c["index"] for c in cases],
            )
        )
        for c in cases:
            lookup[task, arm, c["seed"]] = c
comparisons = []
qualifications = {}
for task in p["tasks"]:
    qualifications[task] = all(
        lookup[task, arm, s]["reporting_mse"] <= 0.8 * lookup[task, arm, s]["zero_reporting_mse"]
        for arm in ("gelu", "swiglu")
        for s in p["seeds"]
    )
    for seed in p["seeds"]:
        candidate = lookup[task, "learned", seed]["reporting_mse"]
        ratios = {
            arm: candidate / lookup[task, arm, seed]["reporting_mse"]
            for arm in ("gelu", "swiglu", "budget_gelu", "fixed", "affine")
        }
        gates = {
            arm: v <= (0.95 if task != "linear" and arm in ("fixed", "affine") else 1.01)
            for arm, v in ratios.items()
            if task != "linear" or arm not in ("fixed", "affine")
        }
        comparisons.append(
            dict(
                task=task,
                seed=seed,
                qualified=qualifications[task],
                ratios=ratios,
                gates=gates,
                passed=qualifications[task] and all(gates.values()),
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
        selected=selected,
        qualifications=qualifications,
        comparisons=comparisons,
        promoted=all(v["passed"] for v in comparisons),
        backwards=0,
        cuda_initialized=False,
    ),
)
print(
    "198 saved states/396 scores verified. Qualified tasks:",
    qualifications,
    "Promotion:",
    all(v["passed"] for v in comparisons),
)
