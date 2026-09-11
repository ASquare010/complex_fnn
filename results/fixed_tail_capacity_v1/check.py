"""Freeze and test a fixed-tail decomposition on saved models, using CPU only."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import math

ROOT = Path("results/fixed_tail_capacity_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/fused_timing_confirmation_v1/receipt.json")["files"])
base = read("results/fused_timing_confirmation_v1/protocol.json")
sources = [
    c
    for c in read("results/fused_timing_confirmation_v1/result.json")["cases"]
    if c["arm"] == "learned:fused" and c["repeat"] == 0
]
assert len(sources) == 6
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
p = dict(
    study="H151",
    models=sources,
    scales=[1, 4, 8],
    maintained_files=base["maintained_files"],
    prior_receipt=sha("results/fused_timing_confirmation_v1/receipt.json"),
    sources={
        f.as_posix(): sha(f)
        for f in [*ROOT.glob("*.py"), Path("research/fixed_tail_capacity_plan.md")]
    },
)
write_json(ROOT / "protocol.json", p)
cases = []
for i, c in enumerate(sources):
    assert sha(c["state_path"]) == c["state_sha256"]
    s = {k: v.double() for k, v in torch.load(c["state_path"], weights_only=True)["model"].items()}
    q, t, b = s["q"], s["theta"], s["bias"]
    shape = 0.25 * t.tanh()
    d = 384
    A = torch.eye(d, dtype=torch.float64)
    beta = torch.zeros(d, dtype=torch.float64)
    normbounds = []
    for j in range(8):
        A = q[j] @ A
        beta = q[j] @ beta + b[j]
        normbounds.append(
            math.sqrt(1 + (q[j].T @ q[j] - torch.eye(d, dtype=torch.float64)).norm().item() + 1e-12)
        )
    tail = 1.0
    universal = actual = 0.0
    for j in reversed(range(8)):
        universal += tail * 0.25 * math.sqrt(d)
        actual += tail * shape[j].norm().item()
        tail *= normbounds[j]
    # This reproduces H149's target construction before casting to FP64.
    target = torch.linalg.qr(
        torch.randn(d, d, generator=torch.Generator().manual_seed(60000 + c["seed"]))
    ).Q.T.double()
    x0 = torch.randn(
        256,
        d,
        generator=torch.Generator().manual_seed(281 + c["seed"] + c["batch"]),
        dtype=torch.float64,
    )
    payload = dict(A=A, beta=beta, target=target, x0=x0, shape=shape, q=q)
    results = []
    for scale in p["scales"]:
        x = x0 * scale
        y = x
        for j in range(8):
            z = y @ q[j].T + b[j]
            y = z + shape[j] * z / (1 + z.abs())
        residual = y - x @ A.T - beta
        pointwise = residual.norm(dim=1).max().item()
        assert pointwise <= actual + 1e-9 and torch.isfinite(y).all()
        assert (y - x @ A.T).norm(dim=1).max().item() <= beta.norm().item() + actual + 1e-9
        payload[f"output_{scale}"] = y
        for name, M in [("independent", target), ("opposite", -A)]:
            centered = x - x.mean(0)
            mismatch = (centered @ (A - M).T).square().sum(1).mean().sqrt().item()
            lower = max(mismatch - universal, 0) ** 2
            target_y = x @ M.T
            mse = (y - target_y).square().sum(1).mean().item()
            energy = target_y.square().sum(1).mean().item()
            assert mse + 1e-8 >= lower
            results.append(
                dict(
                    scale=scale,
                    target=name,
                    mse=mse,
                    energy=energy,
                    lower=lower,
                    normalized_lower=lower / energy,
                    normalized_mse=mse / energy,
                    population_lower=max(scale * (A - M).norm().item() - universal, 0) ** 2,
                    pointwise_residual_max=pointwise,
                )
            )
    path = ROOT / f"model{i}.pt"
    torch.save(payload, path)
    cases.append(
        dict(
            index=i,
            seed=c["seed"],
            source_batch=c["batch"],
            universal_bound=universal,
            actual_shape_bound=actual,
            normbounds=normbounds,
            beta_norm=beta.norm().item(),
            path=path.as_posix(),
            sha256=sha(path),
            results=results,
        )
    )
    print(
        i,
        c["seed"],
        "scale8 independent lower",
        next(
            v["normalized_lower"]
            for v in results
            if v["scale"] == 8 and v["target"] == "independent"
        ),
        flush=True,
    )
# Exact covariance and mismatch calculations use integers only.
d = 16
points = 16 * torch.cat((torch.eye(d, dtype=torch.int64), -torch.eye(d, dtype=torch.int64)))
assert torch.equal(points.sum(0), torch.zeros(d, dtype=torch.int64))
assert torch.equal(points.T @ points, 512 * torch.eye(d, dtype=torch.int64))
assert int((2 * points).square().sum()) // 32 == 1024
witness = dict(
    d=16,
    L=8,
    c=2,
    sigma=4,
    mismatch_rms=32,
    correction_bound=8,
    mse_lower=576,
    target_energy=256,
    normalized_lower=2.25,
)
assert not torch.cuda.is_initialized()
hashes(p["sources"])
hashes(p["maintained_files"])
write_json(
    ROOT / "result.json",
    dict(cases=cases, witness=witness, training_updates=0, backwards=0, cuda_initialized=False),
)
print("Fixed-tail decomposition and36 risk comparisons verified; exact witness lower2.25.")
