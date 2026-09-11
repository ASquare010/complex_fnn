"""Frozen CPU witnesses for the augmentation dimension argument."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path

ROOT = Path("results/augmentation_barrier_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/narrow_coupling_v1/receipt.json")["files"])
old = read("results/narrow_coupling_v1/protocol.json")
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
p = dict(
    study="H155",
    seeds=[389, 397, 409],
    pairs=[[1, 0], [8, 0], [8, 4], [8, 7], [32, 0], [32, 1], [32, 4], [32, 16], [32, 31]],
    radii=[0.5, 1.0, 2.0],
    readout_scales=[1.0, 0.01],
    d=32,
    training_updates=0,
    gpu_used=False,
    maintained_files=old["maintained_files"],
    prior_receipt=sha("results/narrow_coupling_v1/receipt.json"),
    sources={
        f.as_posix(): sha(f)
        for f in [*ROOT.glob("*.py"), Path("research/augmentation_barrier_plan.md")]
    },
)
write_json(ROOT / "protocol.json", p)
rows = []
examples = []
for seed in p["seeds"]:
    for q, k in p["pairs"]:
        gen = torch.Generator().manual_seed(seed + q * 100 + k * 10000)
        n = 32 + k
        Q = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64)).Q
        B = Q[q:, :32]
        _, _, vh = torch.linalg.svd(B, full_matrices=True)
        v = vh[-1]
        residual = (B @ v).norm().item()
        assert residual <= 1e-10 and abs(v.norm().item() - 1) <= 1e-10
        assert (Q.T @ Q - torch.eye(n, dtype=torch.float64)).abs().max() <= 1e-10
        cases = []
        for radius in p["radii"]:
            x = radius * v
            target = x * torch.roll(x, -1)
            for alpha in p["readout_scales"]:
                pred = alpha * (Q[:q, :32] @ x)
                errors = torch.stack(((pred - target[:q]).norm(), (-pred - target[:q]).norm()))
                bound = alpha * radius
                assert errors.max().item() + 1e-10 >= bound
                cases.append(
                    dict(
                        radius=radius,
                        alpha=alpha,
                        bound=bound,
                        errors=errors.tolist(),
                        output_separation=2 * pred.norm().item(),
                    )
                )
        rows.append(dict(seed=seed, q=q, k=k, kernel_residual=residual, cases=cases))
        examples.append(dict(Q=Q, v=v, seed=seed, q=q, k=k))
# Full affine span is an integer identity, not a numerical rank tolerance.
witness = torch.eye(32, dtype=torch.int64) + torch.roll(torch.eye(32, dtype=torch.int64), 1, dims=1)
products = witness * torch.roll(witness, -1, dims=1)
assert torch.equal(products, torch.eye(32, dtype=torch.int64))
assert torch.equal((-witness) * torch.roll(-witness, -1, dims=1), products)


def shear(state, a, b):
    x, z = state.chunk(2, dim=-1)
    return torch.cat((x, z + a * x * torch.roll(x, -1, dims=-1) + b), dim=-1)


torch.manual_seed(155)
s = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
a = torch.randn(4, dtype=torch.float64, requires_grad=True)
b = torch.randn(4, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(shear, (s, a, b), eps=1e-6, atol=1e-5, rtol=1e-3)
recovery = []
states = []
for dtype in [torch.float32, torch.float64]:
    for scale in [0.001, 1.0, 1000.0]:
        gen = torch.Generator().manual_seed(155)
        x = torch.randn(257, 32, generator=gen, dtype=dtype) * scale
        z = torch.randn(257, 32, generator=gen, dtype=dtype)
        state = torch.cat((x, z), dim=1)
        # Group the product identically in forward/inverse; rounding of z+f(x)
        # can still destroy small z when the product magnitude is large.
        product = x * torch.roll(x, -1, dims=1)
        out = torch.cat((x, z + product), dim=1)
        restored = torch.cat((out[:, :32], out[:, 32:] - product), dim=1)
        errors = [
            ((u.double() - v.double()).norm() / v.double().norm()).item()
            for u, v in zip(restored.chunk(2, dim=1), state.chunk(2, dim=1), strict=True)
        ]
        if dtype == torch.float64:
            assert max(errors) <= 1e-9
        with_zero = torch.cat((x, torch.zeros_like(z)), dim=1)
        exact = shear(with_zero, torch.ones(32, dtype=dtype), torch.zeros(32, dtype=dtype))[:, 32:]
        assert torch.equal(exact, product)
        recovery.append(dict(dtype=str(dtype), scale=scale, block_relative_errors=errors))
        states.append(
            dict(input=state, output=out, restored=restored, dtype=str(dtype), scale=scale)
        )
assert not torch.cuda.is_initialized()
torch.save(
    dict(examples=examples, recovery=states, witness=witness, products=products),
    ROOT / "witnesses.pt",
)
write_json(
    ROOT / "result.json",
    dict(
        rows=rows,
        recovery=recovery,
        passed=True,
        gradcheck=True,
        integer_span=True,
        training_updates=0,
        gpu_used=False,
        witnesses_sha256=sha(ROOT / "witnesses.pt"),
    ),
)
print("27 matrix witnesses, 162 scaled checks, exact span and six inverse stress cases complete.")
