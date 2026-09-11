"""Freeze equations and verify the scalar inverse before GPU stress tests."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.reversible_softsign_v1.model import phi, inverse
from pathlib import Path

ROOT = Path("results/reversible_softsign_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/modulation_depth_v1/receipt.json")["files"])
base = read("results/modulation_depth_v1/protocol.json")
checks = []
for dtype, tol in [(torch.float32, 1e-6), (torch.float64, 1e-12)]:
    pos = torch.logspace(-12, 12, 481, dtype=dtype)
    x = torch.cat((-pos, torch.zeros(1, dtype=dtype), pos))
    for a in (-0.95, 0, 0.95):
        alpha = torch.tensor(a, dtype=dtype)
        y = phi(x, alpha)
        back = inverse(y, alpha)
        error = ((back - x).abs() / (1 + x.abs())).max().item()
        root = back.abs().double()
        b = 1 + a - y.abs().double()
        residual = (root.square() + b * root - y.abs().double()).abs() / (
            1 + root.square() + b.abs() * root + y.abs().double()
        )
        assert error <= tol and residual.max().item() <= tol
        checks.append(
            dict(dtype=str(dtype), a=a, error=error, quadratic_residual=residual.max().item())
        )
x = torch.tensor([-0.8, -0.03, 0.0, 0.2, 1.7], dtype=torch.float64, requires_grad=True)
a = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(phi, (x, a))
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
write_json(
    ROOT / "protocol.json",
    dict(
        study="H147",
        seeds=[131, 149, 167],
        depths=[8, 32, 128],
        strengths=[0.5, 2.0],
        dtypes=["float32", "float64"],
        cpu_checks=checks,
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/modulation_depth_v1/receipt.json"),
        sources={
            f.as_posix(): sha(f)
            for f in [*ROOT.glob("*.py"), Path("research/reversible_softsign_plan.md")]
        },
    ),
)
print("Scalar inverse, independent quadratic residual and gradcheck passed.")
