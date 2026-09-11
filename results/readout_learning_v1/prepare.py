"""Freeze the full learning budget, verify linear capacity and readout gradients."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.readout_learning_v1.model import Model, ARMS
from results.readout_learning_v1.data import TASKS
from pathlib import Path

ROOT = Path("results/readout_learning_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/fixed_tail_capacity_v1/receipt.json")["files"])
base = read("results/fixed_tail_capacity_v1/protocol.json")
counts = {
    arm: dict(
        parameters=sum(v.numel() for v in Model(arm, 263).parameters()),
        buffer_bytes=sum(v.numel() * v.element_size() for v in Model(arm, 263).buffers()),
    )
    for arm in ARMS
}
assert counts["learned"]["parameters"] == counts["affine"]["parameters"] == 1312
assert counts["fixed"]["parameters"] == 1184 and counts["budget_gelu"]["parameters"] == 1168
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
old = read("results/fused_reconstruction_v1/protocol.json")
sources = {
    **old["sources"],
    **{
        f.as_posix(): sha(f)
        for f in [*ROOT.glob("*.py"), Path("research/readout_learning_plan.md")]
    },
}
write_json(
    ROOT / "protocol.json",
    dict(
        study="H152",
        arms=ARMS,
        tasks=TASKS,
        seeds=[263, 277, 293],
        rates=[0.001, 0.003],
        steps=300,
        counts=counts,
        training_updates=59400,
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/fixed_tail_capacity_v1/receipt.json"),
        sources=sources,
    ),
)
torch.manual_seed(152)
m = Model("learned", 257).double()
with torch.no_grad():
    m.core.theta.zero_()
    A = torch.eye(32, dtype=torch.float64)
    for q in m.core.q:
        A = q @ A
    target = torch.randn(32, 32, dtype=torch.float64)
    m.readout.weight.copy_(torch.linalg.solve(A.T, target.T).T)
    x = torch.randn(257, 32, dtype=torch.float64)
    torch.testing.assert_close(m(x), x @ target.T, rtol=1e-10, atol=1e-11)
linear_error = (m(x) - x @ target.T).abs().max().item()
# Same rounded parameter values, but an independent CPU FP64 native-autograd path.
native = Model("learned", 257).double()
gpu = Model("learned", 257).cuda()
z = torch.randn(17, 32, dtype=torch.float32)
xc = z.double().requires_grad_()
xg = z.cuda().requires_grad_()
yc = native(xc)
yg = gpu(xg)
gc = torch.autograd.grad(yc.square().mean(), (xc, *native.parameters()))
gg = torch.autograd.grad(yg.square().mean(), (xg, *gpu.parameters()))
errors = [
    ((g.detach().cpu().double() - c).norm() / c.norm().clamp_min(1e-30)).item()
    for c, g in zip(gc, gg, strict=True)
]
assert max(errors) <= 1e-4 and torch.isfinite(yg).all()
del gpu, xg, yg, gg
clean = boundary()
write_json(
    ROOT / "checks.json",
    dict(
        linear_witness_max_error=linear_error,
        gradient_relative_errors=errors,
        gpu_backwards=1,
        cpu_backwards=1,
        passed=True,
        final_boundary=clean,
    ),
)
print("Linear-target witness and readout/core/input gradient checks passed.")
