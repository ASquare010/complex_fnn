"""Freeze the recipe then check native coupling mathematics and gradients."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.narrow_coupling_v1.model import Model, ARMS, Coupling
from results.narrow_coupling_v1.data import TASKS
from pathlib import Path
from torch.func import functional_call

ROOT = Path("results/narrow_coupling_v1")
assert not (ROOT / "protocol.json").exists()
prior = read("results/even_tangent_v1/receipt.json")
hashes(prior["files"])
base = read("results/even_tangent_v1/protocol_dtype64.json")
counts = {
    a: dict(
        parameters=sum(v.numel() for v in Model(a, 347).parameters()),
        buffer_bytes=sum(v.numel() * v.element_size() for v in Model(a, 347).buffers()),
    )
    for a in ARMS
}
assert (
    counts["coupling"]["parameters"] == 2176
    and counts["gelu"]["parameters"] == 8448
    and counts["budget_gelu"]["parameters"] == 2208
)
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
previous = read("results/readout_learning_v1/protocol.json")
sources = {
    **previous["sources"],
    **{
        f.as_posix(): sha(f)
        for f in [
            *ROOT.glob("*.py"),
            Path("research/narrow_coupling_plan.md"),
            Path("results/readout_learning_v1/model.py"),
            Path("results/readout_learning_v1/data.py"),
        ]
    },
}
write_json(
    ROOT / "protocol.json",
    dict(
        study="H154",
        arms=ARMS,
        tasks=TASKS,
        seeds=[347, 359, 373],
        rates=[0.001, 0.003],
        steps=600,
        counts=counts,
        training_updates=72000,
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/even_tangent_v1/receipt.json"),
        sources=sources,
    ),
)
torch.manual_seed(154)
m = Coupling("coupling", 341, d=4, depth=2).double()
x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)
# Explicit finite differences cover parameters too, without mutating leaf data.


params = dict(m.named_parameters())
assert torch.autograd.gradcheck(
    lambda *values: functional_call(m, dict(zip(params, values, strict=True)), (x,)),
    tuple(params.values()),
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
)
inverse_errors = []
for seed in [347, 359, 373]:
    m = Model("coupling", seed).double()
    for scale in [0.1, 1.0, 10.0]:
        x = torch.randn(17, 32, dtype=torch.float64) * scale
        err = (m.inverse_core(m.core(x)) - x).abs().max().item()
        assert err <= 1e-10
        inverse_errors.append(err)
    fixed = Model("fixed_coupling", seed).double()
    torch.testing.assert_close(m(x), fixed(x), rtol=0, atol=0)
    with torch.no_grad():
        for layer in m.down:
            layer.weight.zero_()
            layer.bias.zero_()
        target = torch.randn(32, 32, dtype=torch.float64)
        m.readout.weight.copy_(target)
        torch.testing.assert_close(m(x), x @ target.T, rtol=1e-12, atol=1e-12)
checks = []
for arm in ["coupling", "fixed_coupling", "affine_coupling"]:
    cpu = Model(arm, 347).double()
    gpu = Model(arm, 347).cuda()
    z = torch.randn(17, 32)
    xc, xg = z.double().requires_grad_(), z.cuda().requires_grad_()
    yc, yg = cpu(xc), gpu(xg)
    gc = torch.autograd.grad(yc.square().mean(), (xc, *cpu.parameters()))
    gg = torch.autograd.grad(yg.square().mean(), (xg, *gpu.parameters()))
    errors = [
        ((g.cpu().double() - c).norm() / c.norm().clamp_min(1e-30)).item()
        for c, g in zip(gc, gg, strict=True)
    ]
    assert max(errors) <= 1e-4
    checks.append(dict(arm=arm, relative_errors=errors))
    del gpu, xg, yg, gg
    boundary()
write_json(
    ROOT / "checks.json",
    dict(
        passed=True,
        inverse_errors=inverse_errors,
        gradient_checks=checks,
        gpu_backwards=3,
        final_boundary=boundary(),
    ),
)
print("Inverse, input/parameter finite differences, linear witness and GPU gradient checks passed.")
