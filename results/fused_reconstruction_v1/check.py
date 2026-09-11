"""GPU fusion qualification against independent FP64 local formulas and native VJPs."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from results.ordinary_long_training_v1.io import write_json
from results.fused_reconstruction_v1.kernel import local
from results.fused_reconstruction_v1.model import Model
from pathlib import Path
import time
import triton

ROOT = Path("results/fused_reconstruction_v1")
p = read(ROOT / "protocol.json")
hashes(p["sources"])
start = time.perf_counter()
bounds = [boundary()]
checks = []
for n, d in [(1, 1), (7, 33), (128, 384), (2048, 384)]:
    gen = torch.Generator().manual_seed(217 + n + d)
    z = torch.logspace(-12, 12, n * d, dtype=torch.float64).reshape(n, d)
    z[::2] *= -1
    z[0, 0] = 0
    t = torch.empty(d, dtype=torch.float64).uniform_(-1.5, 1.5, generator=gen).float().double()
    b = torch.randn(d, generator=gen, dtype=torch.float64).float().double()
    g = torch.randn(n, d, generator=gen, dtype=torch.float64).float().double()
    a = 0.25 * t.tanh()
    y = (z + a * z / (1 + z.abs())).float()
    # Recompute inverse in FP64 from the rounded input actually passed to GPU.
    v = y.double().abs()
    beta = 1 + a - v
    disc = torch.hypot(beta, 2 * v.sqrt())
    root = torch.where(
        beta >= 0,
        2 * v / torch.where(beta >= 0, disc + beta, torch.ones_like(beta)),
        0.5 * (disc - beta),
    )
    zz = y.double().sign() * root
    dz = g * (1 + a / (1 + zz.abs()).square())
    expected = [
        zz - b,
        dz,
        (g * zz / (1 + zz.abs())).sum(0) * 0.25 * (1 - t.tanh().square()),
        dz.sum(0),
    ]
    tt, bb = t.float().cuda(), b.float().cuda()
    gt, gb = torch.empty_like(tt), torch.empty_like(bb)
    pre, dgrad = local(y.cuda(), g.float().cuda(), tt, bb, 0.25, gt, gb)
    actual = [pre, dgrad, gt, gb]
    errors = []
    for u, v in zip(actual, expected, strict=True):
        u = u.cpu().double()
        e = dict(
            relative=((u - v).norm() / v.norm().clamp_min(1e-30)).item(),
            scaled=((u - v).abs() / (1 + v.abs())).max().item(),
        )
        assert torch.isfinite(u).all() and e["relative"] <= 5e-5 and e["scaled"] <= 1e-4
        errors.append(e)
    checks.append(dict(shape=[n, d], errors=errors))
    del tt, bb, gt, gb, pre, dgrad, actual
    bounds.append(boundary())
for n in (128, 2048):
    native = Model("learned:reconstruct", seed=217).cuda()
    fused = Model("learned:fused", seed=217).cuda()
    x = (
        torch.randn(n, 384, generator=torch.Generator().manual_seed(217 + n))
        .cuda()
        .requires_grad_()
    )
    u = x.detach().clone().requires_grad_()
    yn, yf = native(x), fused(u)
    assert torch.equal(yn, yf)
    gn = torch.autograd.grad(yn.square().mean(), (x, *native.parameters()))
    gf = torch.autograd.grad(yf.square().mean(), (u, *fused.parameters()))
    errors = [
        ((b - a).double().norm() / a.double().norm().clamp_min(1e-30)).item()
        for a, b in zip(gn, gf, strict=True)
    ]
    assert max(errors) <= 1e-4 and all(bool(torch.isfinite(v).all()) for v in gf)
    checks.append(dict(full_batch=n, relative_gradient_errors=errors))
    del native, fused, x, u, yn, yf, gn, gf
    bounds.append(boundary())
hashes(p["sources"])
write_json(
    ROOT / "checks.json",
    dict(
        passed=True,
        checks=checks,
        boundaries=bounds,
        setup_and_checks_seconds=time.perf_counter() - start,
        backwards=4,
        optimizer_updates=0,
        triton_version=triton.__version__,
    ),
)
print("All local and complete-stack fusion checks passed.")
