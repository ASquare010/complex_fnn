"""Bounded CUDA reconstruction/VJP stress; no optimization or learning claim."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.reversible_softsign_v1.model import forward, reverse, phi
from pathlib import Path

ROOT = Path("results/reversible_softsign_v1")


def relative(a, b):
    return ((a - b).double().norm() / b.double().norm().clamp_min(1e-30)).item()


def one(depth, c, seed, dtype, index):
    gen = torch.Generator().manual_seed(seed + depth)
    q = (
        torch.linalg.qr(torch.randn(depth, 16, 16, generator=gen, dtype=torch.float64))
        .Q.to(dtype)
        .cuda()
    )
    theta = (
        torch.empty(depth, 16, dtype=torch.float64)
        .uniform_(-1.5, 1.5, generator=gen)
        .to(dtype)
        .cuda()
        .requires_grad_()
    )
    bias = (
        (0.1 * torch.randn(depth, 16, generator=gen, dtype=torch.float64))
        .to(dtype)
        .cuda()
        .requires_grad_()
    )
    x = torch.randn(32, 16, generator=gen, dtype=torch.float64).to(dtype).cuda().requires_grad_()
    g = torch.randn(32, 16, generator=gen, dtype=torch.float64).to(dtype).cuda()
    rho = c / depth
    y = forward(x, q, theta, bias, rho)
    native = torch.autograd.grad(y, (x, theta, bias), g)
    rec, gx, gt, gb = reverse(y.detach(), g, q, theta.detach(), bias.detach(), rho)
    with torch.no_grad():
        cycle = forward(rec, q, theta, bias, rho)
        z = x[:1].clone()
        jac = torch.eye(16, device="cuda", dtype=dtype)
        for j in range(depth):
            z = z @ q[j].T + bias[j]
            a = rho * theta[j].tanh()
            slope = (1 + a / (1 + z.abs()).square()).squeeze(0)
            jac = (slope[:, None] * q[j]) @ jac
            z = phi(z, a)
        singular = torch.linalg.svdvals(jac.double())
    errors = dict(
        input=relative(rec, x),
        cycle=relative(cycle, y),
        gradient=relative(
            torch.cat([t.flatten() for t in (gx, gt, gb)]), torch.cat([t.flatten() for t in native])
        ),
    )
    finite = all(bool(torch.isfinite(t).all()) for t in (y, rec, gx, gt, gb, *native))
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    lower, upper = (1 - rho) ** depth, (1 + rho) ** depth
    spectral = singular.min().item() >= lower * (1 - tol) and singular.max().item() <= upper * (
        1 + tol
    )
    path = ROOT / f"case{index:02d}.pt"
    torch.save(
        {
            k: v.detach().cpu()
            for k, v in dict(
                x=x,
                q=q,
                theta=theta,
                bias=bias,
                g=g,
                y=y,
                reconstructed=rec,
                native_x=native[0],
                native_theta=native[1],
                native_bias=native[2],
                reverse_x=gx,
                reverse_theta=gt,
                reverse_bias=gb,
            ).items()
        },
        path,
    )
    row = dict(
        index=index,
        depth=depth,
        c=c,
        seed=seed,
        dtype=str(dtype),
        errors=errors,
        finite=finite,
        spectral_pass=spectral,
        singular_min=singular.min().item(),
        singular_max=singular.max().item(),
        lower=lower,
        upper=upper,
        constant_target_mse_floor_per_dim=lower**2,
        parameters=2 * depth * 16,
        fixed_matrix_bytes=q.numel() * q.element_size(),
        peak_bytes=torch.cuda.max_memory_allocated(),
        passed=finite and spectral and max(errors.values()) <= tol,
        path=path.as_posix(),
        sha256=sha(path),
    )
    write_json(ROOT / f"case{index:02d}.json", row)
    return row


def run():
    p = read(ROOT / "protocol.json")
    hashes(p["sources"])
    hashes(p["maintained_files"])
    bounds = [boundary()]
    cases = []
    for dtype in p["dtypes"]:
        for depth in p["depths"]:
            for c in p["strengths"]:
                for seed in p["seeds"]:
                    r = one(depth, c, seed, getattr(torch, dtype), len(cases))
                    cases.append(r)
                    bounds.append(boundary())
                    print(r["index"], dtype, depth, c, r["errors"], r["passed"], flush=True)
    hashes(p["sources"])
    write_json(
        ROOT / "result.json",
        dict(
            cases=cases,
            boundaries=bounds,
            training_updates=0,
            native_backwards=36,
            passed=all(c["passed"] for c in cases),
        ),
    )


if __name__ == "__main__":
    run()
