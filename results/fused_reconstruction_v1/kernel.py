"""Fixed-tile FP32 inverse/VJP fusion with bounded-range numerical qualification."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _local(
    Y,
    G,
    T,
    B,
    P,
    DZ,
    PT,
    PB,
    N: tl.constexpr,
    D: tl.constexpr,
    RHO: tl.constexpr,
    PRE: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (rr[:, None] < N) & (cc[None, :] < D)
    y = tl.load(Y + rr[:, None] * D + cc[None, :], mask, other=0)
    g = tl.load(G + rr[:, None] * D + cc[None, :], mask, other=0)
    t = libdevice.tanh(tl.load(T + cc, cc < D, other=0))
    a = RHO * t
    bias = tl.load(B + cc, cc < D, other=0)
    v = tl.abs(y)
    b = 1 + a[None, :] - v
    disc = tl.sqrt(b * b + 4 * v)
    denom = tl.where(b >= 0, disc + b, 1.0)
    root = tl.where(b >= 0, 2 * v / denom, 0.5 * (disc - b))
    z = tl.where(y > 0, root, tl.where(y < 0, -root, 0.0))
    inv = 1 / (1 + tl.abs(z))
    dz = g * (1 + a[None, :] * inv * inv)
    dt = g * z * inv * (RHO * (1 - t * t))[None, :]
    if PRE:
        tl.store(P + rr[:, None] * D + cc[None, :], z - bias[None, :], mask)
    tl.store(DZ + rr[:, None] * D + cc[None, :], dz, mask)
    tl.store(PT + tl.program_id(0) * D + cc, tl.sum(dt, axis=0), cc < D)
    tl.store(PB + tl.program_id(0) * D + cc, tl.sum(dz, axis=0), cc < D)


@torch.no_grad()
def local(y, g, theta, bias, rho, gt, gb, need_pre=True):
    assert y.is_cuda and y.dtype == torch.float32 and y.ndim == 2
    assert all(
        t.is_contiguous() and t.device == y.device and t.dtype == y.dtype
        for t in (y, g, theta, bias, gt, gb)
    )
    n, d = y.shape
    assert g.shape == y.shape and theta.shape == bias.shape == gt.shape == gb.shape == (d,)
    rows = triton.cdiv(n, 64)
    pre = torch.empty_like(y) if need_pre else None
    dz = torch.empty_like(y)
    pt = torch.empty((rows, d), device=y.device)
    pb = torch.empty_like(pt)
    _local[(rows, triton.cdiv(d, 32))](
        y,
        g,
        theta,
        bias,
        pre if need_pre else dz,
        dz,
        pt,
        pb,
        n,
        d,
        rho,
        need_pre,
        64,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    torch.sum(pt, dim=0, out=gt)
    torch.sum(pb, dim=0, out=gb)
    return pre, dz
