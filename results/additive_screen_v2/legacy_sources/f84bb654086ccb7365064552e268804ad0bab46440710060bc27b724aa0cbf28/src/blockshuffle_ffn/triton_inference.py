"""Optional four-kernel BF16 inference for existing BlockShuffle factors.

Import this module only with a supported Triton installation. Training remains
in the native architecture; no learned factors are transformed or cached.
"""

import torch
import triton
import triton.language as tl
from torch import nn

from src.blockshuffle_ffn import BlockShuffleFFN


@triton.jit
def _paired_first(
    X,
    U,
    V,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    kk = tl.arange(0, BK)
    local: tl.constexpr = D // G
    au = tl.full((BM, BN), 0, tl.float32)
    av = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(local, BK)):
        k = start * BK + kk
        x = tl.load(
            X + rows[:, None] * D + group * local + k[None, :],
            (rows[:, None] < M) & (k[None, :] < local),
            0,
        ).to(tl.bfloat16)
        offsets = group * local * local + cols[None, :] * local + k[:, None]
        mask = (k[:, None] < local) & (cols[None, :] < local)
        u = tl.load(U + offsets, mask, 0).to(tl.bfloat16)
        v = tl.load(V + offsets, mask, 0).to(tl.bfloat16)
        au += tl.dot(x, u)
        av += tl.dot(x, v)
    # Direct shuffle: original group*local+col becomes col*G+group.
    offsets = rows[:, None] * D + cols[None, :] * G + group
    mask = (rows[:, None] < M) & (cols[None, :] < local)
    tl.store(Y + offsets, au.to(tl.bfloat16), mask)
    tl.store(Y + M * D + offsets, av.to(tl.bfloat16), mask)


@triton.jit
def _paired_second(
    X,
    U,
    V,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ni: tl.constexpr = D // G
    no: tl.constexpr = H // G
    au = tl.full((BM, BN), 0, tl.float32)
    av = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(ni, BK)):
        k = start * BK + kk
        offsets = rows[:, None] * D + group * ni + k[None, :]
        mask = (rows[:, None] < M) & (k[None, :] < ni)
        xu = tl.load(X + offsets, mask, 0)
        xv = tl.load(X + M * D + offsets, mask, 0)
        wo = group * no * ni + cols[None, :] * ni + k[:, None]
        wm = (k[:, None] < ni) & (cols[None, :] < no)
        u = tl.load(U + wo, wm, 0).to(tl.bfloat16)
        v = tl.load(V + wo, wm, 0).to(tl.bfloat16)
        au += tl.dot(xu, u)
        av += tl.dot(xv, v)
    # Preserve native BF16 rounding at both expansion and SiLU boundaries.
    u = au.to(tl.bfloat16).to(tl.float32)
    v = av.to(tl.bfloat16).to(tl.float32)
    silu = (u / (1 + tl.exp(-u))).to(tl.bfloat16).to(tl.float32)
    values = (silu * v).to(tl.bfloat16)
    raw = group * no + cols
    natural = (raw % G) * no + raw // G
    tl.store(
        Y + rows[:, None] * H + natural[None, :], values, (rows[:, None] < M) & (cols[None, :] < no)
    )


@triton.jit
def _down_factor(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    UNSHUFFLE: tl.constexpr,
):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ni: tl.constexpr = N // G
    no: tl.constexpr = OUTPUT_WIDTH // G
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(ni, BK)):
        k = start * BK + kk
        x = tl.load(
            X + rows[:, None] * N + group * ni + k[None, :],
            (rows[:, None] < M) & (k[None, :] < ni),
            0,
        )
        w = tl.load(
            W + group * no * ni + cols[None, :] * ni + k[:, None],
            (k[:, None] < ni) & (cols[None, :] < no),
            0,
        ).to(tl.bfloat16)
        acc += tl.dot(x, w)
    if UNSHUFFLE:
        raw = group * no + cols
        dest = (raw % G) * no + raw // G
    else:
        dest = cols * G + group
    tl.store(
        Y + rows[:, None] * OUTPUT_WIDTH + dest[None, :],
        acc.to(tl.bfloat16),
        (rows[:, None] < M) & (cols[None, :] < no),
    )


class TritonInferenceFFN(nn.Module):
    """Read existing FP32 factors; allocate only ephemeral BF16 intermediates."""

    def __init__(self, base: BlockShuffleFFN, token_tile: int = 32):
        super().__init__()
        if not isinstance(base, BlockShuffleFFN) or base.gate is None:
            raise ValueError("Four-kernel inference requires gated BlockShuffle")
        if base.up.output_width < base.up.input_width or token_tile not in (16, 32):
            raise ValueError("Expected expansion FFN and token tile16 or32")
        if any(p.dtype != torch.float32 for p in base.parameters()):
            raise ValueError("The inference path requires original FP32 factor storage")
        self.base = base
        self.width = base.up.input_width
        self.hidden = base.up.output_width
        self.groups = base.up.groups
        self.token_tile = token_tile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Triton FFN is inference-only; use eval() and no_grad()")
        if x.device.type != "cuda" or x.shape[-1] != self.width or not x.is_contiguous():
            raise ValueError("Expected contiguous CUDA input with the configured width")
        if x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Only FP32 or BF16 activations are supported")
        m = x.numel() // self.width
        d, h, g = self.width, self.hidden, self.groups
        first = torch.empty((2, m, d), device=x.device, dtype=torch.bfloat16)
        features = torch.empty((m, h), device=x.device, dtype=torch.bfloat16)
        middle = torch.empty((m, d), device=x.device, dtype=torch.bfloat16)
        output = torch.empty((*x.shape[:-1], d), device=x.device, dtype=torch.bfloat16)
        bm = self.token_tile
        bn = max(16, triton.next_power_of_2(d // g))
        grid = (triton.cdiv(m, bm), g)
        opts = {"num_warps": 4, "num_stages": 2, "enable_fp_fusion": False}
        _paired_first[grid](
            x,
            self.base.up.first.weight,
            self.base.gate.first.weight,
            first,
            m,
            d,
            g,
            bm,
            bn,
            32,
            **opts,
        )
        _paired_second[(grid[0], g, triton.cdiv(h // g, 64))](
            first,
            self.base.up.second.weight,
            self.base.gate.second.weight,
            features,
            m,
            d,
            h,
            g,
            bm,
            64,
            32,
            **opts,
        )
        _down_factor[grid](
            features, self.base.down.first.weight, middle, m, h, d, g, bm, bn, 32, False, **opts
        )
        _down_factor[grid](
            middle, self.base.down.second.weight, output, m, d, d, g, bm, bn, 32, True, **opts
        )
        return output


def replace_ffns(model, token_tile: int = 32) -> None:
    for block in model.blocks:
        block.ffn = TritonInferenceFFN(block.ffn, token_tile).eval()


def matrix_work(
    width: int, hidden: int, groups: int, tokens: int = 2048, token_tile: int = 32
) -> dict:
    """Scalar FMA counts implied by tile dimensions, not instruction counters."""
    local = width // groups
    square_n = max(16, triton.next_power_of_2(local))
    local_k = triton.cdiv(local, 32) * 32
    hidden_n = triton.cdiv(hidden // groups, 64) * 64
    hidden_k = triton.cdiv(hidden // groups, 32) * 32
    padded_tokens = triton.cdiv(tokens, token_tile) * token_tile
    padded = (
        2
        * groups
        * (
            2 * square_n * local_k
            + 2 * hidden_n * local_k
            + square_n * hidden_k
            + square_n * local_k
        )
    )
    return {
        "logical_matrix_flops_per_token_per_layer": 6 * width * (width + hidden) // groups,
        "padded_scalar_matrix_flops_per_token_per_layer": padded * padded_tokens / tokens,
        "ffn_kernel_launches_per_layer": 4,
        "persistent_cache_bytes": 0,
        "notes": "Counts matrix multiply-add as two FLOPs. Includes masked token/reduction/output tile work, excludes pointwise/memory costs and backend instruction scheduling.",
    }
