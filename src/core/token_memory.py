"""Token-local checkpoint helpers; no parameters or approximate gradients.

Only use ``chunked_ffn`` for deterministic, token-independent modules. Training
with dropout, cross-token normalization, or side-effectful forwards needs its
own qualification. These helpers do not alter the model's default execution.
"""

from collections.abc import Callable

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


def chunked_ffn(
    forward: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Evaluate/checkpoint independent token rows, preserving their original order."""
    _validate_chunk_size(chunk_size)
    if x.ndim < 2 or x.numel() == 0:
        raise ValueError("Expected nonempty token rows with a feature dimension")
    rows = x.reshape(-1, x.shape[-1])
    chunks = []
    for part in rows.split(chunk_size):
        if torch.is_grad_enabled():
            y = checkpoint(forward, part, use_reentrant=False, preserve_rng_state=True)
        else:
            y = forward(part)
        chunks.append(y)
    result = torch.cat(chunks, dim=0)
    return result.reshape(*x.shape[:-1], result.shape[-1])


def chunked_linear_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Mean CE with bounded token-by-vocabulary intermediates, including -100 masks.

    Weight is [vocabulary, hidden]. Each chunk uses sum reduction, then the total is
    divided by the global number of unmasked targets. Averaging chunk means would
    incorrectly overweight an uneven final chunk. All-ignored targets return NaN,
    matching mean-reduced PyTorch CE. This operation does not shift target tokens.
    """
    _validate_chunk_size(chunk_size)
    if hidden.ndim < 2 or hidden.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("Expected nonempty targets matching hidden token dimensions")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("Expected [vocabulary, hidden] classifier weight")
    rows = hidden.reshape(-1, hidden.shape[-1])
    labels = targets.reshape(-1)

    def sum_loss(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, w)
        # Preserve double precision for mathematical qualification.
        logits = logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
        return F.cross_entropy(logits, y, reduction="sum")

    losses = []
    for x, y in zip(rows.split(chunk_size), labels.split(chunk_size), strict=True):
        if torch.is_grad_enabled():
            loss = checkpoint(sum_loss, x, weight, y, use_reentrant=False, preserve_rng_state=True)
        else:
            loss = sum_loss(x, weight, y)
        losses.append(loss)
    return torch.stack(losses).sum() / (labels != -100).sum()
