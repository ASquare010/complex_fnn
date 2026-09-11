"""Opt-in eager training memory helpers; ordinary model defaults are unchanged.

The classifier owns and reuses its logits/gradient buffers. Only first-order
FP32/FP64 hard-label mean cross entropy is supported. See
research/training_memory_usage.md for the measured scope and limitations.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F

from src.core.transformer import Block, Transformer


class _BufferCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets):
        # These temporaries are private: never overwrite caller-owned tensors.
        logp = F.linear(hidden, weight).reshape(-1, weight.shape[0])
        torch.log_softmax(logp, 1, out=logp)
        target = targets.reshape(-1)
        loss, total = torch.ops.aten.nll_loss_forward(logp, target, None, 1, -100)
        ctx.save_for_backward(hidden, weight, target, logp, total)
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hidden, weight, target, logp, total = ctx.saved_tensors
        grad = torch.ops.aten.nll_loss_backward(upstream, logp, target, None, 1, -100, total)
        torch.ops.aten._log_softmax_backward_data.out(grad, logp, 1, hidden.dtype, out=grad)
        flat = hidden.reshape(-1, hidden.shape[-1])
        column_major = flat.stride(0) == 1 and flat.stride(1) == flat.shape[0]
        dh = (weight.t() @ grad.t()).t() if column_major else grad @ weight
        return dh.reshape_as(hidden), grad.t() @ flat, None


def buffer_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Mean hard-label CE with ignore_index=-100 and no bias/class weights.

    Hidden is [tokens, features] or contiguous [batch, tokens, features];
    classifier weight is contiguous [classes, features]. Targets must have the
    hidden leading shape and int64 dtype. FP32/FP64 only, with autocast disabled.
    All-ignored targets retain native NaN loss semantics. No higher derivatives,
    torch.func, compiler or graph-capture support is claimed.
    """
    if hidden.dtype not in (torch.float32, torch.float64) or weight.dtype != hidden.dtype:
        raise ValueError("Expected matching FP32/FP64 hidden and weight")
    if torch.is_autocast_enabled(hidden.device.type):
        raise ValueError("Disable autocast for buffer_cross_entropy")
    if hidden.ndim not in (2, 3) or weight.ndim != 2:
        raise ValueError("Expected 2D/3D hidden and 2D weight")
    if not weight.is_contiguous() or (hidden.ndim == 3 and not hidden.is_contiguous()):
        raise ValueError("Weight and 3D hidden must be contiguous")
    if hidden.shape[-1] != weight.shape[1] or min(*hidden.shape, *weight.shape) <= 0:
        raise ValueError("Nonempty hidden/weight feature dimensions must agree")
    if targets.dtype != torch.int64 or targets.shape != hidden.shape[:-1]:
        raise ValueError("Expected int64 targets matching hidden leading shape")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("Hidden, weight and targets must share a device")
    return _BufferCrossEntropy.apply(hidden, weight, targets)


def buffer_model_loss(
    model: Transformer, tokens: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Explicit tied-classifier loss for the repository's ordinary Transformer.

    This bypasses Transformer.forward and its top-level forward hooks, exactly
    as the qualified research loss does. Embedding/block/norm hooks still run.
    """
    if type(model) is not Transformer:
        raise TypeError("buffer_model_loss requires the ordinary Transformer class")
    if tokens.ndim != 2 or tokens.shape[1] > model.config.context:
        raise ValueError("Expected [batch, length] within configured context")
    hidden = model.embedding(tokens)
    for block in model.blocks:
        hidden = block(hidden)
    return buffer_cross_entropy(model.norm(hidden), model.embedding.weight, targets)


@contextmanager
def offload_checkpoint_inputs(model: Transformer, last_n_blocks: int = 4) -> Iterator[None]:
    """Temporarily offload inputs saved by the final block checkpoints on CUDA.

    Set model.set_recompute_scope("block") first. Keep this context around both
    forward and backward; do not use the same model concurrently. CPU/eval/no-grad
    forwards bypass the transfer hooks. Module/parameter identities and state-dict
    paths stay unchanged. Existing per-instance forward overrides are rejected
    before installing any wrapper; nested use of the same blocks is rejected.
    """
    if type(model) is not Transformer:
        raise TypeError("Expected the ordinary Transformer class")
    if type(last_n_blocks) is not int or not 0 <= last_n_blocks <= len(model.blocks):
        raise ValueError("last_n_blocks must be an integer between zero and the block count")
    blocks = list(model.blocks)[len(model.blocks) - last_n_blocks :]
    for block in blocks:
        if type(block) is not Block or "forward" in block.__dict__:
            raise ValueError("Selected blocks must have their original forward methods")
        if block.recompute_scope != "block":
            raise ValueError("Enable whole-block recomputation before offloading inputs")
    installed = []

    def forward(this, x):
        if (
            x.is_cuda
            and this.training
            and torch.is_grad_enabled()
            and this.recompute_scope == "block"
        ):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return Block.forward(this, x)
        return Block.forward(this, x)

    try:
        for block in blocks:
            block.forward = MethodType(forward, block)
            installed.append(block)
        yield
    finally:
        for block in installed:
            del block.forward
