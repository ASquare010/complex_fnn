"""H128: preserve native input-gradient matrix layout as well as reductions."""

import torch
from torch.autograd.function import once_differentiable

from results.native_buffer_loss_v1.operator import NativeBufferLoss


class LayoutBufferLoss(NativeBufferLoss):
    @staticmethod
    def forward(ctx, hidden, weight, targets):
        if not weight.is_contiguous() or hidden.ndim not in (2, 3):
            raise ValueError("Qualified scope: contiguous weights, 2D/3D hidden")
        if hidden.ndim == 3 and not hidden.is_contiguous():
            raise ValueError("3D hidden must be contiguous in this prototype")
        return NativeBufferLoss.forward(ctx, hidden, weight, targets)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        hidden, weight, target, logp, total = ctx.saved_tensors
        grad = torch.ops.aten.nll_loss_backward(upstream, logp, target, None, 1, -100, total)
        torch.ops.aten._log_softmax_backward_data.out(grad, logp, 1, hidden.dtype, out=grad)
        flat = hidden.reshape(-1, hidden.shape[-1])
        column_major = flat.stride(0) == 1 and flat.stride(1) == flat.shape[0]
        dh = (weight.t() @ grad.t()).t() if column_major else grad @ weight
        return dh.reshape_as(hidden), grad.t() @ flat, None


def model_loss(model, tokens, targets, policy, chunk=512):
    assert policy == "fp32_default_native"
    hidden = model.embedding(tokens)
    for block in model.blocks:
        hidden = block(hidden)
    return LayoutBufferLoss.apply(model.norm(hidden), model.embedding.weight, targets)
