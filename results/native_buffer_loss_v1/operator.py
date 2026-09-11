"""Own classifier temporaries; reuse native log-softmax output buffers."""

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


class NativeBufferLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets):
        if hidden.dtype not in (torch.float32, torch.float64) or weight.dtype != hidden.dtype:
            raise ValueError("Only matching FP32/FP64 inputs are qualified")
        if torch.is_autocast_enabled(hidden.device.type):
            raise ValueError("Autocast is outside this experiment")
        # The buffer is privately allocated here: caller tensors are never overwritten.
        logp = F.linear(hidden, weight).reshape(-1, weight.shape[0])
        torch.log_softmax(logp, 1, out=logp)
        target = targets.reshape(-1)
        loss, total = torch.ops.aten.nll_loss_forward(logp, target, None, 1, -100)
        ctx.save_for_backward(hidden, weight, target, logp, total)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        hidden, weight, target, logp, total = ctx.saved_tensors
        grad = torch.ops.aten.nll_loss_backward(upstream, logp, target, None, 1, -100, total)
        torch.ops.aten._log_softmax_backward_data.out(grad, logp, 1, hidden.dtype, out=grad)
        flat = hidden.reshape(-1, hidden.shape[-1])
        return (grad @ weight).reshape_as(hidden), grad.t() @ flat, None


def model_loss(model, tokens, targets, policy, chunk=512):
    assert policy == "fp32_default_native"
    hidden = model.embedding(tokens)
    for block in model.blocks:
        hidden = block(hidden)
    return NativeBufferLoss.apply(model.norm(hidden), model.embedding.weight, targets)
