"""Scoped research adapter; restore gradients before existing clipping/Adam code."""

import types
from contextlib import ExitStack
from unittest.mock import patch

import torch

from results.checkpoint_input_offload_v1.source.offload import install as offload_inputs
from results.compact_rmsnorm_v1.norm import CompactRMS
from results.gradient_staging_v1.staging import GradientStager
from src.core.transformer import RMSNorm


class Adapter:
    def __init__(self, loop, combined, loss_scale=1):
        self.loop, self.combined, self.loss_scale = loop, combined, loss_scale
        self.model = self.store = self.current_loss = self.restore_inputs = None
        self.restorations = []
        self.stack = ExitStack()

    def __enter__(self):
        factory, loss_fn, backward = (
            self.loop.Transformer,
            self.loop.training_loss,
            torch.Tensor.backward,
        )

        def create(*args, **kwargs):
            model = factory(*args, **kwargs)
            self.model = model
            # H120 moves the returned model to CUDA after construction. Allocate
            # pinned buffers lazily in the first training-loss call.
            self.restore_inputs = None
            if self.combined:

                def forward(this, x):
                    if this.training and torch.is_grad_enabled() and x.dtype == torch.float32:
                        return CompactRMS.apply(x, this.weight)
                    return RMSNorm.forward(this, x)

                for module in model.modules():
                    if isinstance(module, RMSNorm):
                        module.forward = types.MethodType(forward, module)
            return model

        def loss(model, *args, **kwargs):
            if self.restore_inputs is None:
                self.restore_inputs = offload_inputs(model, True)
            if self.combined:
                if self.store is None:
                    self.store = GradientStager(model)
                self.store.begin()
            value = (
                loss_fn(model, *args, **kwargs) * self.loss_scale
                if self.loss_scale != 1
                else loss_fn(model, *args, **kwargs)
            )
            self.current_loss = value
            return value

        def run_backward(value, *args, **kwargs):
            result = backward(value, *args, **kwargs)
            if value is self.current_loss:
                if self.store is not None:
                    assert all(p.grad is None for p in self.model.parameters())
                    torch.cuda.synchronize()
                    self.store.restore(self.model)
                    self.restorations.append(
                        dict(
                            hooks=len(self.store.seen),
                            restored_bytes=sum(
                                p.grad.numel() * p.grad.element_size()
                                for p in self.model.parameters()
                            ),
                        )
                    )
                self.current_loss = None
            return result

        self.stack.enter_context(patch.object(self.loop, "Transformer", create))
        self.stack.enter_context(patch.object(self.loop, "training_loss", loss))
        self.stack.enter_context(patch.object(torch.Tensor, "backward", run_backward))
        return self

    def __exit__(self, *exc):
        self.stack.__exit__(*exc)
        if self.store is not None:
            self.store.close()
        if self.restore_inputs is not None:
            self.restore_inputs()
        self.model = self.store = self.current_loss = self.restore_inputs = None
