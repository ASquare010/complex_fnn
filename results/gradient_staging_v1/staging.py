"""Pinned gradient staging; no optimizer or parameter mutation."""

import torch


class GradientStager:
    def __init__(self, model):
        self.buffers = {
            name: torch.empty_like(p, device="cpu", pin_memory=True)
            for name, p in model.named_parameters()
        }
        self.seen = []
        self.handles = []
        for name, p in model.named_parameters():

            def stage(parameter, key=name):
                assert key not in self.seen, "Multiple accumulations require a different protocol"
                self.seen.append(key)
                self.buffers[key].copy_(parameter.grad, non_blocking=True)
                parameter.grad = None

            self.handles.append(p.register_post_accumulate_grad_hook(stage))

    def begin(self):
        self.seen.clear()

    def restore(self, model):
        assert set(self.seen) == set(self.buffers)
        # The caller synchronizes backward/D2H before CPU access and restoration.
        for name, p in model.named_parameters():
            assert p.grad is None
            p.grad = self.buffers[name].to(p.device, non_blocking=True)

    def close(self):
        for h in self.handles:
            h.remove()
