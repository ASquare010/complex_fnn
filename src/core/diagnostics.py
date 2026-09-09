"""Small sampled layer diagnostics, kept outside timed training."""

import torch

from src.core.benchmark import autocast


@torch.no_grad()
def inspect_layers(model, x: torch.Tensor, device: str, precision: str) -> dict:
    records, handles = {}, []
    native_overrides = []
    current_layer = [-1]
    seen_ffns = set()
    seen_projections = set()

    def enter_layer(index):
        def enter(module, inputs):
            current_layer[0] = index

        return enter

    def hook(name):
        def collect(module, inputs, output):
            z = output.float()
            records[f"layer_{current_layer[0]}.{name}"] = {
                "mean": z.mean().item(),
                "std": z.std(unbiased=False).item(),
                "abs_max": z.abs().max().item(),
                "near_zero_fraction": (z.abs() < 1e-6).float().mean().item(),
                "finite": bool(torch.isfinite(z).all()),
                "input_rms": inputs[0].float().square().mean().sqrt().item(),
                "output_rms": z.square().mean().sqrt().item(),
            }

            shape = getattr(module, "shape_parameters", None)
            if callable(shape):
                records[f"layer_{current_layer[0]}.{name}"]["shape"] = shape()

        return collect

    for i, block in enumerate(model.blocks):
        if hasattr(block.ffn, "_native_forward"):
            native_overrides.append((block.ffn, block.ffn.forward))
            block.ffn.forward = block.ffn._native_forward
        handles.append(block.register_forward_pre_hook(enter_layer(i)))
        handles.append(block.register_forward_hook(hook("residual")))
        if id(block.ffn) not in seen_ffns:
            seen_ffns.add(id(block.ffn))
            handles.append(block.ffn.register_forward_hook(hook("ffn")))
        curve = getattr(block.ffn, "curve", None)
        if isinstance(curve, torch.nn.Module):
            handles.append(curve.register_forward_hook(hook("learnable_activation")))
        up = getattr(block.ffn, "up", None)
        if up is not None and id(up) not in seen_projections:
            seen_projections.add(id(up))
            handles.append(up.register_forward_hook(hook("preactivation")))
    try:
        with autocast(device, precision):
            model(x)
    finally:
        for handle in handles:
            handle.remove()
        for ffn, forward in native_overrides:
            ffn.forward = forward
    return records


def gradient_stats(model) -> dict:
    records = {}
    for i, block in enumerate(model.blocks):
        modules = [("attention", block.attn), ("ffn", block.ffn)]
        curve = getattr(block.ffn, "curve", None)
        if isinstance(curve, torch.nn.Module):
            modules.append(("activation", curve))
        for name, module in modules:
            grads = [
                p.grad.detach().float().square().sum()
                for p in module.parameters()
                if p.grad is not None
            ]
            records[f"layer_{i}.{name}"] = torch.stack(grads).sum().sqrt().item() if grads else 0.0
    return records
