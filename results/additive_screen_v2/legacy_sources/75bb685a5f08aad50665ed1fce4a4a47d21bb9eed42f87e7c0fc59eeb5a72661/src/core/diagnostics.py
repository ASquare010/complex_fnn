"""Small sampled layer diagnostics, kept outside timed training."""

import torch

from src.bezier_ffn import QuadraticBezierActivation
from src.core.benchmark import autocast


@torch.no_grad()
def inspect_layers(model, x: torch.Tensor, device: str, precision: str) -> dict:
    records, handles = {}, []
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

            if isinstance(module, QuadraticBezierActivation):
                t = (
                    inputs[0]
                    .float()
                    .reshape(-1, module.groups, module.hidden // module.groups)
                    .sigmoid()
                )
                c = module.controls()
                slope = (
                    2
                    * ((1 - t) * c[:, 0, None] + t * (c[:, 1, None] - c[:, 0, None]))
                    * t
                    * (1 - t)
                )
                if module.residual:
                    raw = inputs[0].float().reshape_as(t)
                    slope = slope + t + raw * t * (1 - t)
                records[f"layer_{current_layer[0]}.{name}"].update(
                    sigmoid_tail_fraction=((t < 0.01) | (t > 0.99)).float().mean().item(),
                    activation_slope_abs_mean=slope.abs().mean().item(),
                    activation_slope_below_1e_minus3_fraction=(slope.abs() < 0.001)
                    .float()
                    .mean()
                    .item(),
                    controls=c.detach().tolist(),
                    control_saturation_fraction=(module.theta.tanh().abs() > 0.98)
                    .float()
                    .mean()
                    .item(),
                )
            if getattr(module, "mix_theta", None) is not None:
                records[f"layer_{current_layer[0]}.{name}"]["branch_product_coefficients"] = (
                    (0.5 * module.mix_theta.tanh()).detach().tolist()
                )

        return collect

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_pre_hook(enter_layer(i)))
        handles.append(block.register_forward_hook(hook("residual")))
        if id(block.ffn) not in seen_ffns:
            seen_ffns.add(id(block.ffn))
            handles.append(block.ffn.register_forward_hook(hook("ffn")))
        curve = getattr(block.ffn, "curve", None)
        if isinstance(curve, QuadraticBezierActivation):
            handles.append(curve.register_forward_hook(hook("quadratic_activation")))
        if id(block.ffn.up) not in seen_projections:
            seen_projections.add(id(block.ffn.up))
            handles.append(block.ffn.up.register_forward_hook(hook("preactivation")))
    try:
        with autocast(device, precision):
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return records


def gradient_stats(model) -> dict:
    records = {}
    for i, block in enumerate(model.blocks):
        for name, module in (("attention", block.attn), ("ffn", block.ffn)):
            grads = [
                p.grad.detach().float().square().sum()
                for p in module.parameters()
                if p.grad is not None
            ]
            records[f"layer_{i}.{name}"] = torch.stack(grads).sum().sqrt().item() if grads else 0.0
    return records
