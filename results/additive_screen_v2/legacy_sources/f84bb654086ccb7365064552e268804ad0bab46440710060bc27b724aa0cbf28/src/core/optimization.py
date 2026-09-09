"""Explicit optimizer groups for dense and structured projection parameters."""

import math

import torch

from src.blockshuffle_ffn import BlockShuffleLinear
from src.dense_ffn import DenseFFN
from src.grouped_ffn import GroupedLinear


def dense_width_ratio(model, config) -> float:
    if config.ffn_width_init_mode == config.ffn_width_lr_mode == "none":
        return 1.0
    if model.config.variant.startswith("shared_") or not all(
        isinstance(b.ffn, DenseFFN) for b in model.blocks
    ):
        raise ValueError("Width calibration currently supports unshared dense FFNs only")
    reference = (
        8 * model.config.width // 3 if "swiglu" in model.config.variant else 4 * model.config.width
    )
    return reference / model.config.ffn_width


@torch.no_grad()
def initialize_dense_width(model, config) -> dict:
    """Apply once, before training; checkpoint loading already restores scaled weights."""
    ratio = dense_width_ratio(model, config)
    gain = math.sqrt(ratio) if config.ffn_width_init_mode == "fan_in" else 1.0
    if gain != 1.0:
        for block in model.blocks:
            block.ffn.down.weight.mul_(gain)
    return {
        "reference_to_actual_hidden_ratio": ratio
        if config.ffn_width_init_mode != "none" or config.ffn_width_lr_mode != "none"
        else None,
        "initial_down_gain": gain,
    }


def parameter_groups(model, config) -> list[dict]:
    multipliers, factors = {}, {}
    for block in model.blocks:
        for module in block.ffn.modules():
            if isinstance(module, BlockShuffleLinear):
                for factor in (module.first, module.second):
                    multipliers[id(factor.weight)] = module.input_width / (
                        2 * factor.weight.shape[-1]
                    )
                    factors[id(factor.weight)] = 2
            elif isinstance(module, GroupedLinear):
                multipliers.setdefault(id(module.weight), float(module.groups))
                factors.setdefault(id(module.weight), 1)
    width_multipliers = {}
    ratio = dense_width_ratio(model, config)
    if config.ffn_width_lr_mode == "fan_in":
        for block in model.blocks:
            width_multipliers[id(block.ffn.down.weight)] = ratio
            factors[id(block.ffn.down.weight)] = 1
    grouped = {}
    for name, parameter in model.named_parameters():
        decay = config.weight_decay if parameter.ndim >= 2 and "theta" not in name else 0.0
        scale = multipliers.get(id(parameter), 1.0) if config.ffn_lr_mode == "fan_in" else 1.0
        scale *= width_multipliers.get(id(parameter), 1.0)
        if config.ffn_decay_mode == "product" and id(parameter) in factors:
            decay /= factors[id(parameter)] * scale
        group = grouped.setdefault(
            (decay, scale),
            {
                "params": [],
                "weight_decay": decay,
                "lr_scale": scale,
                "lr": config.learning_rate * scale,
            },
        )
        group["params"].append(parameter)
    return list(grouped.values())


def group_summary(groups: list[dict]) -> list[dict]:
    return [
        {
            "weight_decay": g["weight_decay"],
            "lr_scale": g["lr_scale"],
            "parameter_count": sum(p.numel() for p in g["params"]),
        }
        for g in groups
    ]
