"""Post-training projection spectra and invocation-specific gradient diagnostics."""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.blockshuffle_ffn import BlockShuffleLinear
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.optimization import initialize_dense_width
from src.core.reproducibility import provenance, sha256, write_json
from src.core.transformer import Transformer
from src.grouped_ffn import GroupedLinear


def gradient_flow(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Loss-direction activation gradients; shared parameters are not conflated with uses."""
    captured, handles, seen_ffn, seen_up = {}, [], set(), set()
    layer = [-1]
    was_training = model.training

    def enter(index):
        def hook(module, inputs):
            layer[0] = index

        return hook

    def collect_ffn(module, inputs, output):
        captured[f"layer_{layer[0]}.input"] = inputs[0]
        captured[f"layer_{layer[0]}.output"] = output

    def collect_up(module, inputs, output):
        captured[f"layer_{layer[0]}.preactivation"] = output

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_pre_hook(enter(i)))
        if id(block.ffn) not in seen_ffn:
            seen_ffn.add(id(block.ffn))
            handles.append(block.ffn.register_forward_hook(collect_ffn))
        if id(block.ffn.up) not in seen_up:
            seen_up.add(id(block.ffn.up))
            handles.append(block.ffn.up.register_forward_hook(collect_up))
    model.eval()
    try:
        with torch.enable_grad():
            loss = model.loss(x, y)
            gradients = torch.autograd.grad(loss, tuple(captured.values()))
        records = {}
        for (name, value), gradient in zip(captured.items(), gradients, strict=True):
            records[name] = {
                "activation_rms": value.detach().float().square().mean().sqrt().item(),
                "gradient_rms": gradient.detach().float().square().mean().sqrt().item(),
                "gradient_abs_max": gradient.detach().abs().max().item(),
                "finite": bool(torch.isfinite(value).all() and torch.isfinite(gradient).all()),
            }
        gains = {
            f"layer_{i}": records[f"layer_{i}.input"]["gradient_rms"]
            / max(records[f"layer_{i}.output"]["gradient_rms"], 1e-30)
            for i in range(len(model.blocks))
        }
        return {
            "batch_nll": loss.item(),
            "target_tokens": y.numel(),
            "precision": "fp32",
            "records": records,
            "ffn_loss_direction_backward_gain": gains,
            "interpretation": "One fixed-batch loss direction. Gradients are distinct activation uses, including shared FFNs. Ratios are not extremal singular values, global conditioning or a trained-depth stability guarantee.",
        }
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


@torch.no_grad()
def spectra(model) -> dict:
    records, aliases, seen = {}, {}, {}
    for i, block in enumerate(model.blocks):
        ffn = getattr(block.ffn, "base", block.ffn)
        for label in ("up", "gate", "down"):
            module = getattr(ffn, label, None)
            if module is None:
                continue
            name = f"layer_{i}.{label}"
            if id(module) in seen:
                aliases[name] = seen[id(module)]
                continue
            seen[id(module)] = name
            if isinstance(module, BlockShuffleLinear):
                weight = module(torch.eye(module.input_width)).T
            elif isinstance(module, GroupedLinear):
                weight = torch.block_diag(*module.weight.unbind())
            elif isinstance(module, nn.Linear):
                weight = module.weight
            else:
                raise TypeError(f"Unknown projection type {type(module)}")
            singular = torch.linalg.svdvals(weight.double())
            records[name] = {
                "shape": list(weight.shape),
                "singular_values": singular.tolist(),
                "condition_nonzero": (singular[0] / singular[-1]).item()
                if singular[-1] > 0
                else None,
                "stable_rank": (singular.square().sum() / singular[0].square()).item(),
                "rank_relative_1e_6": int((singular > singular[0] * 1e-6).sum()),
                "maximum_possible_rank": min(weight.shape),
            }
    return {"projections": records, "shared_aliases": aliases}


def audit(run: Path, output: Path, cache: Path = Path("data/tinystories_v1")) -> dict:
    if output.exists():
        raise ValueError("Do not overwrite model audit evidence")
    torch.set_num_threads(4)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = Transformer(config, checkpoint["training_config"]["seed"])
    initialize_dense_width(model, TrainConfig(**checkpoint["training_config"]))
    data = TokenData(cache, "cpu", 10017)
    x, y = next(data.validation(2, config.context, 1))
    initial = {"spectra": spectra(model), "gradient_flow": gradient_flow(model, x, y)}
    model.load_state_dict(checkpoint["model"])
    final = {"spectra": spectra(model), "gradient_flow": gradient_flow(model, x, y)}
    result = {
        "run": run.name,
        "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "data_hashes": data.manifest["files"],
        "provenance": provenance(),
        "initial": initial,
        "final": final,
        "limitations": "CPU FP32 audit, one held-out batch, and projection matrices rather than full Transformer Jacobians. Does not establish robustness at larger depth or other data.",
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
