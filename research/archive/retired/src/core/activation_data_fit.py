"""Fit affine corrections on actual training-prefix rational preactivations."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch

from src.core.activation_screen import CACHE
from src.core.benchmark import autocast
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer


def fit_affine(z, correction):
    """CPU FP64 centered least squares, plus uncentered correction energy."""
    z, correction = z.double().cpu(), correction.double().cpu()
    zm, cm = z.mean(-1, keepdim=True), correction.mean(-1, keepdim=True)
    centered = z - zm
    variance = centered.square().sum(-1, keepdim=True)
    assert (variance > 0).all()
    gain = (centered * (correction - cm)).sum(-1, keepdim=True) / variance
    bias = cm - gain * zm
    error = correction - (gain * z + bias)
    return {
        "gain": gain.flatten().tolist(),
        "bias": bias.flatten().tolist(),
        "squared_error_by_group": error.square().sum(-1).tolist(),
        "correction_energy_by_group": correction.square().sum(-1).tolist(),
        "z_mean_by_group": zm.flatten().tolist(),
        "z_std_by_group": z.std(-1, correction=0).tolist(),
        "z_min_by_group": z.min(-1).values.tolist(),
        "z_max_by_group": z.max(-1).values.tolist(),
        "samples_per_group": z.shape[-1],
    }


def audit(output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    selection = json.loads(Path("results/activation_screen_v1/result.json").read_text())
    path = Path("results/runs") / selection["selected"]["blockshuffle_rational"]
    metrics = json.loads((path / "metrics.json").read_text())
    checkpoint_hash = sha256(path / "checkpoint.pt")
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**metrics["model"]), 17).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    data = TokenData(CACHE, "cuda", 10017)
    assert data.manifest["files"] == metrics["data"]["files"]
    x = data.train[:1024].reshape(8, 128)
    layers, handles = [], []

    def collect(index):
        def hook(curve, args):
            u = args[0].detach()
            groups = curve.groups
            grouped = u.reshape(-1, groups, curve.hidden // groups)
            samples = grouped.permute(1, 0, 2).reshape(groups, -1)
            stride = math.ceil(samples.shape[-1] / 65536)
            samples = samples[:, ::stride].float()
            correction = curve.residual(samples)
            assert torch.isfinite(samples).all() and torch.isfinite(correction).all()
            layers.append({"layer": index, "stride": stride, **fit_affine(samples, correction)})

        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.ffn.curve.register_forward_pre_hook(collect(index)))
    try:
        with torch.no_grad(), autocast("cuda", "bf16"):
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert len(layers) == 8 and sha256(path / "checkpoint.pt") == checkpoint_hash
    squared_error = sum(sum(r["squared_error_by_group"]) for r in layers)
    energy = sum(sum(r["correction_energy_by_group"]) for r in layers)
    count = sum(len(r["gain"]) * r["samples_per_group"] for r in layers)
    result = {
        "run": path.name,
        "checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_unchanged": True,
        "plan_sha256": sha256(Path("research/affine_activation_plan.md")),
        "data_hashes": data.manifest["files"],
        "tokens": 1024,
        "token_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        "sample_count": count,
        "affine_energy_fraction": 1 - squared_error / energy,
        "correction_rms": math.sqrt(energy / count),
        "affine_fit_error_rms": math.sqrt(squared_error / count),
        "layers": layers,
        "environment": environment(),
        "provenance": provenance(),
        "scope": "First eight consecutive training windows, deterministic channel stride; actual BF16 projection inputs and FP32 corrections before cast. Uncentered energy, not centered R-squared. No labels, updates, validation tuning or population representativeness claim.",
    }
    write_json(output / "result.json", result)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in result["provenance"]["source_files"]:
            archive.write(name, name)
    print(json.dumps({k: v for k, v in result.items() if k not in ("layers", "provenance")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
