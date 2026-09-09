"""Controlled post-training intervention on square down-projection factors."""

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import torch

from src.core.benchmark import evaluate_forward
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.model_audit import gradient_flow
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer


def floor_square_blocks(weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """Floor global block singular values and preserve aggregate Frobenius norm.

    This offline diagnostic computes in FP64 and returns the original dtype.
    It is not an optimizer or a differentiable training parametrization.
    """
    if weight.ndim != 3 or weight.shape[-1] != weight.shape[-2]:
        raise ValueError("Expected grouped square blocks")
    if not 0 <= alpha <= 1 or not weight.is_floating_point():
        raise ValueError("Floating weights and alpha in [0,1] are required")
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
        if s.norm() == 0:
            return weight.clone()
        adjusted = s.clamp_min(alpha * s.max())
        adjusted = adjusted * (s.norm() / adjusted.norm())
        return ((u * adjusted.unsqueeze(-2)) @ vh).to(weight.dtype)


def matched_random(weight: torch.Tensor, target: torch.Tensor, seed: int) -> torch.Tensor:
    """Match per-block edit norm, using one fixed direction per seed."""
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    noise = torch.randn(
        weight.shape, generator=generator, device=weight.device, dtype=torch.float64
    )
    delta_norm = (target.double() - weight.double()).norm(dim=(-2, -1), keepdim=True)
    noise = noise * (delta_norm / noise.norm(dim=(-2, -1), keepdim=True))
    return (weight.double() + noise).to(weight.dtype)


def spectrum(weight: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(weight.double())
    return {
        "singular_values": s.tolist(),
        "condition_nonzero": (s.max() / s.min()).item() if s.min() > 0 else None,
        "frobenius_norm": s.norm().item(),
    }


@torch.no_grad()
def down_spectra(model) -> dict:
    records = {}
    for i, block in enumerate(model.blocks):
        down = copy.deepcopy(block.ffn.down).double()
        full = down(torch.eye(down.input_width, dtype=torch.float64)).T
        records[f"layer_{i}"] = {
            "rectangular_factor": spectrum(down.first.weight),
            "square_factor": spectrum(down.second.weight),
            "projection": spectrum(full),
        }
    return records


def audit(run: Path, output: Path, replication: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_hash = sha256(run / "checkpoint.pt")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    if config.variant != "blockshuffle_swiglu":
        raise ValueError("This frozen intervention is specific to BlockShuffle SwiGLU")
    if replication and (
        config.width != 192
        or config.layers != 4
        or config.hidden != 1024
        or checkpoint["training_config"]["seed"] not in (17, 29, 43)
    ):
        raise ValueError("Replication requires the frozen smaller-model cohort")
    model = Transformer(config, checkpoint["training_config"]["seed"]).eval()
    prov = provenance()
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write("research/conditioning_plan.md", "research/conditioning_plan.md")
        if replication:
            archive.write(
                "research/conditioning_replication_plan.md",
                "research/conditioning_replication_plan.md",
            )
    data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
    batches = list(data.validation(16, config.context, 16))
    cpu_data = TokenData(Path("data/tinystories_v1"), "cpu", 10017)
    x, y = next(cpu_data.validation(2, config.context, 1))
    protocol = {
        "run": run.name,
        "replication": replication,
        "replication_plan_sha256": sha256(Path("research/conditioning_replication_plan.md"))
        if replication
        else None,
        "checkpoint_sha256": source_hash,
        "environment": environment(),
        "provenance": prov,
        "plan_sha256": sha256(Path("research/conditioning_plan.md")),
        "data_hashes": data.manifest["files"],
        "model_config": checkpoint["model_config"],
        "seed_for_random_controls": 1026,
        "spectrum_precision": "FP64 factors and FP64 materialized product from FP32 stored weights",
        "gradient_precision": "CPU FP32, one fixed loss direction",
        "inference_precision": "CUDA BF16",
        "timing_claim": False,
    }
    write_json(output / "protocol.json", protocol)
    cases = [("unchanged", None, False), ("svd_reconstruction", 0.0, False)]
    strengths = (0.1,) if replication else (0.01, 0.1, 1.0)
    cases += [(f"floor_{a:g}", a, False) for a in strengths]
    cases += [(f"random_{a:g}", a, True) for a in strengths]
    results, transforms = {}, {}
    for name, alpha, random_control in cases:
        print(f"Starting {name}", flush=True)
        model.cpu()
        model.load_state_dict(checkpoint["model"])
        changed = {}
        if alpha is not None:
            with torch.no_grad():
                for i, block in enumerate(model.blocks):
                    weight = block.ffn.down.second.weight
                    before = weight.clone()
                    after = floor_square_blocks(before, alpha)
                    if random_control:
                        after = matched_random(before, after, 1026 + i)
                    weight.copy_(after)
                    changed[f"blocks.{i}.ffn.down.second.weight"] = {
                        "max_abs_change": (after - before).abs().max().item(),
                        "relative_frobenius_change": (
                            (after - before).double().norm() / before.double().norm()
                        ).item(),
                        "per_block_change_norm": (after - before)
                        .double()
                        .norm(dim=(-2, -1))
                        .tolist(),
                    }
        allowed = set(changed)
        assert all(
            torch.equal(value, checkpoint["model"][key])
            for key, value in model.state_dict().items()
            if key not in allowed
        )
        transforms[name] = {key: model.state_dict()[key].clone() for key in allowed}
        record = {
            "alpha": alpha,
            "random_control": random_control,
            "changes": changed,
            "other_weights_bitwise_unchanged": True,
            "down_spectra": down_spectra(model),
            "gradient_flow": gradient_flow(model, x, y),
        }
        model.cuda()
        nll, tokens = evaluate_forward(model, batches, "cuda", "bf16")
        record.update(
            {
                "validation_nll": nll,
                "validation_tokens": tokens,
                "unique_parameters": sum(p.numel() for p in model.parameters()),
                "transformed_weights_sha256": {
                    k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                    for k, v in transforms[name].items()
                },
            }
        )
        record["worst_down_condition"] = max(
            v["projection"]["condition_nonzero"] for v in record["down_spectra"].values()
        )
        record["finite"] = all(
            r["finite"] for r in record["gradient_flow"]["records"].values()
        ) and bool(torch.isfinite(torch.tensor(nll)))
        results[name] = record
        write_json(output / f"{name}.json", record)
        print(
            f"Completed {name}: NLL={nll:.9f}, worst condition={record['worst_down_condition']:.3f}",
            flush=True,
        )
    baseline = results["unchanged"]
    archived_nll = json.loads((run / "metrics.json").read_text())["validation_loss"]
    checks = {
        "baseline_nll_reproduced": abs(baseline["validation_nll"] - archived_nll) <= 0.0002,
        "reconstruction_nll_preserved": abs(
            results["svd_reconstruction"]["validation_nll"] - baseline["validation_nll"]
        )
        <= 0.0001,
        "reconstruction_weights_preserved": max(
            v["max_abs_change"] for v in results["svd_reconstruction"]["changes"].values()
        )
        <= 1e-6,
        "source_checkpoint_unchanged": sha256(run / "checkpoint.pt") == source_hash,
    }
    for name, row in results.items():
        row["relative_nll_percent"] = 100 * (row["validation_nll"] / baseline["validation_nll"] - 1)
        row["condition_improvement_factor"] = (
            baseline["worst_down_condition"] / row["worst_down_condition"]
        )
        row["local_retention_gates"] = {
            "all_integrity_checks": all(checks.values()),
            "finite": row["finite"],
            "at_least_tenfold_condition_improvement": row["condition_improvement_factor"] >= 10,
            "nll_degradation_at_most_point_one_percent": row["relative_nll_percent"] <= 0.1,
        }
    torch.save(transforms, output / "transforms.pt")
    summary = {
        "protocol": protocol,
        "integrity_checks": checks,
        "cases": results,
        "interpretation": "Post-training diagnostic, one checkpoint and one noise direction. Projection conditioning is not full-network gradient stability; no training, serving-speed or novelty claim.",
    }
    write_json(output / "result.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    audit(args.run, args.output, args.replication)
