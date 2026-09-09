"""One controlled training loop for all FFN candidates."""

import json
import math
import time
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.core.benchmark import autocast, evaluate, forward_flops, inference_benchmark, synchronize
from src.core.config import ModelConfig, TrainConfig
from src.core.data import TokenData
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance, write_json
from src.core.transformer import Transformer


def learning_rate(step: int, config: TrainConfig) -> float:
    warmup = max(1, config.steps // 10)
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, config.steps - warmup - 1)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def resolve_precision(config: TrainConfig) -> str:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    supported = config.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    if config.precision == "bf16" and not supported:
        raise ValueError("BF16 requested without a supported CUDA device")
    return "bf16" if supported and config.precision != "fp32" else "fp32"


def configure_gate_recomputation(model, config: TrainConfig) -> None:
    """Apply optional gate execution consistently in training and qualification."""
    if not config.recompute_gate:
        return
    variant = model.config.variant
    if not variant.startswith("blockshuffle_swiglu"):
        raise ValueError("Gate recomputation supports BlockShuffle SwiGLU")
    if variant.endswith("_rational") and config.gate_recompute_method != "checkpoint":
        raise ValueError("Learnable activations require checkpoint gate recomputation")
    for block in model.blocks:
        block.ffn.recompute_gate = True
        block.ffn.gate_recompute_method = config.gate_recompute_method


def train(model_config: ModelConfig, config: TrainConfig, cache: Path, output: Path) -> dict:
    model_config.validate()
    config.validate()
    precision = resolve_precision(config)
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    data = TokenData(cache, config.device, config.seed + 10000)
    if model_config.vocab_size != data.manifest["vocab_size"]:
        raise ValueError("Model vocabulary must match frozen tokenizer")
    prov = provenance()
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
    model = Transformer(model_config, config.seed).to(config.device)
    width_initialization = initialize_dense_width(model, config)
    configure_gate_recomputation(model, config)
    compilation = None
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    base = {
        "model": asdict(model_config),
        "training": asdict(config),
        "precision": precision,
        "environment": environment(),
        "provenance": prov,
        "data": data.manifest,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "micro_v1",
        "optimizer_parameter_groups": group_summary(groups),
        "ffn_width_initialization": width_initialization,
        "activation_compilation": compilation,
    }
    write_json(output / "config.json", base)
    records, clip_count, timed_seconds, timed_tokens = [], 0, 0.0, 0

    def validation():
        return evaluate(
            model,
            data.validation(config.batch_size, model_config.context, config.eval_batches),
            config.device,
            precision,
        )

    fixed_x, _ = next(data.validation(config.batch_size, model_config.context, 1))
    initial, validation_tokens = validation()
    write_json(
        output / "initial_diagnostics.json",
        inspect_layers(model, fixed_x, config.device, precision),
    )
    synchronize(config.device)
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(config.steps):
            model.train()
            synchronize(config.device)
            start = time.perf_counter()
            x, y = data.batch(config.batch_size, model_config.context)
            rate = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["lr_scale"]
            optimizer.zero_grad(set_to_none=True)
            with autocast(config.device, precision):
                loss = model.loss(x, y)
            loss.backward()
            # error_if_nonfinite checks all parameter gradients through the total norm.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            clip_count += int(norm.item() > 1.0)
            logged = step == 0 or (step + 1) % config.log_every == 0 or step + 1 == config.steps
            gradients = gradient_stats(model) if logged else None
            optimizer.step()
            synchronize(config.device)
            elapsed = time.perf_counter() - start
            if step >= 10:
                timed_seconds += elapsed
                timed_tokens += y.numel()
            if logged:
                valid, _ = validation()
                row = {
                    "step": step + 1,
                    "training_loss": loss.item(),
                    "validation_loss": valid,
                    "gradient_norm_pre_clip": norm.item(),
                    "layer_gradient_norms_post_clip": gradients,
                    "learning_rate": rate,
                }
                records.append(row)
                with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(
                    json.dumps(
                        {
                            "run": output.name,
                            **{
                                k: row[k]
                                for k in (
                                    "step",
                                    "training_loss",
                                    "validation_loss",
                                    "gradient_norm_pre_clip",
                                )
                            },
                        }
                    ),
                    flush=True,
                )
        final, _ = validation()
        peak = torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else 0
        diagnostics = inspect_layers(model, fixed_x, config.device, precision)
        write_json(output / "final_diagnostics.json", diagnostics)
        timing = inference_benchmark(model, fixed_x, config.device, precision)
        params = sum(p.numel() for p in model.parameters())
        ffn_unique = {id(p): p for b in model.blocks for p in b.ffn.parameters()}
        ffn_params = sum(p.numel() for p in ffn_unique.values())
        assert ffn_params == model_config.unique_ffn_parameters
        full_ffn = model_config.layers * 8 * model_config.width**2
        reference_total = params - ffn_params + full_ffn
        metrics = {
            **base,
            "run": output.name,
            "status": "SCREENED",
            "initial_validation_loss": initial,
            "validation_loss": final,
            "perplexity": math.exp(final),
            "validation_tokens": validation_tokens,
            "training_tokens": config.steps * config.batch_size * model_config.context,
            "training_cache_exposure_ratio": config.steps
            * config.batch_size
            * model_config.context
            / len(data.train),
            "ffn_parameters": ffn_params,
            "total_parameters": params,
            "ffn_reduction_percent": 100 * (1 - ffn_params / full_ffn),
            "total_reduction_percent": 100 * (1 - params / reference_total),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "optimizer_state_bytes": sum(
                t.numel() * t.element_size()
                for s in optimizer.state.values()
                for t in s.values()
                if isinstance(t, torch.Tensor)
            ),
            "peak_allocated_vram_bytes": peak,
            "final_gradient_norm": norm.item(),
            "gradient_diagnostics_note": "Shared FFN parameter gradients aggregate all uses; repeated per-layer values are not independent invocation gradients.",
            "clipped_step_fraction": clip_count / config.steps,
            "training_tokens_per_second": timed_tokens / timed_seconds if timed_seconds else None,
            "timed_training_seconds": timed_seconds,
            "training_timing_note": "Excludes first 10 steps, evaluation and checkpoints; includes sampling/optimizer and occasional gradient diagnostics",
            **timing,
            **forward_flops(model_config),
            "wall_seconds_before_checkpoint": time.perf_counter() - wall_start,
        }
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_config),
                "training_config": asdict(config),
                "step": config.steps,
                "sampling_rng": data.generator.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            output / "checkpoint.pt",
        )
        write_json(output / "metrics.json", metrics)
        return metrics
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                **base,
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "completed_logs": records,
            },
        )
        raise
