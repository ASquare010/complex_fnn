"""Run reproducible experiments with uv run python -m src.core.cli."""

import argparse
import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.data import load_manifest, prepare
from src.core.report import report
from src.core.reproducibility import environment


def build_configs(args: argparse.Namespace, vocab_size: int) -> tuple[ModelConfig, TrainConfig]:
    settings = json.loads(args.config.read_text()) if args.config else {}
    model = dict(settings.get("model", {}))
    training = dict(settings.get("training", {}))
    for field in fields(ModelConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            model[field.name] = value
    for field in fields(TrainConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            training[field.name] = value
    model["vocab_size"] = vocab_size
    model_config, train_config = ModelConfig(**model), TrainConfig(**training)
    model_config.validate()
    train_config.validate()
    return model_config, train_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hardware")
    sub.add_parser("counts")
    data = sub.add_parser("prepare")
    data.add_argument("--cache", type=Path, default=Path("data/tinystories_v1"))
    data.add_argument("--tokenizer", type=Path, help="Reuse an archived train-only tokenizer")
    data.add_argument("--train-stories", type=int, default=12000)
    data.add_argument("--validation-stories", type=int, default=1000)
    run = sub.add_parser("train")
    run.add_argument(
        "--config", type=Path, help="JSON model/training settings; explicit CLI flags override"
    )
    run.add_argument("--variant", choices=VARIANTS)
    for name in (
        "steps",
        "seed",
        "batch-size",
        "width",
        "layers",
        "heads",
        "context",
        "hidden",
        "groups",
        "eval-batches",
        "log-every",
    ):
        run.add_argument(f"--{name}", type=int)
    for name in ("learning-rate", "weight-decay"):
        run.add_argument(f"--{name}", type=float)
    run.add_argument("--ffn-width-init-mode", choices=("none", "fan_in"))
    run.add_argument("--ffn-width-lr-mode", choices=("none", "fan_in"))
    run.add_argument("--gate-recompute-method", choices=("checkpoint", "native"))
    run.add_argument(
        "--activation-backend",
        choices=("eager",),
    )
    run.add_argument("--recompute-gate", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--ffn-decay-mode", choices=("parameter", "product"))
    run.add_argument("--ffn-lr-mode", choices=("uniform", "fan_in"))
    run.add_argument("--device")
    run.add_argument("--precision", choices=("auto", "fp32", "bf16"))
    run.add_argument("--cache", type=Path, default=Path("data/tinystories_v1"))
    run.add_argument("--output", type=Path)
    summary = sub.add_parser("report")
    summary.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.command == "hardware":
        print(json.dumps(environment(), indent=2))
    elif args.command == "counts":
        for variant in VARIANTS:
            c = ModelConfig(variant=variant)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "hidden": c.ffn_width,
                        "ffn_per_invocation": c.ffn_parameters,
                        "unique_ffn_total": c.unique_ffn_parameters,
                        "groups": c.groups,
                        "total": c.total_parameters,
                    }
                )
            )
    elif args.command == "prepare":
        print(
            json.dumps(
                prepare(args.cache, args.train_stories, args.validation_stories, args.tokenizer),
                indent=2,
            )
        )
    elif args.command == "report":
        from src.core.analysis import analyze

        print(json.dumps(report(args.root), indent=2))
        analyze(args.root)
    elif args.command == "train":
        from src.core.trainer import train

        manifest = load_manifest(args.cache)
        model, training = build_configs(args, manifest["vocab_size"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        output = args.output or Path("results/runs") / f"{stamp}_{model.variant}_s{training.seed}"
        print(json.dumps({"model": asdict(model), "training": asdict(training)}), flush=True)
        result = train(model, training, args.cache, output)
        report(output.parent.parent)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "run",
                        "validation_loss",
                        "perplexity",
                        "ffn_reduction_percent",
                        "training_tokens_per_second",
                        "inference_tokens_per_second",
                        "peak_allocated_vram_bytes",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
