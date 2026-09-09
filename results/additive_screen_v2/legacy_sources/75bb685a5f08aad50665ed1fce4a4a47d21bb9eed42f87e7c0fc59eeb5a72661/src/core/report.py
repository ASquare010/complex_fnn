"""Regenerate descriptive leaderboards without mixing experimental cohorts."""

import csv
import hashlib
import json
from pathlib import Path

FIELDS = [
    "run",
    "candidate",
    "cohort",
    "seed",
    "training_tokens",
    "ffn_parameters",
    "total_parameters",
    "ffn_reduction_percent",
    "total_reduction_percent",
    "validation_loss",
    "perplexity",
    "training_tokens_per_second",
    "inference_tokens_per_second",
    "peak_allocated_vram_bytes",
    "final_gradient_norm",
    "clipped_step_fraction",
    "status",
]


def cohort(metrics: dict) -> str:
    model = {k: v for k, v in metrics["model"].items() if k not in ("variant", "hidden", "groups")}
    training = {k: v for k, v in metrics["training"].items() if k != "seed"}
    for key in ("ffn_width_init_mode", "ffn_width_lr_mode"):
        if training.get(key, "none") == "none":
            training.pop(key, None)
    # New default has exactly the historical optimizer behavior.
    if training.get("ffn_lr_mode", "uniform") == "uniform":
        training.pop("ffn_lr_mode", None)
    if training.get("ffn_decay_mode", "parameter") == "parameter":
        training.pop("ffn_decay_mode", None)
    if not training.get("recompute_gate", False):
        training.pop("recompute_gate", None)
    if training.get("gate_recompute_method", "checkpoint") == "checkpoint" or not training.get(
        "recompute_gate", False
    ):
        training.pop("gate_recompute_method", None)
    key = {
        "model": model,
        "training": training,
        "precision": metrics["precision"],
        "data": metrics["data"]["files"],
        "protocol": metrics["protocol"],
        "hardware_software": {
            k: metrics["environment"].get(k) for k in ("gpu", "torch", "cuda", "python")
        },
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def dominates(a: dict, b: dict) -> bool:
    if a["cohort"] != b["cohort"] or a["seed"] != b["seed"]:
        return False
    keys = ("validation_loss", "total_parameters", "peak_allocated_vram_bytes")
    weak = (
        all(a[k] <= b[k] for k in keys)
        and a["inference_tokens_per_second"] >= b["inference_tokens_per_second"]
    )
    strict = (
        any(a[k] < b[k] for k in keys)
        or a["inference_tokens_per_second"] > b["inference_tokens_per_second"]
    )
    return weak and strict


def report(root: Path) -> list[dict]:
    rows, experiments = [], []
    for path in sorted((root / "runs").glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        experiments.append(metrics)
        row = {k: metrics.get(k) for k in FIELDS}
        row.update(
            candidate=metrics["model"]["variant"],
            cohort=cohort(metrics),
            seed=metrics["training"]["seed"],
        )
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    for name, entries in (
        ("leaderboard.csv", rows),
        ("pareto.csv", [b for b in rows if not any(dominates(a, b) for a in rows)]),
    ):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(entries)
    with (root / "experiments.jsonl").open("w", encoding="utf-8") as handle:
        for record in experiments:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        for path in sorted((root / "runs").glob("*/failure.json")):
            handle.write(path.read_text(encoding="utf-8").replace("\n", "") + "\n")
    return rows
