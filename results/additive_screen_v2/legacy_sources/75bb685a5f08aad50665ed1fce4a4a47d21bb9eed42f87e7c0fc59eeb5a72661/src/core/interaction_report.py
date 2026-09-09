"""Report every cross-group function task and seed, including negative controls."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.analysis import summarize
from src.core.reproducibility import write_json


def report(root: Path = Path("results")) -> dict:
    grouped = defaultdict(list)
    input_hashes = set()
    for directory in (root / "interactions_v1", root / "interactions_adaptive_v1"):
        result = json.loads((directory / "summary.json").read_text())
        for row in result["rows"]:
            input_hashes.add(row["inputs_sha256"])
            grouped[(row["task"], row["variant"])].append(row)
    if len(input_hashes) != 1:
        raise ValueError("Do not mix function trials with different saved inputs")
    aggregate = {}
    for (task, variant), rows in grouped.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Duplicate function seeds")
        aggregate[f"{task}:{variant}"] = {
            "seeds": [r["seed"] for r in rows],
            "parameters": rows[0]["parameters"],
            "heldout_normalized_mse": summarize([r["heldout_normalized_mse"] for r in rows]),
            "training_examples_per_second": summarize(
                [r["training_examples_per_second"] for r in rows]
            ),
        }
    tasks = ["constructed_polynomial", "additive_control", "pure_product"]
    variants = [
        "swiglu",
        "swiglu_narrow",
        "structured_swiglu",
        "structured_swiglu_coupled",
        "structured_swiglu_adaptive",
        "swiglu_budget",
    ]
    labels = [
        "Full (193)",
        "Narrow (49)",
        "Grouped (49)",
        "Fixed mix (49)",
        "Adaptive (65)",
        "Budget (61)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, task in zip(axes, tasks, strict=True):
        for i, variant in enumerate(variants):
            values = [r["heldout_normalized_mse"] for r in grouped[(task, variant)]]
            center = statistics.mean(values)
            ax.errorbar(
                i,
                center,
                yerr=[[center - min(values)], [max(values) - center]],
                fmt="o",
                color=f"C{i}",
                capsize=4,
            )
            ax.scatter(np.repeat(i, len(values)), values, color=f"C{i}", alpha=0.4, s=12)
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(task.replace("_", " "))
        ax.set_yscale("log")
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Held-out normalized MSE (log scale; lower better)")
    fig.suptitle("Chosen mechanism diagnostics: 3 seeds, 1000 steps; mean and min/max")
    fig.tight_layout()
    (root / "plots").mkdir(exist_ok=True)
    fig.savefig(root / "plots/interactions.png", dpi=150)
    plt.close(fig)
    coefficients = {}
    for path in (root / "interactions_adaptive_v1").glob(
        "*structured_swiglu_adaptive*/checkpoint.pt"
    ):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        raw = checkpoint["model"]["ffn.mix_theta"]
        coefficient = 0.75 * raw.tanh()
        coefficients[path.parent.name] = {
            "raw": raw.tolist(),
            "coefficient": coefficient.tolist(),
            "max_abs": coefficient.abs().max().item(),
            "rms": coefficient.square().mean().sqrt().item(),
        }
    result = {
        "aggregate": aggregate,
        "adaptive_coefficients": coefficients,
        "input_hashes": sorted(input_hashes),
        "limits": "Three chosen mechanism tasks; finite optimization and one fixed holdout. Adaptive has 65 parameters versus 49 for original controls and 61 for extra-budget narrow control. Not an LM or broad expressivity-performance claim.",
    }
    write_json(root / "interaction_summary.json", result)
    return result


if __name__ == "__main__":
    report()
