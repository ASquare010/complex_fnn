"""Reproduce the locked three-seed comparison of BlockShuffle training recipes."""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.analysis import summarize
from src.core.config import TrainConfig
from src.core.reproducibility import sha256, write_json

SEEDS = (17, 29, 43)
RUNS = {
    "Full SwiGLU (512)": ["swiglu_s17_800", "swiglu_h0_s29_800", "swiglu_h0_s43_800"],
    "Narrow SwiGLU (152)": [f"swiglu_narrow_h152_s{s}_800" for s in SEEDS],
    "BlockShuffle (1024)": [f"blockshuffle_h1024_native_s{s}_800" for s in SEEDS],
}
RECIPE_FIELDS = {"ffn_lr_mode", "ffn_decay_mode", "recompute_gate", "gate_recompute_method"}


def paired(values, reference) -> dict:
    deltas = [a - b for a, b in zip(values, reference, strict=True)]
    relative = [100 * (a / b - 1) for a, b in zip(values, reference, strict=True)]
    # Student-t(2), two-sided 95%; exploratory normality assumption across seeds.
    critical = math.sqrt(2) * 0.95 / math.sqrt(1 - 0.95**2)
    margin = critical * statistics.stdev(deltas) / math.sqrt(3)
    return {
        "seeds": list(SEEDS),
        "nll_differences": deltas,
        "relative_nll_percent_samples": relative,
        "nll_difference": summarize(deltas),
        "relative_nll_percent": summarize(relative),
        "paired_nll_95pct_t_interval": [
            statistics.mean(deltas) - margin,
            statistics.mean(deltas) + margin,
        ],
        "better_seeds": sum(d < 0 for d in deltas),
    }


def validate(records) -> None:
    reference = next(iter(records.values()))[0]
    for label, runs in records.items():
        for seed, run in zip(SEEDS, runs, strict=True):
            assert run["training"]["seed"] == seed
            assert run["training"]["steps"] == 800
            assert run["training_tokens"] == 1638400
            assert run["validation_tokens"] == 32768
            assert run["data"]["files"] == reference["data"]["files"], "Data mismatch"
            assert run["protocol"] == reference["protocol"]
            assert run["precision"] == reference["precision"]
            for key in ("width", "layers", "heads", "context", "vocab_size"):
                assert run["model"][key] == reference["model"][key], "Non-FFN shape mismatch"
            for key in ("gpu", "torch", "cuda", "python"):
                assert run["environment"][key] == reference["environment"][key]
            left = vars(TrainConfig(**run["training"]))
            right = vars(TrainConfig(**reference["training"]))
            for key in left.keys() - RECIPE_FIELDS - {"seed"}:
                assert left[key] == right[key], f"Unexpected training difference: {key}"
            expected_lr = "fan_in" if label.startswith("BlockShuffle") else "uniform"
            assert left["ffn_lr_mode"] == expected_lr
            assert left["ffn_decay_mode"] == "parameter"
            assert left["recompute_gate"] == label.startswith("BlockShuffle")
            if left["recompute_gate"]:
                assert left["gate_recompute_method"] == "native"
        assert all(r["model"] == runs[0]["model"] for r in runs)


def report(
    root: Path = Path("results"), document: Path = Path("research/blockshuffle_results.md")
) -> dict:
    records = {
        label: [
            json.loads((root / "runs" / name / "metrics.json").read_text(encoding="utf-8"))
            for name in names
        ]
        for label, names in RUNS.items()
    }
    validate(records)
    metrics = (
        "validation_loss",
        "training_tokens_per_second",
        "inference_tokens_per_second",
        "peak_allocated_vram_bytes",
        "clipped_step_fraction",
    )
    aggregate = {
        label: {
            "runs": RUNS[label],
            "seeds": list(SEEDS),
            "ffn_parameters": runs[0]["ffn_parameters"],
            "total_parameters": runs[0]["total_parameters"],
            "ffn_matrix_forward_flops_per_token": runs[0]["ffn_matrix_forward_flops_per_token"],
            **{key: summarize([r[key] for r in runs]) for key in metrics},
        }
        for label, runs in records.items()
    }
    candidate = [r["validation_loss"] for r in records["BlockShuffle (1024)"]]
    comparisons = {
        label: paired(candidate, [r["validation_loss"] for r in runs])
        for label, runs in records.items()
        if not label.startswith("BlockShuffle")
    }
    result = {
        "aggregate": aggregate,
        "paired_candidate_minus_reference": comparisons,
        "metric_sha256": {
            name: sha256(root / "runs" / name / "metrics.json")
            for names in RUNS.values()
            for name in names
        },
        "interpretation": "Architecture plus initialization and optimizer recipe comparison, not an isolated architecture ablation. Seed 17 selected width/recipe; seeds 29 and 43 replicate the locked recipe. Same short token budget and held-out validation subset. Seed intervals are exploratory, assume normal differences, and are not adjusted for model selection. Broader corpus, converged training, equal hyperparameter tuning and trained scaling remain untested.",
    }
    write_json(root / "blockshuffle_summary.json", result)
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    colors = ("#424b5a", "#cc792b", "#087e8b")
    for (label, runs), color in zip(records.items(), colors, strict=True):
        histories = [
            [
                json.loads(line)
                for line in (root / "runs" / run["run"] / "history.jsonl").read_text().splitlines()
            ]
            for run in runs
        ]
        steps = [r["step"] for r in histories[0] if r["step"] >= 200]
        assert all([r["step"] for r in hist if r["step"] >= 200] == steps for hist in histories)
        values = np.array(
            [[r["validation_loss"] for r in hist if r["step"] >= 200] for hist in histories]
        )
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[0].plot(steps, mean, label=label, color=color)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        axes[1].errorbar(
            runs[0]["ffn_parameters"] / 1000,
            mean[-1],
            yerr=std[-1],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    full_mean = aggregate["Full SwiGLU (512)"]["validation_loss"]["mean"]
    axes[1].axhline(
        full_mean * 1.01,
        color="#666666",
        linestyle="--",
        linewidth=1,
        label="Full mean + 1% target",
    )
    axes[0].set(
        xlabel="Training steps (200 onward)",
        ylabel="Validation NLL",
        title="Mean and sample SD across three seeds",
    )
    axes[1].set(
        xlabel="Unique FFN parameters (thousands)",
        ylabel="Final validation NLL",
        title="800 steps; lower NLL is better",
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(plots / "blockshuffle_three_seed.png", dpi=100)
    fig.savefig(plots / "blockshuffle_three_seed.svg")
    plt.close(fig)
    lines = [
        "# BlockShuffle: locked three-seed results",
        "",
        "BlockShuffle improves quality over the narrow parameter-matched SwiGLU control in all three seeds. The full research target remains unmet.",
        "",
        "The candidate uses 350,208 FFN weights: **70.3125% fewer** than full SwiGLU. Its factorized FFN matrix FLOPs fall by the same fraction. Total model parameters fall from 2,557,632 to 1,728,192 (32.43%). Wider hidden activations and six grouped matrix operations per FFN still cost time and memory.",
        "",
        "## Controlled comparison",
        "",
        "All nine runs use the same cached TinyStories data, tokenizer, decoder dimensions, BF16 precision, seed-specific data order, and 1,638,400 training tokens (800 steps). Validation uses 32,768 fixed targets. Seeds are 17, 29 and 43. Seed 17 informed candidate selection; the other two replicate the locked recipe.",
        "",
        "The structured candidate uses orthogonal factor initialization, per-factor fan-in learning rates and native gate recomputation. Dense controls retain the original initialization and uniform rate. This compares complete training recipes; it does not isolate architecture or equalize hyperparameter-search effort. The [frozen width plan](blockshuffle_width_plan.md) and [memory plan](gate_recompute_plan.md) disclose selection.",
        "",
        "| Method | Unique FFN weights | Total weights | Mean NLL +/- sample SD | Peak training MiB (max) | Training tokens/s (mean) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in aggregate.items():
        nll = row["validation_loss"]
        lines.append(
            f"| {label} | {row['ffn_parameters']:,} | {row['total_parameters']:,} | {nll['mean']:.6f} +/- {nll['std']:.6f} | {row['peak_allocated_vram_bytes']['max'] / 2**20:.2f} | {row['training_tokens_per_second']['mean']:,.0f} |"
        )
    lines += [
        "",
        "Training timing excludes evaluation, checkpointing and the first ten steps. Separate-run throughput varies with laptop GPU clocks; use the [serving audit](../results/serving_blockshuffle_final.json) for rotating-order forward comparisons. These are full-sequence forwards, not autoregressive generation.",
        "",
        "### Every seed",
        "",
        "| Seed | Full SwiGLU | Narrow SwiGLU | BlockShuffle |",
        "|---|---:|---:|---:|",
    ]
    for i, seed in enumerate(SEEDS):
        losses = " | ".join(f"{runs[i]['validation_loss']:.6f}" for runs in records.values())
        lines.append(f"| {seed} | {losses} |")
    lines += [
        "",
        "### Paired differences",
        "",
        "Negative differences favor BlockShuffle. Intervals are exploratory Student-t 95% intervals across three paired seeds, assuming normal differences; they do not account for model selection or data uncertainty.",
        "",
    ]
    for label, row in comparisons.items():
        low, high = row["paired_nll_95pct_t_interval"]
        lines.append(
            f"- Versus {label}: mean NLL difference {row['nll_difference']['mean']:+.6f}; interval [{low:+.6f}, {high:+.6f}]; mean paired relative difference {row['relative_nll_percent']['mean']:+.3f}%; better in {row['better_seeds']}/3 seeds."
        )
    lines += [
        "",
        "![Three-seed learning curves and parameter comparison](../results/plots/blockshuffle_three_seed.png)",
        "",
        "## Limits and decision",
        "",
        "This is a promising compressed reference built from prior work, with a useful implementation and optimizer study. It is not a verified new architecture or a breakthrough. The mean quality gap against full SwiGLU remains above 1%; the complete gold target cannot pass. Full GELU currently has only one 800-step seed. Trained scaling, a broader corpus, equal tuning, convergence and robust deployment timing remain open.",
        "",
        "The [progress and ablation report](blockshuffle_progress.md) includes failed branches, memory tradeoffs and inference packing. Raw [summary JSON](../results/blockshuffle_summary.json) identifies every input metric by hash; original histories, configurations, checkpoints and source archives remain in results/runs.",
        "",
        "A later [calibrated narrow reference](width_calibration_results.md) improves the conventional control. The comparison above remains the original locked experiment; use both reports when assessing architecture gains.",
        "",
        "Regenerate with `uv run python -m src.core.blockshuffle_report`.",
        "",
    ]
    document.write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    report(args.root)
