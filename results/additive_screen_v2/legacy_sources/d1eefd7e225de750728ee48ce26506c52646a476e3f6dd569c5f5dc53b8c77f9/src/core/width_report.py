"""Reproduce dense-width ablations and the stronger three-seed narrow control."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.analysis import summarize
from src.core.blockshuffle_report import RUNS, SEEDS, paired
from src.core.config import TrainConfig
from src.core.reproducibility import sha256, write_json


def report(root=Path("results")) -> dict:
    def read(name):
        return json.loads((root / "runs" / name / "metrics.json").read_text(encoding="utf-8"))

    runs = {
        **RUNS,
        "Calibrated narrow SwiGLU": [f"swiglu_h152_width_init_and_lr_s{s}_800" for s in SEEDS],
    }
    records = {label: [read(name) for name in names] for label, names in runs.items()}
    reference = records["Full SwiGLU (512)"][0]
    changed = {
        "ffn_lr_mode",
        "ffn_decay_mode",
        "recompute_gate",
        "gate_recompute_method",
        "ffn_width_init_mode",
        "ffn_width_lr_mode",
        "seed",
    }
    for label, group in records.items():
        for seed, m in zip(SEEDS, group, strict=True):
            assert m["training"]["seed"] == seed
            assert m["training_tokens"] == 1638400 and m["validation_tokens"] == 32768
            assert m["data"]["files"] == reference["data"]["files"]
            assert (
                m["protocol"] == reference["protocol"] and m["precision"] == reference["precision"]
            )
            for key in ("width", "layers", "heads", "context", "vocab_size"):
                assert m["model"][key] == reference["model"][key]
            for key in ("gpu", "torch", "cuda", "python"):
                assert m["environment"][key] == reference["environment"][key]
            training, base = (
                vars(TrainConfig(**m["training"])),
                vars(TrainConfig(**reference["training"])),
            )
            assert all(training[k] == base[k] for k in training.keys() - changed)
            if label == "Calibrated narrow SwiGLU":
                assert training["ffn_width_init_mode"] == training["ffn_width_lr_mode"] == "fan_in"
                assert (
                    training["ffn_decay_mode"] == "product" and training["ffn_lr_mode"] == "uniform"
                )
                assert m["ffn_parameters"] == 350208 and m["model"]["hidden"] == 152
    aggregates = {
        label: {
            "runs": runs[label],
            "ffn_parameters": group[0]["ffn_parameters"],
            "total_parameters": group[0]["total_parameters"],
            **{
                key: summarize([m[key] for m in group])
                for key in (
                    "validation_loss",
                    "training_tokens_per_second",
                    "inference_tokens_per_second",
                    "peak_allocated_vram_bytes",
                    "clipped_step_fraction",
                )
            },
        }
        for label, group in records.items()
    }
    losses = {label: [m["validation_loss"] for m in group] for label, group in records.items()}
    comparisons = {
        "calibrated_minus_original_narrow": paired(
            losses["Calibrated narrow SwiGLU"], losses["Narrow SwiGLU (152)"]
        ),
        "calibrated_minus_full": paired(
            losses["Calibrated narrow SwiGLU"], losses["Full SwiGLU (512)"]
        ),
        "blockshuffle_minus_calibrated": paired(
            losses["BlockShuffle (1024)"], losses["Calibrated narrow SwiGLU"]
        ),
    }
    screens = {
        "Original narrow": read("swiglu_narrow_h152_s17_200"),
        **{
            name: read(f"swiglu_h152_width_{name}_s17_200")
            for name in ("init_only", "lr_only", "init_and_lr")
        },
    }
    result = {
        "aggregate": aggregates,
        "paired_differences": comparisons,
        "screen_runs": {label: m["run"] for label, m in screens.items()},
        "metric_sha256": {
            name: sha256(root / "runs" / name / "metrics.json")
            for names in runs.values()
            for name in names
        },
        "interpretation": "A stronger conventional reference, not a new architecture. Three calibration choices were screened on seed 17; seeds 29 and 43 replicate the locked choice. Same data and training budgets; explicit initialization, LR and decay treatments differ. Short-budget, single-corpus evidence. Student-t intervals assume normal paired seed differences and do not adjust for selection.",
    }
    write_json(root / "width_calibration_summary.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    colors = ("#424b5a", "#cc792b", "#087e8b", "#87569b")
    for (label, group), color in zip(records.items(), colors, strict=True):
        histories = [
            [
                json.loads(line)
                for line in (root / "runs" / m["run"] / "history.jsonl").read_text().splitlines()
            ]
            for m in group
        ]
        steps = [r["step"] for r in histories[0] if r["step"] >= 200]
        values = np.array(
            [[r["validation_loss"] for r in history if r["step"] >= 200] for history in histories]
        )
        mean, std = values.mean(0), values.std(0, ddof=1)
        axes[0].plot(steps, mean, label=label, color=color)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.13)
    labels = list(aggregates)
    axes[1].errorbar(
        np.arange(4),
        [aggregates[k]["validation_loss"]["mean"] for k in labels],
        yerr=[aggregates[k]["validation_loss"]["std"] for k in labels],
        fmt="o",
        capsize=5,
        color="#424b5a",
    )
    axes[1].set_xticks(range(4), ("Full", "Original\nnarrow", "BlockShuffle", "Calibrated\nnarrow"))
    axes[0].set(
        xlabel="Training steps (200 onward)",
        ylabel="Validation NLL",
        title="Mean and sample SD across three seeds",
    )
    axes[1].set(
        ylabel="Final validation NLL",
        title="800 steps\nCompressed models: 350,208 FFN weights",
    )
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(root / "plots" / "width_calibration.png", dpi=100, bbox_inches="tight")
    fig.savefig(root / "plots" / "width_calibration.svg", bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# A stronger narrow SwiGLU reference",
        "",
        "Changing the output-projection learning rate improves the conventional narrow control without adding parameters or forward operations. This reduces the apparent advantage of the structured recipe. It does not establish a new nonlinear primitive or satisfy the full research target.",
        "",
        "## Frozen ablations",
        "",
        "At hidden width 152, calibration uses initial down-weight gain sqrt(64/19) and down-projection LR multiplier 64/19. The matching decay coefficient is 0.1/(64/19), preserving the original pure AdamW shrinkage exactly. Initialization and LR are tested separately. All other weights, optimizer settings, data order and compute budgets stay fixed. See [the calculation and frozen plan](width_calibration_plan.md).",
        "",
        "| Seed-17 recipe, 200 steps | NLL | Change versus original narrow |",
        "|---|---:|---:|",
    ]
    original = screens["Original narrow"]["validation_loss"]
    for label, m in screens.items():
        lines.append(
            f"| {label} | {m['validation_loss']:.6f} | {100 * (m['validation_loss'] / original - 1):+.3f}% |"
        )
    lines += [
        "",
        "Initialization alone does not improve this screen. Most of the observed gain comes from LR; both together give the lowest NLL and earn the predeclared longer-budget check. The early gain is not extrapolated to 800 steps.",
        "",
        "## Three-seed comparison at 800 steps",
        "",
        "Every run uses 1,638,400 training tokens, the same frozen TinyStories cache and 32,768 validation targets. Seeds are 17, 29 and 43. The calibrated recipe was selected using seed 17. All compressed methods have 350,208 FFN weights and 1,728,192 total weights: reductions of 70.3125% and 32.43% relative to full SwiGLU. Initialization and optimizer recipes are explicitly different; this does not equalize all hyperparameter-search effort.",
        "",
        "| Method | Mean NLL +/- sample SD | Training tokens/s, mean | Peak training MiB, max |",
        "|---|---:|---:|---:|",
    ]
    for label, row in aggregates.items():
        n = row["validation_loss"]
        lines.append(
            f"| {label} | {n['mean']:.6f} +/- {n['std']:.6f} | {row['training_tokens_per_second']['mean']:,.0f} | {row['peak_allocated_vram_bytes']['max'] / 2**20:.2f} |"
        )
    lines += [
        "",
        "| Seed | Full SwiGLU | Original narrow | BlockShuffle | Calibrated narrow |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, seed in enumerate(SEEDS):
        lines.append(
            f"| {seed} | "
            + " | ".join(f"{group[i]['validation_loss']:.6f}" for group in records.values())
            + " |"
        )
    lines += [
        "",
        "Negative paired differences favor the first method. These exploratory 95% Student-t intervals assume normal differences across three seeds and exclude data/model-selection uncertainty.",
        "",
    ]
    for label, row in comparisons.items():
        lo, hi = row["paired_nll_95pct_t_interval"]
        lines.append(
            f"- {label.replace('_', ' ')}: mean difference {row['nll_difference']['mean']:+.6f}, interval [{lo:+.6f}, {hi:+.6f}], mean paired relative difference {row['relative_nll_percent']['mean']:+.3f}%."
        )
    lines += [
        "",
        "![Three-seed results with the calibrated reference](../results/plots/width_calibration.png)",
        "",
        "## Scope and next use",
        "",
        "Retain both original and calibrated references. Any new primitive should compare against the stronger conventional recipe. The original structured result remains a measurement against its stated control; it must not be presented as a gain over a fully tuned narrow model. Calibration is an optimization control inspired by prior width-aware parameterization, with an elementary variance/decay calculation. It is not architectural novelty or a universal stability result.",
        "",
        "Equal-execution serving appears below; checkpoint diagnostics are linked from the current state. No scaling, broader-corpus, convergence or autoregressive-generation claim follows from these micro-model runs. Regenerate this report with `uv run python -m src.core.width_report`. Raw [summary JSON](../results/width_calibration_summary.json) records all paired differences and metric hashes.",
        "",
    ]
    serving_lines = [
        "",
        "## Equal-execution serving, seed 17",
        "",
        "Both modes use batch 16, context 128 and BF16. Ratios are candidate throughput divided by full SwiGLU throughput within rotating timing rounds; this is full-sequence inference, not autoregressive generation.",
        "",
        "| Execution | Median throughput ratio | Range | Candidate peak MiB | Full peak MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, filename in (
        ("Eager", "serving_width_calibrated_eager.json"),
        ("CUDA graph, corrected memory", "serving_width_calibrated_graph_v2.json"),
    ):
        path = root / filename
        if path.exists():
            audit = json.loads(path.read_text())
            ratio = audit["paired_timing"]["throughput_ratio_to_dense_reference"]["candidate"]
            memory = {
                r["mode"]: r["peak_inference_allocated_vram_bytes"] / 2**20 for r in audit["rows"]
            }
            serving_lines.append(
                f"| {label} | {ratio['median']:.3f} | {ratio['min']:.3f}-{ratio['max']:.3f} | {memory['candidate']:.2f} | {memory['dense_reference']:.2f} |"
            )
    serving_lines += [
        "",
        "Graph memory includes context workspaces and live model/output storage; it uses the corrected [shared-stream protocol](graph_memory_correction.md). Raw [eager](../results/serving_width_calibrated_eager.json) and [graph](../results/serving_width_calibrated_graph_v2.json) records retain all rounds and correctness checks. Training-memory measurements are separate and unaffected by the graph correction.",
        "",
    ]
    lines += serving_lines
    Path("research/width_calibration_results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    report()
