"""Strict three-seed report for the completed locked larger-model cohort."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import sha256, write_json
from src.core.scale_report import RECIPES

SEEDS = (17, 29, 43)


def load(root):
    records = {}
    common = None
    for seed in SEEDS:
        records[seed] = {}
        for key in RECIPES:
            path = root / "runs" / f"scale384_{key}_s{seed}_800" / "metrics.json"
            m = json.loads(path.read_text())
            cfg = json.loads(Path(f"configs/scale384_{key}_800.json").read_text())
            cfg["training"]["seed"] = seed
            assert ModelConfig(**m["model"]) == ModelConfig(**cfg["model"])
            assert TrainConfig(**m["training"]) == TrainConfig(**cfg["training"])
            assert (
                m["precision"] == "bf16"
                and m["validation_tokens"] == 32768
                and m["training_tokens"] == 1638400
            )
            if records[seed]:
                first = next(iter(records[seed].values()))
                assert (
                    m["data"]["files"] == first["data"]["files"]
                    and m["environment"] == first["environment"]
                )
            diagnostics = json.loads((path.parent / "final_diagnostics.json").read_text())
            assert all(v["finite"] for v in diagnostics.values()) and np.isfinite(
                m["final_gradient_norm"]
            )
            if common is None:
                common = m
            assert m["data"]["files"] == common["data"]["files"]
            assert m["environment"] == common["environment"]
            records[seed][key] = m
    return records


def report(root: Path = Path("results")):
    records = load(root)
    stats = {}
    for key in RECIPES:
        values = [records[s][key]["validation_loss"] for s in SEEDS]
        stats[key] = {
            "mean_nll": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "nll_by_seed": dict(zip(SEEDS, values, strict=True)),
        }
    paired = {}
    gates = {}
    for seed, group in records.items():
        b = group["blockshuffle"]
        paired[seed] = {
            k: 100 * (b["validation_loss"] / group[k]["validation_loss"] - 1)
            for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
        }
        gates[seed] = {
            "ffn_weight_reduction_at_least_70_percent": b["ffn_parameters"]
            <= 0.3 * group["full_swiglu"]["ffn_parameters"],
            "within_1_percent_full_swiglu": paired[seed]["full_swiglu"] <= 1,
            "within_1_percent_full_gelu": paired[seed]["full_gelu"] <= 1,
            "beats_calibrated_narrow": paired[seed]["calibrated_narrow"] < 0,
            "training_peak_within_10_percent_full_swiglu": b["peak_allocated_vram_bytes"]
            <= 1.1 * group["full_swiglu"]["peak_allocated_vram_bytes"],
        }
    result = {
        "seeds": list(SEEDS),
        "statistics": stats,
        "paired_relative_nll_percent": paired,
        "per_seed_gates": gates,
        "all_per_seed_gates_pass": all(all(g.values()) for g in gates.values()),
        "aggregate_relative_nll_percent": {
            k: 100 * (stats["blockshuffle"]["mean_nll"] / stats[k]["mean_nll"] - 1)
            for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
        },
        "metric_sha256": {
            m["run"]: sha256(root / "runs" / m["run"] / "metrics.json")
            for group in records.values()
            for m in group.values()
        },
        "scope": "Three initialization/data-sampling seeds of locked recipes at 800 steps on the same fixed TinyStories prefix. Seed 17 informed selection; 29/43 replicate it. No convergence, broad-data, equal historical tuning or novelty conclusion.",
    }
    lines = [
        "# Larger-model replication: three seeds, four locked recipes",
        "",
        "The eight fresh runs add seeds 29 and 43 to the earlier width384/layer8 seed 17 cohort. No recipe was selected or dropped using the new results. All comparisons use 800 steps, 1,638,400 training tokens and 32,768 fixed validation targets. Read the [frozen plan](scale_replication_plan.md).",
        "",
        "| Recipe | FFN weights | Total weights | Seed17 NLL | Seed29 NLL | Seed43 NLL | Mean +/- sample SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in RECIPES.items():
        m = records[17][key]
        values = " | ".join(f"{records[s][key]['validation_loss']:.6f}" for s in SEEDS)
        lines.append(
            f"| {label} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {values} | {stats[key]['mean_nll']:.6f} +/- {stats[key]['sample_sd']:.6f} |"
        )
    lines += [
        "",
        "Both compressed recipes retain 70.3125% fewer FFN weights and logical matrix FLOPs, with 42.17% fewer total model weights. The transformer outside the FFN and the training/data code remain unchanged. Preflight verified matching non-FFN initial tensors within each new seed and finite CPU backward for all eight configurations.",
        "",
        "## Paired results and gates",
        "",
        "| Seed | NLL change/full SwiGLU | NLL change/full GELU | NLL change/calibrated narrow | BTT/full training peak ratio | All local gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for seed, group in records.items():
        p = paired[seed]
        memory = (
            group["blockshuffle"]["peak_allocated_vram_bytes"]
            / group["full_swiglu"]["peak_allocated_vram_bytes"]
        )
        lines.append(
            f"| {seed} | {p['full_swiglu']:+.4f}% | {p['full_gelu']:+.4f}% | {p['calibrated_narrow']:+.4f}% | {memory:.4f} | {'PASS' if all(gates[seed].values()) else 'FAIL'} |"
        )
    lines += [
        "",
        "Negative NLL change favors BlockShuffle. Each seed must meet the <=1% limits against both full references, beat calibrated narrow, retain >=70% FFN compression and keep training allocated peak <=1.1x full SwiGLU. All final recorded diagnostics and gradients are finite. Per-seed gate fields are retained in the raw summary; averaging cannot hide a failed seed.",
        "",
        f"The prespecified all-seed decision is **{'PASS' if result['all_per_seed_gates_pass'] else 'FAIL'}**. Aggregate relative NLL changes are "
        + ", ".join(
            f"{RECIPES[k]} {v:+.4f}%" for k, v in result["aggregate_relative_nll_percent"].items()
        )
        + ".",
        "",
        "![Per-seed larger-model loss and memory](../results/plots/scale384_replication.png)",
        "",
        "## Interpretation and outstanding evidence",
        "",
        "This is replication of recipe-level comparisons. BlockShuffle uses factor-aware initialization/LR and native gate recomputation, narrow SwiGLU uses width calibration, and full references use the original dense recipe. Historical tuning effort is unequal and remains a limitation. Seed 17 informed selection; 29/43 test the locked choices. Reusing the same small validation prefix across prior experiments does not establish broad generalization.",
        "",
        "The separate [fused execution audit](fused_execution_results.md) measured 1.206x throughput versus equally compiled full SwiGLU on the original seed 17 checkpoint. The new training runs do not themselves measure fused serving, and equal shapes are not new timing samples. Training still uses the native FFN. The optional condition floor is also a separate post-training treatment.",
        "",
        "An 800-step run sees only about .667 times the training-cache token count in randomly sampled windows; repeated sampling is not an epoch without replacement. This is not convergence. Longer budgets, broader data, equal tuning effort, a credible novelty argument and scoped gradient evidence remain necessary for the original research objective.",
        "",
        "Reproduce with `uv run python -m src.core.scale_replication_report`. The report requires all twelve results and rejects partial cohorts.",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = ("#505a6b", "#cc792b", "#87569b", "#087e8b")
    for i, (key, label) in enumerate(RECIPES.items()):
        values = [records[s][key]["validation_loss"] for s in SEEDS]
        axes[0].plot(range(3), values, "o-", label=label, color=colors[i])
        memory = [records[s][key]["peak_allocated_vram_bytes"] / 2**20 for s in SEEDS]
        axes[1].bar(
            np.arange(3) + (i - 1.5) * 0.2, memory, width=0.19, color=colors[i], label=label
        )
    for ax in axes:
        ax.set_xticks(range(3), [f"Seed {s}" for s in SEEDS])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(
        ylabel="Validation NLL (lower is better)", title="Same 800-step budget for every recipe"
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        ylabel="Peak allocated training memory (MiB)",
        title="Includes model, optimizer and training tensors",
    )
    for suffix in ("png", "svg"):
        fig.savefig(root / "plots" / f"scale384_replication.{suffix}", dpi=100, bbox_inches="tight")
    plt.close(fig)
    write_json(root / "scale384_replication_summary.json", result)
    Path("research/scale_replication_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
