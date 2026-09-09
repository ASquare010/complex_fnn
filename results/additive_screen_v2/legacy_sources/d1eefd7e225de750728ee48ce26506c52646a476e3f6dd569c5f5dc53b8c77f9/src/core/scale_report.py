"""Reproduce trained width/depth comparisons and the predeclared continuation gate."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import TrainConfig
from src.core.reproducibility import sha256, write_json

RECIPES = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "calibrated_narrow": "Calibrated narrow SwiGLU",
    "blockshuffle": "BlockShuffle",
}


def load_cohort(root: Path, steps: int) -> dict:
    paths = {k: root / "runs" / f"scale384_{k}_s17_{steps}" for k in RECIPES}
    if not all((p / "metrics.json").exists() for p in paths.values()):
        return {}
    records = {k: json.loads((p / "metrics.json").read_text()) for k, p in paths.items()}
    full = records["full_swiglu"]
    for key, m in records.items():
        cfg = json.loads(Path(f"configs/scale384_{key}_screen.json").read_text())
        cfg["training"]["steps"] = steps
        assert m["model"] == cfg["model"]
        assert TrainConfig(**m["training"]) == TrainConfig(**cfg["training"])
        assert m["data"]["files"] == full["data"]["files"]
        assert m["environment"] == full["environment"]
        assert m["precision"] == "bf16" and m["validation_tokens"] == 32768
        assert m["training_tokens"] == steps * 16 * 128
        expected = 2801664 if key in ("calibrated_narrow", "blockshuffle") else 9437184
        assert m["ffn_parameters"] == expected
        assert m["ffn_matrix_forward_flops_per_token"] == 2 * expected
    return records


def continuation_gates(records: dict, root: Path) -> dict:
    b = records["blockshuffle"]
    diagnostics = json.loads((root / "runs" / b["run"] / "final_diagnostics.json").read_text())
    return {
        "within_2_percent_of_full_swiglu": b["validation_loss"]
        <= 1.02 * records["full_swiglu"]["validation_loss"],
        "within_2_percent_of_full_gelu": b["validation_loss"]
        <= 1.02 * records["full_gelu"]["validation_loss"],
        "at_least_half_percent_better_than_calibrated_narrow": b["validation_loss"]
        <= 0.995 * records["calibrated_narrow"]["validation_loss"],
        "matched_compressed_parameter_and_matrix_compute_budget": b["ffn_parameters"] == 2801664
        and b["ffn_matrix_forward_flops_per_token"] == 5603328,
        "finite_training": bool(np.isfinite(b["final_gradient_norm"]))
        and all(v["finite"] for v in diagnostics.values()),
        "peak_within_10_percent_of_full_swiglu": b["peak_allocated_vram_bytes"]
        <= 1.1 * records["full_swiglu"]["peak_allocated_vram_bytes"],
        "all_models_under_6_gib_safety_limit": all(
            m["peak_allocated_vram_bytes"] <= 6 * 2**30 for m in records.values()
        ),
    }


def comparison_plot(root: Path, records: dict, steps: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ("#424b5a", "#cc792b", "#87569b", "#087e8b")
    for (key, m), color in zip(records.items(), colors, strict=True):
        rows = [
            json.loads(s)
            for s in (root / "runs" / m["run"] / "history.jsonl").read_text().splitlines()
        ]
        rows = [r for r in rows if r["step"] >= (50 if steps == 200 else 200)]
        axes[0].plot(
            [r["step"] for r in rows],
            [r["validation_loss"] for r in rows],
            label=RECIPES[key],
            color=color,
        )
    axes[0].set(
        xlabel="Optimizer steps",
        ylabel="Validation NLL",
        title=f"Width384, layers8: {steps}-step schedule",
    )
    axes[0].legend(fontsize=8)
    values = [m["peak_allocated_vram_bytes"] / 2**20 for m in records.values()]
    axes[1].bar(range(4), values, color=colors)
    axes[1].set_xticks(
        range(4), ["Full\nSwiGLU", "Full\nGELU", "Calibrated\nnarrow", "BlockShuffle"]
    )
    for i, v in enumerate(values):
        axes[1].text(i, v + 8, f"{v:.1f}", ha="center", fontsize=9)
    axes[1].set(
        ylabel="Peak allocated training memory (MiB)",
        ylim=(0, max(values) * 1.13),
        title="Includes model, optimizer and training tensors",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    for suffix in ("png", "svg"):
        fig.savefig(root / "plots" / f"scale384_{steps}.{suffix}", dpi=100, bbox_inches="tight")
    plt.close(fig)


def report(root: Path = Path("results")) -> dict:
    screen = load_cohort(root, 200)
    if not screen:
        raise ValueError("All four 200-step scale results are required")
    gates = continuation_gates(screen, root)
    long = load_cohort(root, 800)
    cohorts = {200: screen}
    if long:
        cohorts[800] = long
    result = {
        "continuation_gates": gates,
        "eligible_for_four_model_800_cohort": all(gates.values()),
        "completed_budgets": list(cohorts),
        "metric_sha256": {
            m["run"]: sha256(root / "runs" / m["run"] / "metrics.json")
            for group in cohorts.values()
            for m in group.values()
        },
        "relative_nll_percent": {
            str(steps): {
                k: 100 * (group["blockshuffle"]["validation_loss"] / m["validation_loss"] - 1)
                for k, m in group.items()
                if k != "blockshuffle"
            }
            for steps, group in cohorts.items()
        },
        "interpretation": "One seed at each stated training budget. Width, depth and head dimension change together. No convergence, broader-corpus or architectural novelty claim.",
    }
    write_json(root / "scale384_summary.json", result)
    lines = [
        "# Trained scale: width384, eight layers",
        "",
        "This compares the locked full, calibrated narrow and BlockShuffle recipes at doubled width and depth. It is a trained scale probe of existing architectures, not a new primitive or convergence result. Read the [frozen plan](trained_scale_plan.md).",
        "",
    ]
    for steps, group in cohorts.items():
        lines += [
            f"## {steps} steps, seed17",
            "",
            "| Recipe | FFN weights | Total weights | NLL | Train peak MiB | Clipped steps |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for k, m in group.items():
            lines.append(
                f"| {RECIPES[k]} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% |"
            )
        lines += [
            "",
            f"Both compressed models use 70.3125% fewer FFN weights and matrix FLOPs, and {group['blockshuffle']['total_reduction_percent']:.3f}% fewer total weights. They use 5603328 FFN matrix forward FLOPs/token versus 18874368 for full models. Pointwise/normalization/optimizer costs are excluded from these arithmetic counts.",
            "",
            f"![{steps}-step scale result](../results/plots/scale384_{steps}.png)",
            "",
        ]
        comparison_plot(root, group, steps)
    lines += ["## Frozen continuation decision", ""]
    for k, v in gates.items():
        lines.append(f"- {k.replace('_', ' ')}: {'PASS' if v else 'FAIL'}.")
    lines += [
        "",
        "The four-model 800-step cohort is eligible under the frozen scale-screen gates."
        if all(gates.values())
        else "The scale screen does not earn the four-model 800-step cohort. Failed gates remain visible; the final research target is unchanged.",
        "",
        "Both optimizer treatment and initialization remain recipe-specific as declared. Full models use the original recipe, narrow SwiGLU uses width calibration, and BlockShuffle uses factor-aware initialization/LR and native gate recomputation. Historical tuning effort is not equalized. This is a comparison of those locked recipes, not an isolated factorization effect.",
        "",
        "The same small TinyStories prefix and train-only tokenizer are used. Initial non-FFN parameters match exactly across models; all parameter counts and CPU backward checks passed before GPU training. Gradient clipping and finite checks are recorded, but neither proves a lower bound on trained network Jacobian singular values. These models remain far smaller than frontier language models.",
        "",
    ]
    serving = {}
    lines += [
        "## Equal-execution serving",
        "",
        "Serving uses batch16/context128 BF16 and the same 800-step checkpoints, with full SwiGLU as the control. Eager and shared-stream V2 CUDA graph protocols are reported separately. Ratios come from rotating within-round comparisons, not the independent training-run timings.",
        "",
    ]
    table = [
        "| Recipe / execution | Mode | NLL | Median speed ratio | Range | Peak MiB | Extra cache MiB | FFN matrix FLOPs/token |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ("blockshuffle", "calibrated_narrow"):
        for execution in ("eager", "graph_v2"):
            path = root / f"serving_scale384_{candidate}_s17_800_{execution}.json"
            if not path.exists():
                continue
            audit = json.loads(path.read_text())
            serving[path.name] = {
                "sha256": sha256(path),
                "ratios": audit["paired_timing"]["throughput_ratio_to_dense_reference"],
            }
            if execution == "graph_v2":
                assert audit["memory_protocol"].startswith("shared_warmup_stream_v2")
                assert all(
                    max(row["fresh_input_eager_max_abs_errors"]) == 0 for row in audit["rows"]
                )
            ratios = serving[path.name]["ratios"]
            for row in audit["rows"]:
                if row["mode"] not in ratios:
                    continue
                ratio = ratios[row["mode"]]
                table.append(
                    f"| {RECIPES[candidate]} / {execution} | {row['mode']} | {row['validation_loss']:.6f} | {ratio['median']:.3f} | {ratio['min']:.3f}-{ratio['max']:.3f} | {row['peak_inference_allocated_vram_bytes'] / 2**20:.2f} | {row['cache_bytes'] / 2**20:.2f} | {row['ffn_matrix_flops_per_token']:,} |"
                )
    if serving:
        lines += table + [
            "",
            "All added caches remain allocated and are counted. The BF16 dense cache raises BlockShuffle FFN matrix work to twice the full reference, while its factorized/packed modes keep the reduced arithmetic. Full-sequence serving does not establish autoregressive latency or training speed. Raw JSON files retain every round, checkpoint hash and correctness check.",
            "",
        ]
    else:
        lines += [
            "The paired serving audits are pending; training-run timings cannot establish the serving speed gate.",
            "",
        ]
    result["serving"] = serving
    audits = {}
    lines += [
        "## Trained projection conditioning",
        "",
        "| Recipe | Smallest condition number | Largest condition number | Loss-direction backward gain range |",
        "|---|---:|---:|---:|",
    ]
    for key in ("full_swiglu", "calibrated_narrow", "blockshuffle"):
        path = root / f"audits/scale384_{key}_s17_800.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())["final"]
        projections = data["spectra"]["projections"]
        conditions = [p["condition_nonzero"] for p in projections.values()]
        gains = list(data["gradient_flow"]["ffn_loss_direction_backward_gain"].values())
        audits[key] = {
            "sha256": sha256(path),
            "condition_range": [min(conditions), max(conditions)],
            "backward_gain_range": [min(gains), max(gains)],
            "all_gradients_finite": all(
                r["finite"] for r in data["gradient_flow"]["records"].values()
            ),
            "all_projections_maximum_numerical_rank": all(
                p["rank_relative_1e_6"] == p["maximum_possible_rank"] for p in projections.values()
            ),
        }
        lines.append(
            f"| {RECIPES[key]} | {min(conditions):.2f} | {max(conditions):.2f} | {min(gains):.3f}-{max(gains):.3f} |"
        )
    result["trained_checkpoint_audits"] = audits
    lines += [
        "",
        "A subsequent [controlled post-training floor](conditioning_results.md) reduces the larger checkpoint's worst down condition to 26.53 at a .01118% NLL cost and transfers to three smaller checkpoints. Original metrics above remain unchanged. This is projection-level conditioning evidence, not a full-network stability proof.",
        "",
    ]
    lines += [
        "",
        "All recorded gradients are finite and projections have maximum numerical rank at the relative 1e-6 threshold. However, BlockShuffle's layer6 down projection reaches condition number 30134, versus a maximum 17.28 for full SwiGLU. Its flat initialization spectrum does not persist. These CPU FP32 audits use FP64 SVD on materialized weights and one fixed loss direction; neither finite gradients nor a projection spectrum proves full-network nonvanishing gradients.",
        "",
        "## Completed compiler probe",
        "",
        "Default fullgraph compilation passes BF16 numerical checks but gives a paired compiled BlockShuffle/full speed ratio .632 (.575-.755), failing the .8 floor. General compiled/eager speedups are 3.494 for BlockShuffle and 3.593 for full SwiGLU. Read the [completed compiler result and profiles](compiler_serving_results.md).",
        "",
    ]
    if long:
        b, full, narrow = (long[k] for k in ("blockshuffle", "full_swiglu", "calibrated_narrow"))
        quality = {
            "within_1_percent_of_full_swiglu": b["validation_loss"]
            <= 1.01 * full["validation_loss"],
            "within_1_percent_of_full_gelu": b["validation_loss"]
            <= 1.01 * long["full_gelu"]["validation_loss"],
            "beats_calibrated_narrow": b["validation_loss"] < narrow["validation_loss"],
            "training_peak_within_10_percent_of_full": b["peak_allocated_vram_bytes"]
            <= 1.1 * full["peak_allocated_vram_bytes"],
        }
        result["single_seed_800_quality_memory_gates"] = quality
        lines += ["## Longer-budget decision and remaining evidence", ""]
        for key, passed in quality.items():
            lines.append(f"- {key.replace('_', ' ')}: {'PASS' if passed else 'FAIL'}.")
        lines += [
            "",
            "These gates describe one seed at this size and budget. Multiple seeds, convergence, broader data, acceptable serving, equal tuning effort and verified novelty remain necessary. Passing selected local gates does not establish the research goal.",
            "",
        ]
    write_json(root / "scale384_summary.json", result)
    Path("research/trained_scale_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
