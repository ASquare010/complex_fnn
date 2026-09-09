"""Reproduce the frozen paired-feature screen and its promotion decision."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.report import cohort
from src.core.reproducibility import sha256, write_json

RUNS = {
    "Antipodal pair": "paired_antipodal_swiglu_h228_s17_200",
    "Reciprocal pair": "paired_reciprocal_swiglu_h228_s17_200",
    "Duplicate control": "paired_duplicate_swiglu_h228_s17_200",
    "Narrow SwiGLU": "swiglu_narrow_h152_s17_200",
    "Matched GELU": "gelu_h228_s17_200",
}


def report(root: Path = Path("results")) -> dict:
    records = {
        label: json.loads((root / "runs" / name / "metrics.json").read_text(encoding="utf-8"))
        for label, name in RUNS.items()
    }
    assert len({cohort(m) for m in records.values()}) == 1, "Unexpected protocol differences"
    assert all(
        m["training"]["seed"] == 17 and m["training"]["steps"] == 200 for m in records.values()
    )
    assert all(
        m["ffn_parameters"] == 350208 and m["total_parameters"] == 1728192 for m in records.values()
    )
    assert all(m["ffn_matrix_forward_flops_per_token"] == 700416 for m in records.values())
    narrow, gelu, duplicate = (
        records[k]["validation_loss"]
        for k in ("Narrow SwiGLU", "Matched GELU", "Duplicate control")
    )
    reference = json.loads((root / "runs" / "baseline_swiglu_s17_200" / "metrics.json").read_text())
    decisions = {}
    for label in ("Antipodal pair", "Reciprocal pair"):
        m = records[label]
        gates = {
            "at_least_1pct_better_than_matched_swiglu": m["validation_loss"] <= 0.99 * narrow,
            "within_1pct_of_matched_gelu": m["validation_loss"] <= 1.01 * gelu,
            "better_than_duplicate_control": m["validation_loss"] < duplicate,
            "training_peak_within_10pct_of_full_reference": m["peak_allocated_vram_bytes"]
            <= 1.10 * reference["peak_allocated_vram_bytes"],
            "finite_final_gradient": bool(np.isfinite(m["final_gradient_norm"])),
        }
        decisions[label] = {
            "gates": gates,
            "eligible": all(gates.values()),
            "relative_nll_percent_vs_narrow": 100 * (m["validation_loss"] / narrow - 1),
            "relative_nll_percent_vs_gelu": 100 * (m["validation_loss"] / gelu - 1),
        }
    eligible = [label for label, d in decisions.items() if d["eligible"]]
    selected = (
        min(eligible, key=lambda label: records[label]["validation_loss"]) if eligible else None
    )
    functions = json.loads((root / "paired_functions_v1" / "summary.json").read_text())
    result = {
        "runs": RUNS,
        "metric_sha256": {
            label: sha256(root / "runs" / name / "metrics.json") for label, name in RUNS.items()
        },
        "decisions": decisions,
        "selected_for_800_steps": selected,
        "function_summary_sha256": sha256(root / "paired_functions_v1" / "summary.json"),
        "interpretation": "One seed, frozen 200-step screening budget and selected function diagnostics. Same optimizer recipe, parameter count and matrix compute. No general capability, long-budget superiority or novelty claim.",
    }
    write_json(root / "paired_screen_summary.json", result)
    colors = dict(zip(RUNS, ("#087e8b", "#cf7125", "#8a6a9f", "#424b5a", "#74854e"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, m in records.items():
        history = [
            json.loads(x)
            for x in (root / "runs" / m["run"] / "history.jsonl").read_text().splitlines()
        ]
        history = [x for x in history if x["step"] >= 50]
        axes[0].plot(
            [x["step"] for x in history],
            [x["validation_loss"] for x in history],
            label=label,
            color=colors[label],
        )
    axes[0].set(
        xlabel="Training steps (50 onward)",
        ylabel="Validation NLL",
        title="TinyStories: matched parameter/compute screen",
    )
    axes[0].legend(fontsize=8)
    tasks = ("constructed_polynomial", "additive_control", "pure_product")
    lookup = {(r["variant"], r["task"]): r for r in functions["rows"]}
    for i, (label, m) in enumerate(list(records.items())[:4]):
        values = [lookup[(m["model"]["variant"], task)]["heldout_normalized_mse"] for task in tasks]
        axes[1].bar(
            np.arange(3) + (i - 1.5) * 0.19, values, width=0.18, color=colors[label], label=label
        )
    axes[1].set_xticks(range(3), ("Polynomial", "Additive", "Product"))
    axes[1].set(
        yscale="log",
        ylabel="Held-out normalized MSE",
        title="Function diagnostics: 300 steps, seed 17",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2, axis="y")
    fig.savefig(root / "plots" / "paired_feature_screen.png", dpi=100)
    fig.savefig(root / "plots" / "paired_feature_screen.svg")
    plt.close(fig)
    lines = [
        "# Paired-feature FFN: frozen screen results",
        "",
        "The exact signed-feature identities and scalar derivative bound pass independent checks. Their scope is limited to the feature map before the output projection; the full Jacobian can be singular. The language-model results determine whether the mechanism earns more compute.",
        "",
        "## Matched language-model comparison",
        "",
        "All five runs use seed 17, 200 steps, the same data order, 409,600 training tokens and 32,768 validation targets. They share uniform AdamW settings, precision and all non-FFN dimensions. Every FFN has 350,208 unique weights, 700,416 matrix FLOPs per token across four layers, and 1,728,192 total model weights. Pair modes and GELU also share the exact initial projection tensors.",
        "",
        "| Method | Validation NLL | Peak training MiB | Training tokens/s | Full-sequence forward tokens/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, m in records.items():
        lines.append(
            f"| {label} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {m['training_tokens_per_second']:,.0f} | {m['inference_tokens_per_second']:,.0f} |"
        )
    lines += [
        "",
        "Timings here are separate-run screening observations. Laptop clocks and kernel overhead can change them; they do not establish a deployment-speed win. The [frozen plan](paired_feature_plan.md) defines the promotion gates before measurements.",
        "",
        "## Promotion decision",
        "",
    ]
    for label, d in decisions.items():
        gate_names = {
            "at_least_1pct_better_than_matched_swiglu": "less than 1% improvement over matched SwiGLU",
            "within_1pct_of_matched_gelu": "more than 1% behind matched GELU",
            "better_than_duplicate_control": "worse than the duplicate control",
            "training_peak_within_10pct_of_full_reference": "training memory exceeds the limit",
            "finite_final_gradient": "nonfinite final gradient",
        }
        failed = [gate_names[k] for k, ok in d["gates"].items() if not ok]
        lines.append(
            f"- {label}: NLL difference versus matched SwiGLU {d['relative_nll_percent_vs_narrow']:+.3f}%, versus GELU {d['relative_nll_percent_vs_gelu']:+.3f}%. "
            + ("Passes the screen." if not failed else "Failed gates: " + ", ".join(failed) + ".")
        )
    lines += [
        "",
        f"Selected for one 800-step check: **{selected or 'none'}**. This is a screening decision, not final acceptance.",
        "",
        "## Function diagnostics",
        "",
        "Each regressor has 49 learned parameters, including the same scalar bias. The three chosen targets are small mechanism diagnostics; an exact product identity should help the product task, but that is not evidence of broad language capability. This table retains every seed-17 result.",
        "",
        "| Method | Polynomial MSE | Additive MSE | Product MSE |",
        "|---|---:|---:|---:|",
    ]
    for label, m in list(records.items())[:4]:
        values = " | ".join(
            f"{lookup[(m['model']['variant'], task)]['heldout_normalized_mse']:.7f}"
            for task in tasks
        )
        lines.append(f"| {label} | {values} |")
    lines += [
        "",
        "The duplicate map collapses algebraically to ordinary SwiGLU at half the feature width, with the two output-weight blocks added. Its optimizer dynamics and redundant parameter count differ from a minimal narrow implementation. It checks whether the second feature carries useful function diversity.",
        "",
        "![Learning curves and function diagnostic errors](../results/plots/paired_feature_screen.png)",
        "",
        "Raw [decision JSON](../results/paired_screen_summary.json) records all metric hashes and gates. Function inputs, checkpoints, histories and source snapshot are in results/paired_functions_v1. No verified novelty, general superiority, larger-scale stability or gold result is established. Regenerate this report with `uv run python -m src.core.paired_report`.",
        "",
    ]
    Path("research/paired_feature_results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    report()
