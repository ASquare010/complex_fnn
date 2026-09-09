"""Reproduce the quadratic Bezier screen, controls, diagnostics and frozen gates."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.report import cohort
from src.core.reproducibility import sha256, write_json

RUNS = {
    "Direct quadratic": "quadratic_direct_s17_200",
    "Three-branch products": "quadratic_mixed_s17_200",
    "Shared SwiGLU correction": "quadratic_residual_shared_s17_200",
    "Grouped SwiGLU correction": "quadratic_residual_grouped_s17_200",
    "Calibrated direct quadratic": "quadratic_direct_calibrated_s17_200",
    "Original GELU h228": "gelu_h228_s17_200",
    "Calibrated GELU h228": "gelu_h228_calibrated_s17_200",
    "Calibrated SwiGLU h152": "swiglu_h152_width_init_and_lr_s17_200",
    "Full SwiGLU h512": "baseline_swiglu_s17_200",
}
CANDIDATES = tuple(RUNS)[:5]


def report(root: Path = Path("results")) -> dict:
    paths = {label: root / "runs" / run for label, run in RUNS.items()}
    records = {label: json.loads((p / "metrics.json").read_text()) for label, p in paths.items()}
    for labels in (
        ("Direct quadratic", "Three-branch products", "Original GELU h228", "Full SwiGLU h512"),
        (
            "Shared SwiGLU correction",
            "Grouped SwiGLU correction",
            "Calibrated direct quadratic",
            "Calibrated GELU h228",
            "Calibrated SwiGLU h152",
        ),
    ):
        assert len({cohort(records[k]) for k in labels}) == 1, "Changed control recipe"
    assert all(
        m["training"]["steps"] == 200
        and m["training"]["seed"] == 17
        and m["validation_tokens"] == 32768
        for m in records.values()
    )
    full = records["Full SwiGLU h512"]
    narrow = records["Calibrated SwiGLU h152"]["validation_loss"]
    decisions, diagnostics = {}, {}
    for label in CANDIDATES:
        m = records[label]
        gelu_label = (
            "Calibrated GELU h228"
            if label == "Calibrated direct quadratic"
            else "Original GELU h228"
        )
        gelu = records[gelu_label]["validation_loss"]
        d = json.loads((paths[label] / "final_diagnostics.json").read_text())
        activations = {k: v for k, v in d.items() if "quadratic_activation" in k}
        assert len(activations) == 4
        assert m["ffn_matrix_forward_flops_per_token"] == 700416
        gates = {
            "at_least_half_percent_better_than_calibrated_swiglu": m["validation_loss"]
            <= 0.995 * narrow,
            "at_least_half_percent_better_than_gelu_control": m["validation_loss"] <= 0.995 * gelu,
            "over_70_percent_ffn_reduction": m["ffn_reduction_percent"] > 70,
            "peak_within_10_percent_of_full": m["peak_allocated_vram_bytes"]
            <= 1.1 * full["peak_allocated_vram_bytes"],
            "finite_outputs_and_gradients": all(r["finite"] for r in d.values())
            and bool(np.isfinite(m["final_gradient_norm"])),
        }
        if label == "Three-branch products":
            gates["at_least_point_two_percent_better_than_unmixed"] = (
                m["validation_loss"] <= 0.998 * records["Direct quadratic"]["validation_loss"]
            )
        decisions[label] = {
            "gates": gates,
            "eligible": all(gates.values()),
            "gelu_control": gelu_label,
            "relative_nll_percent_vs_calibrated_swiglu": 100 * (m["validation_loss"] / narrow - 1),
            "relative_nll_percent_vs_gelu_control": 100 * (m["validation_loss"] / gelu - 1),
        }
        diagnostics[label] = {
            "activations": activations,
            "branch_products": {
                k: v["branch_product_coefficients"]
                for k, v in d.items()
                if "branch_product_coefficients" in v
            },
        }
    eligible = [k for k, d in decisions.items() if d["eligible"]]
    selected = min(eligible, key=lambda k: records[k]["validation_loss"]) if eligible else None
    functions = json.loads((root / "quadratic_functions_v1" / "summary.json").read_text())
    result = {
        "runs": RUNS,
        "metric_sha256": {k: sha256(p / "metrics.json") for k, p in paths.items()},
        "decisions": decisions,
        "selected_for_800_steps": selected,
        "diagnostics": diagnostics,
        "function_summary_sha256": sha256(root / "quadratic_functions_v1" / "summary.json"),
        "interpretation": "Single-seed short-budget screens. Initial and follow-up gates use their explicitly frozen GELU control. Matrix FLOPs exclude nonlinear work. Cross-date timing is not a controlled relative speed comparison.",
    }
    write_json(root / "quadratic_screen_summary.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    labels = list(RUNS)
    colors = ["#087e8b" if k in CANDIDATES else "#7c8492" for k in labels]
    values = [records[k]["validation_loss"] for k in labels]
    axes[0].scatter(values, range(len(labels)), c=colors, s=45)
    for i, value in enumerate(values):
        axes[0].annotate(
            f"{value:.4f}",
            (value, i),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set(
        xlim=(4.05, 4.33),
        xlabel="Validation NLL (lower is better)",
        title="TinyStories: 200 steps, seed 17",
    )
    axes[0].axvline(full["validation_loss"], color="#7c8492", ls="--", alpha=0.5)
    axes[0].grid(axis="x", alpha=0.2)
    variants = (
        "quadratic_bezier",
        "quadratic_bezier_mixed",
        "swiglu_quadratic",
        "gelu",
        "swiglu_narrow",
    )
    tasks = ("constructed_polynomial", "additive_control", "pure_product")
    values = np.array(
        [
            [
                next(
                    r["heldout_normalized_mse"]
                    for r in functions["rows"]
                    if r["variant"] == v and r["task"] == t
                )
                for t in tasks
            ]
            for v in variants
        ]
    )
    from matplotlib.colors import LogNorm

    plot = axes[1].imshow(values, norm=LogNorm(), cmap="YlOrRd", aspect="auto")
    axes[1].set_yticks(
        range(5),
        ["Direct (55 weights)", "Products (58)", "Residual (51)", "GELU (49)", "SwiGLU (49)"],
        fontsize=8,
    )
    axes[1].set_xticks(range(3), ["Polynomial", "Additive", "Product"])
    for i in range(5):
        for j in range(3):
            axes[1].text(
                j,
                i,
                f"{values[i, j]:.4f}",
                ha="center",
                va="center",
                color="white" if values[i, j] > 0.06 else "black",
                fontsize=9,
            )
    axes[1].set_title("Selected function tasks: 300 steps, seed 17")
    fig.colorbar(plot, ax=axes[1], label="Held-out normalized MSE")
    (root / "plots").mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(root / "plots" / f"quadratic_bezier_screen.{extension}", dpi=100)
    plt.close(fig)
    lines = [
        "# Quadratic Bezier: measured screen and follow-up",
        "",
        "No quadratic candidate passes its frozen promotion gates. The best residual gives only a 0.095% NLL improvement over calibrated SwiGLU. Applying width calibration to both direct quadratic and GELU improves both, with GELU remaining better. The full research target is unmet.",
        "",
        "The requested equation and three-branch products are implemented in [quadratic.py](../src/bezier_ffn/quadratic.py). Read the [frozen plan and scoped proofs](quadratic_bezier_plan.md). All runs use the common trainer, fixed TinyStories data and 32768 validation targets. Initial and follow-up optimizer recipes differ as declared; no candidate is selected using a changed threshold.",
        "",
        "## Language-model screen",
        "",
        "| Recipe | FFN weights | Total weights | NLL | Train peak MiB | Clipped steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, m in records.items():
        lines.append(
            f"| {label} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% |"
        )
    lines += [
        "",
        "All compressed rows use 700416 FFN matrix forward FLOPs/token; full SwiGLU uses 2359296. Learned scalar coefficients are included in weights. Nonlinear operations, temporary tensors and casts are excluded from those FLOP counts, but included in measured memory and timing.",
        "",
        "## Frozen decisions",
        "",
        "| Candidate | NLL change vs calibrated SwiGLU | NLL change vs its GELU control | Failed gates |",
        "|---|---:|---:|---|",
    ]
    gate_labels = {
        "at_least_half_percent_better_than_calibrated_swiglu": ">=.5% gain over SwiGLU",
        "at_least_half_percent_better_than_gelu_control": ">=.5% gain over GELU",
        "at_least_point_two_percent_better_than_unmixed": ">=.2% gain from products",
    }
    for label, d in decisions.items():
        failed = "; ".join(gate_labels.get(k, k) for k, v in d["gates"].items() if not v) or "None"
        lines.append(
            f"| {label} | {d['relative_nll_percent_vs_calibrated_swiglu']:+.3f}% | {d['relative_nll_percent_vs_gelu_control']:+.3f}% | {failed} |"
        )
    lines += [
        "",
        "The initial four settings use original matched GELU for their frozen gate; calibrated direct uses equally calibrated GELU. Every quadratic setting also loses to the stronger calibrated GELU found in the follow-up. None earns an 800-step candidate run.",
        "",
        "## Mechanism and stability",
        "",
        "The direct quadratic has bounded outputs and an upper slope bound, but vanishing sigmoid tails. Three pair-product branches add a scoped mixed derivative; zero coefficients exactly recover the unmixed model. The residual begins exactly at SwiGLU, including BF16 outputs and initial common-weight gradients. Tests verify these statements; they do not prove universal expressivity or stable trained depth.",
        "",
        "On the fixed final diagnostic batch, no quadratic module has t<.01 or t>.99, and no curve control reaches 98% of its bound. The direct and mixed controls remain close to initialization. Thus this sample does not support attributing their poorer NLL to sigmoid tail saturation. Complete per-layer controls, slope fractions, branch coefficients, activation statistics and gradient logs are retained. Older initial-screen raw keys named `activation_slope_below_1e3_fraction` mean |slope|<0.001; the follow-up source spells this threshold unambiguously.",
        "",
        "The three-branch version improves all three selected function errors against direct quadratic but worsens LM NLL by 0.191%. Function-fit improvement therefore does not establish language-model improvement. Toy counts are 55/58/51 for direct/mixed/residual versus 49 for controls, so their coefficient overhead is explicitly material.",
        "",
        "![Quadratic screen and function diagnostics](../results/plots/quadratic_bezier_screen.png)",
        "",
        "## Reproduction and limits",
        "",
        "Run `uv run python -m src.core.quadratic_report` to regenerate the gates, hashes and this figure. Each candidate checkpoint also has `learned_curves.json` and `learned_curves.png`. The plot shows scalar activations before product mixing/value gating; full mixed behavior is not a scalar curve.",
        "",
        "Timing changed sharply despite identical recorded GPU/software. The unchanged calibrated SwiGLU repeat is retained under `results/reproducibility/width_calibrated_200_repeat`; its final NLL differs by about 0.00000745. The comparison audit records complete logged differences and checkpoint differences. Consequently numerical trajectory bitwise identity and speed gains across these run dates are not claimed. All training runs were serial; absolute synchronized timing remains available in raw metrics.",
        "",
        "These are elementary Bernstein-basis and gated-activation experiments with substantial prior art, not verified architectural novelty. The [plan](quadratic_bezier_plan.md) links primary sources. No broader-corpus, convergence or trained-scale claim follows from this screen.",
    ]
    audit_paths = [
        root / "audits" / f"{name}.json"
        for name in ("quadratic_direct_calibrated_s17_200", "gelu_h228_calibrated_s17_200")
    ]
    if all(p.exists() for p in audit_paths):
        lines += [
            "",
            "## Paired checkpoint gradient audit",
            "",
            "The calibrated direct curve and GELU control were audited on the same 256-target CPU FP32 batch, before and after training. Every recorded activation gradient is finite and all eight projection matrices per model retain maximal numerical rank. Final nonzero condition numbers span 23.42-73.83 for quadratic and 22.44-67.77 for GELU. FFN backward gains in that loss direction span .097-.221 and .111-.220 respectively. These local measurements show no demonstrated conditioning advantage; they are not extremal Jacobian singular values or a deep-network stability guarantee.",
            "",
            "Raw [quadratic audit](../results/audits/quadratic_direct_calibrated_s17_200.json) and [GELU audit](../results/audits/gelu_h228_calibrated_s17_200.json) retain all spectra and invocation-specific gradients.",
        ]
    long_path = root / "runs" / "gelu_h228_calibrated_s17_800" / "metrics.json"
    if long_path.exists():
        m = json.loads(long_path.read_text())
        lines += [
            "",
            "## Longer-budget conventional control",
            "",
            f"The separately frozen calibrated GELU h228 control reaches NLL {m['validation_loss']:.6f} at 800 steps, seed 17. At the same budget, calibrated SwiGLU h152 scores 3.126104, BlockShuffle 3.098956, full SwiGLU 3.065045 and full GELU 3.193122. Its strong 200-step relative ranking does not persist. This is a conventional-control check; no failed quadratic candidate was promoted.",
        ]
    Path("research/quadratic_bezier_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(report()["decisions"], indent=2))
