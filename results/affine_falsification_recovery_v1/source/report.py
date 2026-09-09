"""Post-audit report and plots; consumes frozen measurements only."""

import json
import statistics as stats
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/affine_falsification_recovery_v1")


def main():
    result = json.loads((ROOT / "result.json").read_bytes())
    prior = json.loads(Path("results/input_basis_fit_v1/result.json").read_bytes())
    assert result["status"] == "PASS"
    summary = result["summary"]
    ratio = summary["neural_ratios_vs_affine"]["duplicate"]
    tasks = ("smooth", "oscillatory", "multiplicative", "piecewise")
    text = [
        "# H096-H098: a stronger affine control changes the interpretation",
        "",
        f"**The even-feature lead improves aggregate reporting MSE by only {100 * (1 - ratio):.3f}%",
        "over an ordinary affine least-squares fit, missing the frozen 2% material-benefit",
        "threshold.** Close the general nonlinear-benefit claim for this 300-update recipe.",
        "Its earlier 12.92% advantage over narrow GELU remains a correct measurement;",
        "it was an incomplete comparison. No benchmark result or checkpoint is erased.",
        "",
        "The lead uses 350,208 trained coefficients, folded to 282,624 for inference.",
        "The affine comparator needs 147,840 coefficients, including its intercept.",
        "The previously measured native inference speed deficit also remains unresolved.",
        "",
        "## What survives, and what does not",
        "",
        "| Task | Affine MSE | Even-feature MSE | Even / affine |",
        "|---|---:|---:|---:|",
    ]
    for task in tasks:
        a = summary["per_task"]["affine"][task]["mean"]
        b = prior["summary"]["per_task"]["duplicate"][task]["mean"]
        text.append(f"| {task} | {a:.6f} | {b:.6f} | {b / a:.6f} |")
    text += [
        "",
        "The even-feature model loses on three tasks and wins on piecewise targets",
        f"by {100 * (1 - summary['even_lead_task_ratios']['piecewise']):.2f}%. Retain that as a scoped",
        "observation, not a general architecture promotion. All three optimization seeds",
        "have a small aggregate win, so this is not a claim that no nonlinear learning",
        "occurred. The specified material-benefit gate fails; the other two gates pass.",
        "",
        "Mean seed ratios: "
        + ", ".join(f"{v:.6f}" for v in summary["even_lead_seed_ratios"])
        + ".",
        "These are optimization/sampling seeds on one fixed dataset, not three independent",
        "datasets. All comparisons are post-selection diagnostics on previously used data.",
        "",
        "## Full conventional comparison",
        "",
        "Ratio below one favors the neural recipe; above one favors the affine fit.",
        "",
        "| Neural recipe | Aggregate MSE / affine |",
        "|---|---:|",
    ]
    for form, value in summary["neural_ratios_vs_affine"].items():
        text.append(f"| {form} | {value:.6f} |")
    text += [
        "",
        "The large bias-free smooth-target error was a warning. Adding an intercept to",
        "least squares changes its mean MSE from 2.229781 to 0.429837. A nonlinear feature",
        "can provide a nonzero mean, so comparison only with a bias-free linear baseline",
        "does not isolate nonlinear expressivity. This is a measured control gap, not a",
        "proof that every neural difference is caused by its treatment of the mean.",
        "",
        "## Structural limit and observed errors",
        "",
        "For an even basis E, the model f(x)=A*x+B*E(U*x) has odd component A*x.",
        "The direct matrix can fit linear trends, but no width increase in that even bank",
        "can produce a nonlinear odd response in a single such layer. This extends the",
        "[existing parity analysis](ungated_blockshuffle_theory.md) to the folded lead.",
        "It says nothing about biased/deeper networks or a complete Transformer.",
        "",
        "Antithetic evaluation pairs x with -x and splits the squared error into odd",
        "and even components. This separate diagnostic does not replace the original",
        "reporting split metric.",
        "",
        "| Task | Even-feature odd error | Even-feature even error |",
        "|---|---:|---:|",
    ]
    for task in tasks:
        selected = [r for r in result["parity"] if r["task"] == task and r["form"] == "duplicate"]
        text.append(
            f"| {task} | {stats.mean(r['odd_mse'] for r in selected):.6f} | "
            f"{stats.mean(r['even_mse'] for r in selected):.6f} |"
        )
    text += [
        "",
        "For the cyclic multiplicative target, the known population odd-error floor",
        "is 12/17 of zero-predictor error under the specified independent uniform",
        "distribution. That is not an exact lower bound on this finite report set.",
        "",
        "## Experimental scope and verification",
        "",
        "We fit 24 linear/affine baselines and 12 constant estimates. For each seed, the",
        "fit uses exactly the multiset of 76,800 samples seen by its 300 batch-256 updates,",
        "including duplicate counts. No unsampled row has positive weight. Selection and",
        "reporting labels never enter the solve. No new neural optimizer update occurs.",
        "",
        "The least-squares solver differs from AdamW, so this tests a stronger function",
        "class baseline rather than an equal-optimizer training comparison. FP64 weighted",
        "statistics are computed on CUDA; the recovered small matrix solve uses CPU.",
        "FP32-exported coefficients supply the primary error comparison. The affine",
        "model is a diagnostic control, not the proposed nonlinear FFN replacement.",
        "",
        "Independent audit reconstructs the sample counts and weighted statistics,",
        "verifies all 24 solutions, checks 96 CPU scores, recomputes all 168 neural",
        "parity evaluations and rechecks 12 saved odd-linearity tensor pairs. Mean,",
        "median, sample variance and standard deviation are retained in the result JSON.",
        "",
        "H096 ended with native access violation 0xC0000005 before saving a completed fit.",
        "H097's isolated CPU/CUDA small-matrix probe passed; it did not reproduce or",
        "diagnose the failure. H098 is a separately frozen recovery using CPU small-matrix",
        "linalg, identical data/tolerances/decisions and phase logging. The old source,",
        "stale RUNNING status, terminal-process receipt and empty fit directory remain",
        "preserved. The backend change is not a proven explanation of the crash.",
        "",
        "## Next research direction",
        "",
        "Do not optimize kernels for this weak general lead. The next assay needs to",
        "measure improvement on the nonlinear residual after accounting for affine",
        "structure, with an explicit representable positive control. Include established",
        "strong nonlinear controls such as squared ReLU",
        "([Primer](https://arxiv.org/abs/2109.08668)) and its learned scale/bias relatives",
        "([StarReLU](https://arxiv.org/abs/2210.13452)). These published mechanisms are",
        "prior art, not new local results or a justified claim of novelty.",
        "",
        "The goal still requires parameter-efficient nonlinear learning, comparable compute,",
        "language NLL, multiple seeds, longer training, scale and broader data. None of",
        "those missing requirements is replaced by a passed least-squares diagnostic.",
        "",
        "[Frozen diagnostic plan](affine_falsification_plan.md),",
        "[explicit recovery plan](affine_falsification_recovery_plan.md),",
        "[audited result](../results/affine_falsification_recovery_v1/result.json).",
        "",
    ]
    Path("research/affine_falsification_results.md").write_bytes(("\n".join(text) + "\n").encode())
    folder = ROOT / "plots"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    displayed = (
        "duplicate",
        "hermite",
        "antipodal",
        "rational",
        "trig",
        "raw_gelu",
        "full_gelu",
        "narrow_gelu",
        "full_swiglu",
    )
    values = [summary["neural_ratios_vs_affine"][f] for f in displayed]
    axes[0].barh(displayed, values, color=["#257a66" if v < 1 else "#8c9eb5" for v in values])
    axes[0].axvline(1, c="black", linewidth=1, label="Affine baseline")
    axes[0].axvline(0.98, c="#b35735", linestyle="--", label="2% benefit threshold")
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0.9, 1.35), xlabel="Reporting MSE / affine", title="Aggregate comparison")
    axes[0].legend(fontsize=8)
    for name, marker in (("affine", "o"), ("duplicate", "s")):
        vals = [
            summary["per_task"]["affine"][t]["mean"]
            if name == "affine"
            else prior["summary"]["per_task"]["duplicate"][t]["mean"]
            for t in tasks
        ]
        axes[1].plot(range(4), vals, marker=marker, label=name, linewidth=1)
    axes[1].set(xticks=range(4), xticklabels=tasks, ylabel="Reporting MSE", title="Per-task means")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("H096-H098: matched training samples; three seeds; different solvers")
    fig.savefig(folder / "affine_comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
