"""Post-audit editorial report and figures; never trains or changes decisions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/input_basis_fit_v1")


def main():
    result = json.loads((ROOT / "result.json").read_bytes())
    process = json.loads((ROOT / "fitting_process.json").read_bytes())
    summary = result["summary"]
    tasks = ("smooth", "oscillatory", "multiplicative", "piecewise")
    forms = (
        "rational",
        "hermite",
        "trig",
        "duplicate",
        "linear",
        "antipodal",
        "raw_gelu",
        "core_linear",
        "full_gelu",
        "full_swiglu",
        "narrow_gelu",
        "narrow_relu",
        "narrow_swiglu",
        "gelu_offset",
    )
    text = [
        "# H094: direct-input nonlinear basis results",
        "",
        "The original gold objective remains unmet. This is a fixed300-update synthetic",
        "screen of learned projections/readouts with fixed nonlinear bases. None of the",
        "three candidates uses a GELU/ReLU/SwiGLU base. Their basis functions have zero",
        "learned shape parameters; the input and output weights learn their combinations.",
        "",
        "[Frozen equations and prior art](input_basis_lift_plan.md),",
        "[fitting plan and decision thresholds](input_basis_fit_plan.md),",
        "[audited machine-readable result](../results/input_basis_fit_v1/result.json).",
        "",
        "## Main comparison",
        "",
        "Ratios are geometric means of12 paired reporting MSE ratios (four tasks, three",
        "seeds), after rate selection on a separate split. Lower is better.",
        "",
        "| Form | MSE vs narrow GELU | Mean median update ms | Weights |",
        "|---|---:|---:|---:|",
    ]
    for form in forms:
        count = (
            147456
            if form == "core_linear"
            else 1179648
            if form.startswith("full_")
            else (350212 if form == "gelu_offset" else 350208)
        )
        text.append(
            f"| {form} | {summary['ratios_vs_narrow_gelu'][form]:.6f} | "
            f"{summary['resources'][form]['median_update_ms']:.3f} | {count:,} |"
        )
    text += [
        "",
        "All direct-input nonlinear designs use70.3125% fewer weights than either full",
        "reference. This does not imply70% lower whole-Transformer size or measured FLOPs.",
        "",
        "## Frozen decisions",
        "",
    ]
    for form, row in summary["decisions"].items():
        failed = ", ".join(k.replace("_", " ") for k, v in row["gates"].items() if not v)
        text += [f"- **{form}: {row['verdict']}**. Failed gates: {failed or 'none'}."]
    text += [
        "",
        "A rejected recipe gets no automatic refinement or longer run. This verdict",
        "is limited to the fixed initialization, optimizer and300-update budget; it is",
        "not an impossibility theorem about rational, polynomial or periodic functions.",
        "",
        "## Per-task reporting MSE",
        "",
        "Mean +/- sample standard deviation over three seeds.",
        "",
        "| Form | Smooth | Oscillatory | Multiplicative | Piecewise |",
        "|---|---:|---:|---:|---:|",
    ]
    for form in forms:
        values = [summary["per_task"][form][t] for t in tasks]
        text.append(
            "| "
            + form
            + " | "
            + " | ".join(f"{v['mean']:.5f} +/- {v['sample_sd']:.5f}" for v in values)
            + " |"
        )
    text += [
        "",
        "Median, sample variance and every seed value are in the public result JSON.",
        "",
        "Zero-predictor reporting MSE: "
        + ", ".join(f"{k}={v:.6f}" for k, v in result["zero_mse"].items())
        + ".",
        "",
        "## Data, training and verification",
        "",
        "Fresh synthetic input seed9844; four previously studied analytic families.",
        "Each task has65,536 training,4,096 rate-selection and4,096 reporting samples,",
        "with384 input/output dimensions. Batch256;300 updates per run (76,800 sampled",
        "presentations, approximately1.17 training-set equivalents, not ordered epochs).",
        "Fourteen forms, three seeds and two rates complete336 runs:100,800 updates and",
        "25,804,800 presentations. No language dataset, real-data test or convergence",
        "claim is part of this experiment.",
        "",
        f"Fitting wall time: {process['elapsed_seconds'] / 60:.2f} minutes, including data,",
        "evaluation and checkpoint work. One RTX4070 Laptop GPU, FP32 native CUDA,",
        "TF32off, four CPU threads. Exact optimizer and initialization differences are",
        "declared in the frozen plan. No hyperparameter extension followed the outcomes.",
        "",
        "Independent audit regenerates data exactly, verifies all336 initializations,",
        "all168 selected rates, histories/counts and all672 checkpoint scores exactly.",
        "Aggregate statistics, paired ratios and every decision are independently checked.",
        "",
        "The H092 launcher had a Windows DLL initialization failure before numerical",
        "tests. H093's explicit single-interpreter attempt passes the unchanged25 checks",
        "and157 saved tensor comparisons, including56 exact checkpoint pairs. The",
        "failure and its logs remain preserved; its root cause is undetermined.",
        "",
        "## Interpretation limits",
        "",
        "The raw-input feature map preserves distances before the final readout. The",
        "readout can still annihilate directions; Transformer residual connections",
        "already supply a direct route. No whole-network gradient guarantee follows.",
        "Duplicate/linear/antipodal controls have algebraic redundancies; optimizer",
        "coordinates and early learning speed can still differ. Better fixed-budget MSE",
        "does not establish a larger function class, novelty or eventual convergence.",
        "",
        "Timing reports synchronized native training, including optimizer update, after",
        "50 warmup iterations. It does not measure inference latency. Peak allocated",
        "GPU bytes and forward/backward timings are in the public result JSON.",
        "",
    ]
    Path("research/input_basis_results.md").write_bytes(("\n".join(text) + "\n").encode())
    folder = ROOT / "plots"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    displayed = ("rational", "hermite", "trig", "duplicate", "raw_gelu", "full_gelu", "narrow_gelu")
    for ax, task in zip(axes.flat, tasks):
        values = [summary["per_task"][f][task] for f in displayed]
        ax.errorbar(
            [v["mean"] for v in values],
            range(len(values)),
            xerr=[v["sample_sd"] for v in values],
            fmt="o",
            capsize=3,
        )
        ax.set(yticks=range(len(values)), yticklabels=displayed, title=task, xlabel="Reporting MSE")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("H094: 300 updates, three seeds; error bars are sample SD")
    fig.savefig(folder / "reporting.png", dpi=115)
    plt.close(fig)


if __name__ == "__main__":
    main()
