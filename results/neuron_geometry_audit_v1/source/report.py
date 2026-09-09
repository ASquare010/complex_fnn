"""Render the audited H090 endpoint evidence into a readable research report."""

from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read

root = Path("results/neuron_geometry_recovery_v1")
audit = read("results/neuron_geometry_audit_v1/result.json")
assert audit["status"] == "PASS"
result = read(root / "result.json")
summary = result["summary"]
process = read(root / "fitting_process.json")
forms = list(summary["all_ratios_vs_narrow_gelu"])
candidates = list(summary["decisions"])
promoted = [
    name for name in candidates if summary["decisions"][name]["verdict"] == "EARNS_RESOURCE_STUDY"
]
lines = [
    "# H090: can a few learned geometry parameters replace FFN width?",
    "",
    ("Candidates earning a separate resource study: " + ", ".join(promoted) + ".")
    if promoted
    else "**None of the four learned candidates passes the frozen promotion gates.**",
    "The complete research goal remains unmet. This is a controlled synthetic fitting result,",
    "not language-model validation or a claim about all learnable activations.",
    "",
    "## What changed",
    "",
    "A centered rational twist couples pairs of hidden features with an input-dependent",
    "rotation. Four learned amplitudes are shared across the layer. We tested both a",
    "nonzero curved start and an exact linear start, with a fixed twist as an ablation.",
    "This differs from altering each scalar's GELU/ReLU curve independently.",
    "",
    "We also tested the user's local quadratic Bezier correction (one parameter per",
    "layer) and a residual bump with two coefficients in each of four groups (eight",
    "parameters). Both scalar alternatives start exactly as ReLU. Full and narrow",
    "GELU/ReLU/SwiGLU, GroupSort, a four-parameter GELU shift and a linear map serve",
    "as controls. The [equations and prior work](research_direction_2026_09_10.md)",
    "separate activation stability proofs from empirical learning claims.",
    "",
    "## Dataset and budget",
    "",
    "- Four analytic target families: smooth, oscillatory, multiplicative and piecewise.",
    "- Fresh 384-dimensional inputs, seed 9814: 65,536 training examples, 4,096 for",
    "  learning-rate selection and 4,096 held-out reporting examples per task.",
    "- Targets mix cyclic input coordinates, then use a fixed orthogonal output",
    "  rotation and training-only standard-deviation scaling. They are not class labels.",
    "- Each run uses 600 updates, batch 256, and the same sampled index stream for",
    "  each seed. That is 153,600 example presentations per run, about 2.34 training",
    "  set exposures with replacement; these are not ordered epochs.",
    "- Fourteen forms × four tasks × three optimization seeds (17/29/43) × two",
    "  learning rates (0.001/0.003): 336 runs, 201,600 updates, 51,609,600 presentations.",
    "- AdamW, constant base rates, clipping at 1, FP32 CUDA with TF32 off; the narrow",
    "  down-projection rate uses the same declared fan-in calibration for every form.",
    "- One RTX 4070 Laptop GPU. Qualify first, fit once, then audit; no training retry.",
    "",
    f"Fitting process: {process['started_utc']} to {process['finished_utc']};",
    f"**{process['elapsed_seconds'] / 60:.2f} minutes** including dataset preparation, checkpoints, evaluation and plots.",
    "This wall time is not a measurement of GPU kernel time alone.",
    "",
    "## Error, parameters and measured cost",
    "",
    "The error column is the geometric mean of 12 paired held-out MSE ratios",
    "(four tasks × three seeds), relative to narrow GELU. Lower is better; it is",
    "not classification accuracy. Each recipe's rate is selected on its separate",
    "selection split. Both rates and matched-rate comparisons remain in the results.",
    "",
    "| Form | Trainable parameters | Extra activation parameters | MSE change vs narrow GELU | Update ms | Peak MiB |",
    "|---|---:|---:|---:|---:|---:|",
]
for form in forms:
    row = next(r for r in result["selected"] if r["form"] == form)
    resources = summary["resources"][form]
    lines.append(
        f"| {form} | {row['parameters']:,} | {row['activation_parameters']} | "
        f"{100 * (summary['all_ratios_vs_narrow_gelu'][form] - 1):+.3f}% | "
        f"{resources['mean_median_update_ms']:.3f} | {resources['max_peak_bytes'] / 2**20:.2f} |"
    )
lines += [
    "",
    "Full references have 1,179,648 FFN weights; narrow bases have 350,208.",
    "Including their learned controls, the four candidates use 350,209–350,216",
    "parameters: 70.3118–70.3124% fewer than a full reference. These are isolated",
    "FFN counts, not total Transformer parameter reductions.",
    "",
    "Update times are means of per-run medians after 50 warmup updates. They",
    "include forward, backward, clipping and optimizer work, with synchronization;",
    "they exclude batch indexing and gradient clearing. Peak allocation includes",
    "the staged task dataset. Native timing does not establish equally optimized",
    "inference speed. Matrix-only FLOPs exclude activation operations and backward.",
    "",
    "## Per-task results and seed variability",
    "",
    "Held-out MSE, mean ± sample standard deviation over three seeds. The JSON",
    "also records every value, median, sample variance and selected learning rate.",
    "",
    "| Form | Smooth | Oscillatory | Multiplicative | Piecewise |",
    "|---|---:|---:|---:|---:|",
]
for form in forms:
    cells = [f"{v['mean']:.5f} ± {v['sample_sd']:.5f}" for v in summary["per_task"][form].values()]
    lines.append("| " + form + " | " + " | ".join(cells) + " |")
lines += [
    "",
    "## Decisions",
    "",
    "The frozen gate requires at least 2% lower aggregate MSE than **each** of",
    "five narrow controls, improvement in every seed's task aggregate, no task",
    "more than 5% worse than the best control, and at least 70% compression.",
    "Learned twists must additionally beat the fixed twist by at least 1%.",
    "Passing would earn only a separately budgeted resource/width study.",
    "",
]
for form in candidates:
    decision = summary["decisions"][form]
    failed = [name.replace("_", " ") for name, value in decision["gates"].items() if not value]
    lines.append(
        f"- **{form}: {decision['verdict'].replace('_', ' ').lower()}.** "
        + ("Failed: " + "; ".join(failed) + "." if failed else "All screening gates passed.")
    )
lines += [
    "",
    "These decisions close these fixed recipes at this budget. They do not rule",
    "out every paired activation, polynomial basis, scale, sharing scheme or",
    "optimizer. Reopening requires a distinct hypothesis and fresh evidence;",
    "additional training is not automatically allocated.",
    "",
    "## Learned shapes and training diagnostics",
    "",
    "![Learned activation functions and pair map](../results/neuron_geometry_recovery_v1/plots/learned_geometry.png)",
    "![Four shared twist amplitudes during training](../results/neuron_geometry_recovery_v1/plots/twist_evolution.png)",
    "",
    "These figures show seed 17 and explicitly identified tasks/groups. They are",
    "examples of learned functions, not averages or evidence of universal behavior.",
    "The records retain shape coefficients, gradient norms, saturation, activation",
    "statistics and all 600 losses/timings per run. All checkpoints, optimizer",
    "step counts and recorded training losses/gradient norms pass finite checks.",
    "A finite gradient is not proof of well-conditioned learning or convergence.",
    "",
    "## Verification and limitations",
    "",
    "H088 failed its first BF16 independent-gradient comparison before training.",
    "H089 diagnosed the rounding boundary; H090 made one explicit precision",
    "recovery, retaining the original tolerances. All 27 qualification checks",
    "passed. The earlier failed artifacts and six original CPU tensor payloads",
    "were preserved exactly. [Failure and recovery details](neuron_geometry_precision_results.md).",
    "",
    "The independent audit reconstructed the data exactly, verified all 336",
    "initializations and checkpoints, reproduced all 336 selection and 336",
    "reporting scores exactly, checked all 168 selections, and independently",
    "reduced every summary and promotion decision. Its 1e-12 summary tolerance",
    "applies to floating-point reductions; checkpoint scores require exact equality.",
    "",
    "These are four reused analytic task families with fresh inputs and three",
    "optimization seeds, not unrelated real datasets or independent dataset seeds.",
    "We have not shown convergence, the 1% language-NLL allowance, total-model",
    "resource gains, broader-corpus transfer or a breakthrough. No new active",
    "model folder is justified solely by this synthetic screen.",
    "",
    "Sources: [frozen fitting plan](neuron_geometry_plan.md),",
    "[recovery plan](neuron_geometry_recovery_plan.md),",
    "[complete endpoint results](../results/neuron_geometry_recovery_v1/result.json),",
    "[independent audit](../results/neuron_geometry_audit_v1/result.json).",
    "Raw tensors, datasets and histories stay local under the",
    "[artifact policy](ARTIFACTS.md); the compact clone preserves source and conclusions.",
    "",
]
Path("research/neuron_geometry_results.md").write_text("\n".join(lines), encoding="utf-8")
