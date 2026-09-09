"""Generate the H068 report and standalone figure from audited frozen endpoints."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.reproducibility import sha256

ROOT = Path("results/token_activation_fit_v2")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
a = read(Path("results/verification/token_activation_fit_analysis_v1.json"))
assert a["status"] == "PASS"
forms = ("plain", "static", "dynamic", "narrow", "full_swiglu", "full_gelu")
tasks = list(r["task_mean_mse"])
labels = {
    "plain": "Plain BlockShuffle",
    "static": "Static activation",
    "dynamic": "Token-conditioned activation",
    "narrow": "Calibrated narrow SwiGLU",
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
}
params = dict(zip(forms, (5472, 5473, 5521, 5472, 18432, 18432)))
lines = [
    "# H068 - Token activation fitting results",
    "",
    f"**{r['scientific_verdict'].replace('_', ' ')}.** The unchanged H067 forms complete the",
    "fixed seven-target, three-seed, two-rate screen. This result concerns standalone",
    "function fitting; the gold language-model goal remains unmet.",
    "",
    "| Form | Error change vs plain | Error change vs static | Error change vs narrow | Earns full-model resource qualification |",
    "|---|---:|---:|---:|---|",
]
for f in ("static", "dynamic"):
    q = r["ratios"][f]
    earned = r["gates"][f]["earns_full_model_resource_qualification"]
    lines.append(
        f"| {labels[f]} | {100 * (q['plain'] - 1):+.4f}% | {100 * (q['static'] - 1):+.4f}% | {100 * (q['narrow'] - 1):+.4f}% | {'Yes' if earned else 'No'} |"
    )
lines += [
    "",
    "Changes use geometric means of paired held-out MSE ratios across 21 target/seed",
    "cells, after selecting each cell's learning rate on a separate split. Negative",
    "is better. Gates were frozen before training; three seeds do not establish",
    "statistical significance, convergence or a universal approximation advantage.",
    "",
    "![Function fitting and learned-gate ablation](figures/token_activation_fit.png)",
    "",
    "## What was compared",
    "",
    "FFNs with bias-free projections at d48/h256/G8 preserve the main model's width ratios.",
    "The candidate has 5,521 parameters (70.0467% fewer than either 18,432-weight full",
    "control); static has 5,473, plain and narrow 5,472. These are whole standalone",
    "models, not total Transformer parameter counts. Every output has 48 coordinates.",
    "",
    "H070's [routing analysis](rotated_shuffle_theory.md) adds a scale limitation:",
    "k48/G8 has 16 identically zero canonical projection blocks and 48 rank-at-most-1",
    "blocks, whereas k384/G8 has 64 rank-at-most-6 blocks. Relative widths therefore",
    "do not preserve full-size block connectivity. This does not alter H068's frozen",
    "measurements or rejection, and does not automatically reopen its tested recipes.",
    "",
    "The three shared-projection teachers are plain, static and token-conditioned.",
    "Four other targets are smooth, oscillatory, multiplicative and piecewise vector",
    "functions with a fixed output rotation. Input/teacher data are shared across",
    "training seeds: these are optimization replications, not dataset replications.",
    "",
    "All forms receive 300 AdamW updates at each of two base rates (.001/.003), batch",
    "256, identical sampler streams, FP32 with TF32 disabled, eager native Torch and",
    "the existing structured/narrow learning-rate calibration. Initialization has",
    "a target mean row squared norm of one; narrow down uses its corrected fan-in.",
    "Static/dynamic start with exactly the same projection bytes and outputs as plain.",
    "",
    "Each target coordinate is divided by its training standard deviation without",
    "centering. Thus the metric is **variance-scaled MSE**, not centered NMSE or NLL.",
    "The fixed dataset contains 4,096 training, 1,024 rate-selection and 1,024 reporting",
    "rows. Selection uses final checkpoints only and precedes report scoring for both",
    "rates. Both rates' complete artifacts remain available.",
    "",
    "## Held-out errors",
    "",
    "Values are arithmetic means across the three selected seed endpoints. The gates",
    "use paired geometric ratios, so dividing entries in this table is a different",
    "aggregation. Full controls are reported without treating an NLL allowance as an",
    "MSE threshold.",
    "",
    "| Target | Plain | Static | Dynamic | Narrow | Full SwiGLU | Full GELU |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for t in tasks:
    lines.append(
        "| "
        + t.replace("_", " ")
        + " | "
        + " | ".join(f"{r['task_mean_mse'][t][f]:.6f}" for f in forms)
        + " |"
    )
lines += [
    "",
    "The dynamic form remains 8.5808% above full SwiGLU and 23.7703% above full GELU",
    "in paired aggregate error. Its gain on its own teacher is 0.7161%, while error",
    "on the plain teacher increases 0.7436%. This supports retaining baseline-favorable",
    "targets when testing an architecture-derived construction.",
    "",
    "## Frozen decisions",
    "",
]
for f in ("static", "dynamic"):
    failures = [k.replace("_", " ") for k, v in r["gates"][f]["tests"].items() if not v]
    lines.append(
        f"- {labels[f]}: "
        + (
            "fails " + ", ".join(failures) + "."
            if failures
            else "passes every fitting gate; only a separately frozen resource qualification is earned."
        )
    )
if not any(g["earns_full_model_resource_qualification"] for g in r["gates"].values()):
    lines += [
        "",
        "Close these tested fitting recipes. Neither earns a resource worker or language",
        "training, and neither enters the active factory. No extra rate, longer budget,",
        "larger amplitude or richer router follows automatically. Preserve the prototype",
        "and proof as a scoped negative learning result, not a disproof of all learned",
        "or token-conditioned activations.",
    ]
lines += [
    "",
    "## Standalone resource and learning diagnostics",
    "",
    "| Form | Weights | Median update ms | Training peak allocated MiB (range) | Mean clip fraction | Selected rates .001 / .003 |",
    "|---|---:|---:|---:|---:|---:|",
]
for f in forms:
    q = a["resources"][f]
    lines.append(
        f"| {labels[f]} | {params[f]:,} | {q['median_selected_update_ms']:.3f} | {q['min_peak_allocated_mib']:.3f}..{q['max_peak_allocated_mib']:.3f} | {q['mean_clip_fraction']:.4f} | {q['selected_rate_counts']['0.001']} / {q['selected_rate_counts']['0.003']} |"
    )
lines += [
    "",
    "Timing is the median of selected cells' medians after 50 warmup updates, with",
    "CUDA synchronization around each update. Peaks are reset after warmup and include",
    "the small GPU data/index cache. These single-FFN FP32 measurements exclude",
    "attention, embeddings, residual stacks and full-corpus storage. They do not",
    "qualify full-model BF16 memory, training throughput or deployment speed.",
    "",
    "Selected last-50-update training losses remain about 3.9% to 8.2% below the",
    "preceding 50-update averages across forms. This fixed-budget screen does not",
    "demonstrate convergence; no longer run is earned by the failed improvement gates.",
    "",
    "All 252 final weights and optimizer moments are finite. Every 300-step loss and",
    "pre-clip norm is retained, together with projection/gradient and router statistics.",
]
for f in ("static", "dynamic"):
    q = a["resources"][f]
    lines.append(
        f"Resetting only {labels[f].lower()}'s learned gate raises aggregate held-out error by {100 * (q['reset_gate_error_ratio'] - 1):+.4f}%; maximum measured near-bound fraction is {q['max_near_bound_fraction']:.4f}."
    )
lines += [
    "This post-training ablation measures dependence on learned gates after",
    "coadaptation; it does not isolate the cause of a training trajectory difference.",
    "",
    "H067's exact-function nonrepresentation proof remains a real-arithmetic result",
    "about one FFN. It supplies no finite-budget approximation rate or learning",
    "guarantee. Small fitting changes cannot be promoted to a breakthrough claim.",
    "",
    "## Execution, recovery and verification",
    "",
    "The first launch passed four harness checks, then was interrupted before any",
    "completed cell or checkpoint. Its worker and coordinator were absent in two OS",
    "inspections. Its empty first cell, log and stale RUNNING record are preserved",
    "with a separate terminal reconciliation. The number of updates before that",
    "interruption is unknown, bounded by 0..300; it is not recorded as a scientific",
    "failure or silently counted as zero.",
    "",
    "A separately documented one-time recovery runs the unchanged frozen study with",
    "only the output directory overridden, under a hidden coordinator. All regenerated",
    "input, target and sampler tensors match the initial attempt before fitting.",
    "The recovery completes 252 trials and 75,600 updates (19,353,600 training-example",
    "presentations). Including the interrupted cell, total work is bounded by",
    "75,600..75,900 updates and 19,353,600..19,430,400 presentations. No corpus targets",
    "or full-model resource workers are used.",
    "",
    f"Independent CPU rescoring checks all 126 selected checkpoints and 42 gate ablations; maximum relative CPU/GPU MSE difference is {a['max_cpu_gpu_relative_mse_error']:.3g}.",
    "The audit also verifies every checkpoint/moment state, shared initializations,",
    "sampler and data hashes, all 252 histories, selected rates and report chronology.",
    "The active model/source/configuration/test bytes remain unchanged; the existing",
    "108-test active suite and H067's 21 local checks are separate prior qualifications.",
    "",
    "- [Frozen fitting plan](token_activation_fit_plan.md).",
    "- [Recovery exception](token_activation_fit_recovery_plan.md) and [interruption record](../results/token_activation_fit_v1/interruption.json).",
    "- [Complete recovery results](../results/token_activation_fit_v2/result.json) and [process record](../results/token_activation_fit_v2/process.json).",
    "- [Independent analysis](../results/verification/token_activation_fit_analysis_v1.json).",
    "- [Final preservation audit](../results/verification/token_activation_fit_final_v1.json).",
    "- [H067 proof](token_activation_theory.md) and [local qualification](token_activation_results.md).",
    "",
    f"Result SHA-256: `{sha256(ROOT / 'result.json')}`.",
    f"Protocol SHA-256: `{sha256(ROOT / 'protocol.json')}`.",
]
Path("research/token_activation_fit_results.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
x = np.arange(len(tasks))
width = 0.35
for i, (f, color) in enumerate((("static", "#7853a8"), ("dynamic", "#14868c"))):
    values = [100 * (a["task_ratios_to_plain"][t][f] - 1) for t in tasks]
    axes[0].bar(x + (i - 0.5) * width, values, width, label=labels[f], color=color)
    rows = r["selected_rows"]
    values = []
    for t in tasks:
        ratios = [
            row["reset_gate_mse"] / row["heldout_mse"]
            for row in rows
            if row["task"] == t and row["form"] == f
        ]
        values.append(100 * (float(np.exp(np.mean(np.log(ratios)))) - 1))
    axes[1].bar(x + (i - 0.5) * width, values, width, color=color)
for ax in axes:
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, [t.replace("_", "\n") for t in tasks])
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Error change vs plain (%)\nLower is better")
axes[0].set_title("Held-out function fitting: paired geometric ratios across three seeds")
axes[0].legend(loc="best", frameon=False)
axes[1].set_ylabel("Error increase after gate reset (%)")
axes[1].set_title("Post-training gate ablation; no retraining")
fig.suptitle("H068 | One FFN, 300 updates, equal two-rate selection budgets", fontsize=14)
for ext in ("png", "svg"):
    fig.savefig(Path("research/figures") / f"token_activation_fit.{ext}", dpi=140)
print("Report and figure generated from independently audited results.")
