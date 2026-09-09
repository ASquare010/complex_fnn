"""Render H071 from audited endpoints; no scientific dependency needed for hashing."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path("results/rotated_shuffle_fit_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


r = read(ROOT / "result.json")
a = read("results/verification/rotated_shuffle_fit_analysis_v1.json")
process = read(ROOT / "fitting_process.json")
baseline = read("results/verification/rotated_shuffle_fit_baseline_v1.json")
assert baseline["status"] == "PASS" and not baseline["decision_gates_changed"]
assert a["status"] == "PASS" and process["status"] == "PASS"
forms = ("plain", "absorbable", "cross", "narrow", "full_swiglu", "full_gelu")
labels = dict(
    zip(
        forms,
        (
            "Plain BlockShuffle",
            "Absorbable rotation",
            "Cross-origin rotation",
            "Calibrated narrow SwiGLU",
            "Full SwiGLU",
            "Full GELU",
        ),
    )
)
counts = dict(zip(forms, (350208, 350784, 350784, 350208, 1179648, 1179648)))
tasks = list(r["task_mean_mse"])
lines = [
    "# H071 - Full-size rotated-shuffle fitting results",
    "",
    f"**{r['scientific_verdict'].replace('_', ' ')}.** The fixed screen completes all 252 cells",
    "at the actual d384/h2048/G8 structured dimensions. This is standalone synthetic",
    "FFN learning. The language-model gold target remains unmet.",
    "",
    "| Form | Gain over plain | Gain over absorbable | Gain over narrow | Earns full-model resources |",
    "|---|---:|---:|---:|---|",
]
for f in ("absorbable", "cross"):
    ratios = r["ratios"][f]
    earned = r["gates"][f]["earns_full_model_resource_qualification"]
    lines.append(
        f"| {labels[f]} | {100 * (1 - ratios['plain']):+.4f}% | {100 * (1 - ratios['absorbable']):+.4f}% | {100 * (1 - ratios['narrow']):+.4f}% | {'Yes' if earned else 'No'} |"
    )
lines += [
    "",
    "Gains use paired geometric mean held-out MSE ratios across seven targets and",
    "three optimization seeds; positive means lower error. All forms receive the",
    "same two-rate budget. Seed replications share one dataset and teacher set.",
    "",
    "## Decision against the frozen criteria",
    "",
]
for f in ("absorbable", "cross"):
    failed = [k.replace("_", " ") for k, v in r["gates"][f]["tests"].items() if not v]
    lines.append(
        f"- {labels[f]}: "
        + ("fails " + "; ".join(failed) + "." if failed else "passes every fitting criterion.")
    )
lines += [
    "",
    "Cross-origin needs at least 2% aggregate improvement over plain and 1% over",
    "the absorbable control, wins over plain in each seed, an aggregate narrow win,",
    "no generic task regression above 5%, finite states and at least 70% compression.",
    "The absorbable control uses the same criteria without the extra 1% comparison.",
    "",
    "| Form | Seed 17 / plain | Seed 29 / plain | Seed 43 / plain | / full SwiGLU | / full GELU |",
    "|---|---:|---:|---:|---:|---:|",
]
for f in ("absorbable", "cross"):
    vals = [r["by_seed_ratios_to_plain"][f][str(seed)] for seed in (17, 29, 43)]
    lines.append(
        f"| {labels[f]} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {r['ratios'][f]['full_swiglu']:.6f} | {r['ratios'][f]['full_gelu']:.6f} |"
    )
lines += [
    "",
    "Ratios below one are better. The language-model NLL allowance is not applied",
    "to these synthetic MSE values. No convergence or significance claim follows.",
    "",
    "## What was compared",
    "",
    "One standalone FFN uses width 384 and groups 8; all structured intermediate",
    "widths are 384. Every canonical block has six original paths, matching the",
    "full-size structure qualified in H070. This avoids H068's 16 missing blocks",
    "at width 48; it does not reopen H068's rejected activation recipes.",
    "",
    "All three compressed students share their initial block-factor bytes. Rotation",
    "angles start at zero. Three shared-factor teachers are plain or use the two",
    "pairings with each angle vector set to linspace(-0.3,0.3,192). Four additional",
    "targets are smooth, oscillatory, multiplicative and piecewise vector functions.",
    "The teacher tasks favor these families and cannot establish broad superiority.",
    "",
    "There are 4,096 train, 1,024 rate-selection and 1,024 held-out rows. Target",
    "coordinates are divided by training population standard deviation without",
    "centering. The metric is variance-scaled MSE. FP32/TF32-off native eager Torch",
    "uses 300 AdamW updates, batch 256, rates 0.001/0.003, calibrated projection LRs",
    "and angle LR multiplier one. Both rates receive equal work; selection uses",
    "only final selection MSE and is saved before either held-out score.",
    "",
    "See the [frozen plan](rotated_shuffle_fit_plan.md) for complete definitions.",
    "",
    "## Held-out errors by target",
    "",
    "| Target | Plain | Absorbable | Cross-origin | Narrow | Full SwiGLU | Full GELU |",
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
    "Entries are arithmetic means over the three selected seed endpoints. Gate",
    "ratios instead use paired geometric means. All selected rates and individual",
    "endpoints remain in the raw result, including both rates' checkpoints.",
    "",
    "![Task error ratios and measured update time](figures/rotated_shuffle_fit.png)",
    "",
    "## Actual standalone resources and learning signals",
    "",
    "| Form | Parameters | Median update ms | Maximum allocated MiB | Mean clipping | Last 50 / prior 50 loss |",
    "|---|---:|---:|---:|---:|---:|",
]
for f in forms:
    v = a["resources"][f]
    lines.append(
        f"| {labels[f]} | {counts[f]:,} | {v['median_selected_update_ms']:.4f} | {v['max_peak_allocated_mib']:.4f} | {100 * v['mean_clip_fraction']:.2f}% | {v['geomean_last50_loss_over_prior50']:.6f} |"
    )
lines += [
    "",
    "Each cell synchronizes CUDA around updates and discards the first 50 timings.",
    "Peaks reset after warmup. These measurements include the standalone GPU data",
    "and index cache, and do not measure Transformer training or serving. No",
    "full-model memory threshold is inferred from them. Narrow matches plain's",
    "parameter count exactly; it has 576 fewer weights than either rotation.",
    "",
]
for f in ("absorbable", "cross"):
    v = a["resources"][f]
    lines.append(
        f"- {labels[f]}: resetting learned angles changes aggregate held-out error by {100 * (v['reset_angle_error_ratio'] - 1):+.4f}%; maximum absolute angle {v['max_abs_angle_radians']:.4f} radians; mean projection sine RMS {v['mean_projection_sine_rms']:.4f}."
    )
lines += [
    "",
    "Resetting angles after training measures dependence of the final coadapted",
    "model, not the cause of learning gains. The absorbable control adds optimizer",
    "coordinates but no new projection functions. H070's rank witness and component",
    "isometry remain scoped results; neither proves a whole-FFN learning advantage.",
    "",
    "## Independent verification and retained evidence",
    "",
    f"The single scientific fitting process finishes in {process['elapsed_seconds']:.2f} s.",
    "Four isolated harness checks pass before training. All 252 checkpoints have",
    "finite weights and optimizer moments at step 300; all 75,600 updates and",
    "19,353,600 example presentations complete. No corpus targets are scored.",
    "Independent CPU regeneration reproduces the fixed data, target scales and",
    "samplers exactly. The audit reloads every checkpoint, verifies shared initial",
    "weights and rate-selection chronology, recomputes all aggregate gates, and",
    "rescores 126 selected endpoints plus 42 angle-reset ablations on CPU.",
    f"Maximum selected-endpoint CPU/GPU relative MSE disagreement is {a['max_cpu_gpu_relative_mse_error']:.3g}.",
    "",
    "The prototype stays beside its evidence and adds no active model or recipe.",
]
if r["scientific_verdict"] == "REJECTED_AT_THIS_FITTING_BUDGET":
    lines += [
        "Both tested recipes close at this allocation. No larger angle set, extra",
        "stage, rate, teacher amplitude, longer run or language allocation is earned.",
    ]
else:
    lines += [
        "Only the passing form earns a separately frozen full-model resource test.",
        "Language training requires passing that next qualification.",
    ]
lines += [
    "The active 108-test source state is unchanged. Existing language/profile runs",
    "and prior fitting evidence are preserved; the gold target is still unmet.",
    "",
    "- [H070 scoped proof](rotated_shuffle_theory.md) and [local checks](rotated_shuffle_results.md).",
    "- [Raw result](../results/rotated_shuffle_fit_v1/result.json) and [process](../results/rotated_shuffle_fit_v1/fitting_process.json).",
    "- [Independent checkpoint audit](../results/verification/rotated_shuffle_fit_analysis_v1.json).",
    "- [Final preservation audit](../results/verification/rotated_shuffle_fit_final_v1.json).",
    "",
    f"Result SHA-256: `{sha(ROOT / 'result.json')}`.",
    f"Protocol SHA-256: `{sha(ROOT / 'protocol.json')}`.",
    "",
]
Path("research/rotated_shuffle_fit_results.md").write_text("\n".join(lines), encoding="utf-8")

report_path = Path("research/rotated_shuffle_fit_results.md")
report_text = report_path.read_text(encoding="utf-8")
baseline_lines = [
    "## Absolute performance diagnostic",
    "",
    "A post-hoc zero-output predictor exposes a limitation of this learning regime.",
    "On all three teacher targets, oscillatory and multiplicative targets, every",
    "selected model/seed is worse on held-out data than predicting zero. Training",
    "errors are lower, indicating a generalization gap under this fixed setting.",
    "Smooth is learned better than zero by every form; piecewise is better for the",
    "three structured forms and full GELU. This limits interpretation of the tiny",
    "relative differences. It does not change the frozen gates or reopen recipes.",
    "",
    "| Target | Zero held-out MSE | Plain train MSE | Plain held-out / zero |",
    "|---|---:|---:|---:|",
]
for task, row in baseline["rows"].items():
    v = row["forms"]["plain"]
    baseline_lines.append(
        f"| {task.replace('_', ' ')} | {row['heldout_zero_mse']:.6f} | {v['mean_train_mse']:.6f} | {v['heldout_over_zero']:.6f} |"
    )
baseline_lines += [
    "",
    "The zero predictor is a diagnostic, not an added training cell or a predeclared",
    "promotion gate. All target values come from the saved data; no parameters",
    "are fitted for it. [Diagnostic records](../results/verification/rotated_shuffle_fit_baseline_v1.json).",
    "",
    "",
]
report_text = report_text.replace(
    "## Actual standalone resources and learning signals",
    "\n".join(baseline_lines) + "## Actual standalone resources and learning signals",
)
observation_note = """A tool observation of the first CPU-audit call was interrupted after training.
Follow-up OS inspection found no matching live process and no audit result;
execution before interruption is unknown. The completed CPU audit above ran
under a durable coordinator. [Observation record](../results/rotated_shuffle_fit_v1/postprocess_observation.json).
No scientific training cell was repeated.

"""
report_text = report_text.replace(
    "The prototype stays beside its evidence",
    observation_note + "The prototype stays beside its evidence",
)
report_path.write_text(report_text, encoding="utf-8")

plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(
    1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.9, 1]}, constrained_layout=True
)
values = np.array([[a["task_ratios_to_plain"][t][f] for f in forms] for t in tasks])
bound = max(abs(np.log2(values)).max(), 0.01)
im = axes[0].imshow(
    np.log2(values),
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    aspect="auto",
)
axes[0].set_xticks(
    range(6),
    ["Plain", "Absorbable", "Cross-origin", "Narrow", "Full SwiGLU", "Full GELU"],
    rotation=22,
    ha="right",
)
axes[0].set_yticks(range(7), [t.replace("_", " ") for t in tasks])
axes[0].set_title("Paired held-out MSE / plain (lower is better)")
for i in range(7):
    for j in range(6):
        axes[0].text(
            j,
            i,
            f"{values[i, j]:.3f}",
            ha="center",
            va="center",
            color="white" if abs(np.log2(values[i, j])) > 0.6 * bound else "black",
        )
fig.colorbar(im, ax=axes[0], shrink=0.75, label="log2(error ratio)")
times = [a["resources"][f]["median_selected_update_ms"] for f in forms]
axes[1].barh(
    range(6), times, color=["#777777", "#6b9ac4", "#208577", "#a3a3a3", "#cf8b4a", "#cb6872"]
)
axes[1].set_yticks(
    range(6), ["Plain", "Absorbable", "Cross-origin", "Narrow", "Full SwiGLU", "Full GELU"]
)
axes[1].invert_yaxis()
axes[1].set_xlabel("Milliseconds per update")
axes[1].set_title("Measured standalone FP32 training")
for i, value in enumerate(times):
    axes[1].text(value + 0.025, i, f"{value:.2f}", va="center")
axes[1].set_xlim(0, max(times) * 1.18)
fig.suptitle(
    "H071: full-size topology, seven targets, three seeds, equal two-rate budgets", fontsize=13
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/rotated_shuffle_fit.{ext}", dpi=160)
plt.close(fig)
print(json.dumps({"status": "PASS", "report": "research/rotated_shuffle_fit_results.md"}))
