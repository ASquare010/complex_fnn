"""Render H073 from independently audited fitting, including assay validity."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path("results/ungated_fit_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


r = read(ROOT / "result.json")
a = read("results/verification/ungated_fit_analysis_v1.json")
process = read(ROOT / "fitting_process.json")
subsets = read("results/verification/ungated_fit_subsets_v1.json")
assert subsets["status"] == "PASS" and not subsets["decision_gates_changed"]
assert a["status"] == "PASS" and process["status"] == "PASS"
forms = (
    "plain",
    "gelu_same",
    "gelu_matched",
    "narrow_swiglu",
    "narrow_gelu",
    "full_swiglu",
    "full_gelu",
)
labels = dict(
    zip(
        forms,
        (
            "Plain BlockShuffle SwiGLU",
            "Same-width BlockShuffle GELU",
            "Matched-weight BlockShuffle GELU",
            "Narrow SwiGLU",
            "Narrow GELU",
            "Full SwiGLU",
            "Full GELU",
        ),
    )
)
counts = dict(zip(forms, (350208, 233472, 350208, 350208, 350208, 1179648, 1179648)))
tasks = list(r["task_mean_mse"])
lines = [
    "# H073 - Ungated fitting with positive controls",
    "",
    f"**{r['scientific_verdict'].replace('_', ' ')}.** All 294 fixed cells complete at width384,",
    "with seven forms, seven targets, three seeds and equal two-rate budgets.",
    f"The frozen positive-control assay {'passes' if a['assay_passed'] else 'fails'}; the gold language-model goal remains unmet.",
    "",
    "| Ungated form | Gain over plain | Gain over narrow GELU | Gain over narrow SwiGLU | Gain over same-width GELU | Earns full-model resource test |",
    "|---|---:|---:|---:|---:|---|",
]
for f in ("gelu_same", "gelu_matched"):
    ratios = r["ratios"][f]
    earned = r["gates"][f]["earns_full_model_resource_qualification"]
    lines.append(
        f"| {labels[f]} | {100 * (1 - ratios['plain']):+.3f}% | {100 * (1 - ratios['narrow_gelu']):+.3f}% | {100 * (1 - ratios['narrow_swiglu']):+.3f}% | {100 * (1 - ratios['gelu_same']):+.3f}% | {'Yes' if earned else 'No'} |"
    )
lines += [
    "",
    "Positive gains mean lower held-out variance-scaled MSE. Ratios are paired",
    "geometric means across 21 equally weighted task/seed endpoints. Rates are",
    "selected on a separate split before either rate's report score is computed.",
    "",
    "## Assay validity and absolute performance",
    "",
    "The positive control must learn, rather than merely participate in the table.",
    "Full GELU must reduce linear held-out error by at least 50% versus zero in",
    "every seed, and improve at least two generic tasks by 2% versus zero.",
    "",
    "| Positive-control criterion | Result |",
    "|---|---|",
]
for key, value in r["positive_control_tests"].items():
    lines.append(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |")
lines += ["", "| Form | Linear / zero, seed17 | Seed29 | Seed43 |", "|---|---:|---:|---:|"]
for f in ("full_gelu", "gelu_same", "gelu_matched"):
    values = [
        next(
            row["heldout_mse"] / row["zero_mse"]
            for row in r["selected_rows"]
            if (row["task"], row["form"], row["seed"]) == ("linear", f, s)
        )
        for s in (17, 29, 43)
    ]
    lines.append(f"| {labels[f]} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
lines += [
    "",
    "| Task | Plain / zero | Same-width GELU / zero | Matched GELU / zero | Narrow SwiGLU / zero | Narrow GELU / zero | Full SwiGLU / zero | Full GELU / zero |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for t in tasks:
    lines.append(
        "| "
        + t.replace("_", " ")
        + " | "
        + " | ".join(f"{a['ratios_to_zero'][t][f]:.6f}" for f in forms)
        + " |"
    )
lines += [
    "",
    "Ratios below one beat the zero predictor. A failed assay earns no model",
    "allocation even if a candidate looks better than another weak fit. Candidate",
    "gates also cap each generic target's ratio at 1.05 versus both plain and zero.",
    "",
    "## Frozen candidate decision",
    "",
]
for f in ("gelu_same", "gelu_matched"):
    failed = [key.replace("_", " ") for key, value in r["gates"][f]["tests"].items() if not value]
    lines.append(
        f"- {labels[f]}: "
        + ("fails " + "; ".join(failed) + "." if failed else "passes every fitting gate.")
    )
lines += [
    "",
    "Both need a passing assay, at least 2% aggregate gain over plain, a gain in",
    "each seed, wins over both calibrated narrows, generic regression caps, a",
    "50% linear improvement versus zero in each seed, finite states and at least",
    "70% FFN reduction. Matched GELU also needs 1% over same-width GELU to justify",
    "its extra weights. These are fitting gates; no language NLL allowance is",
    "transplanted onto MSE. Three optimization seeds do not establish significance.",
    "",
    "| Form | Seed17 / plain | Seed29 / plain | Seed43 / plain | / full GELU | / full SwiGLU |",
    "|---|---:|---:|---:|---:|---:|",
]
for f in ("gelu_same", "gelu_matched"):
    vals = [r["by_seed_ratios_to_plain"][f][str(s)] for s in (17, 29, 43)]
    lines.append(
        f"| {labels[f]} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {r['ratios'][f]['full_gelu']:.6f} | {r['ratios'][f]['full_swiglu']:.6f} |"
    )
lines += [
    "",
    "## Matched comparison and interpretation",
    "",
    "The existing ungated operator uses h2048 (233,472 weights) or h3264",
    "(350,208 weights). Plain SwiGLU h2048 and both narrows match 350,208 exactly.",
    "Full GELU h1536 and full SwiGLU h1024 each have 1,179,648 FFN weights.",
    "Matched GELU therefore retains 70.3125% FFN reduction; same-width retains",
    "80.2083%. No new activation or active model is introduced.",
    "",
    "Fixed uniform-input data has 65,536 train, 4,096 selection and 4,096 report",
    "rows. Each cell samples 76,800 training examples in 300 updates. Increasing",
    "the finite training pool was motivated by H071's generalization gap; this",
    "new task/data cohort does not isolate a causal effect of data quantity.",
    "The three seeds are optimization replications of the same dataset.",
    "",
    "Targets are a rotated linear map, fixed plain/ungated structured teachers,",
    "and smooth, oscillatory, multiplicative and piecewise vector functions.",
    "Teacher tasks favor their families. Train-only population standard deviations",
    "scale each output coordinate without centering. This is variance-scaled MSE.",
    "Both rates use 0.001/0.003 with identical update counts, AdamW settings and",
    "calibrated fan-in multipliers. Plain and same-width GELU share up/down initial",
    "factor bytes; activation changes. Matched GELU changes width and factor shapes.",
    "",
    "[H072's proof](ungated_blockshuffle_theory.md) still restricts a single bias-free",
    "GELU layer to a linear odd component. Its multiplicative-target population",
    "relative floor is 12/17; this is not an exact bound on this finite report set.",
    "Fitting success would not establish a universal expressivity advantage or",
    "remove that restriction. Read the [frozen plan](ungated_fit_plan.md) for details.",
    "",
    "![Error relative to plain and to the zero predictor](figures/ungated_fit.png)",
    "",
    "## Actual standalone resources and learning trends",
    "",
    "| Form | Parameters | Median update ms | Maximum allocated MiB | Mean clipping | Last50 / prior50 loss | Selected rates .001 / .003 |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for f in forms:
    v = a["resources"][f]
    lines.append(
        f"| {labels[f]} | {counts[f]:,} | {v['median_selected_update_ms']:.4f} | {v['max_peak_allocated_mib']:.3f} | {100 * v['mean_clip_fraction']:.3f}% | {v['geomean_last50_loss_over_prior50']:.6f} | {v['selected_rate_counts']['0.001']} / {v['selected_rate_counts']['0.003']} |"
    )
lines += [
    "",
    "Update timings synchronize CUDA and exclude the first 50 warmup updates.",
    "Peak counters reset after warmup. The GPU data/index cache is included; these",
    "standalone FP32 measurements do not qualify Transformer memory, serving speed",
    "or a full-model optimization trajectory. Loss trends are not convergence.",
    "",
    "## Independent verification and next step",
    "",
    f"The scientific fitting process finishes first attempt in {process['elapsed_seconds']:.2f} s.",
    "All 294 cells complete: 88,200 optimizer updates and 22,579,200 example",
    "presentations. Five isolated harness checks pass before fitting. No corpus",
    "targets are scored and no full-model resource worker is dispatched.",
    "Independent CPU regeneration reproduces inputs, all targets/scales and",
    "streams exactly. Every checkpoint has finite weights/moments at step300.",
    "All shared initializations, rate selections, held-out chronology and gates",
    "are independently checked; 147 selected endpoints are rescored on CPU.",
    f"Maximum relative CPU/GPU MSE disagreement is {a['max_cpu_gpu_relative_mse_error']:.3g}.",
    "",
]
if not a["assay_passed"]:
    lines += [
        "The fixed positive-control assay fails. No candidate is promoted and this",
        "screen is inconclusive for model allocation; no automatic budget/data/rate",
        "expansion is earned. All outcomes remain recorded.",
    ]
elif any(g["earns_full_model_resource_qualification"] for g in r["gates"].values()):
    lines += [
        "Only a passing form earns a separately frozen actual full-model resource",
        "qualification. This result does not authorize language training or add an",
        "active model. Quality, convergence, broader data and novelty remain open.",
    ]
else:
    lines += [
        "The assay passes but both tested ungated recipes fail promotion at this",
        "budget. No automatic width/rate/budget expansion or activation repair follows.",
    ]
lines += [
    "The active three model folders and six variants remain unchanged. The prior",
    "171 language/profile runs and all older fitting evidence are preserved.",
    "",
    "- [Raw result](../results/ungated_fit_v1/result.json) and [process](../results/ungated_fit_v1/fitting_process.json).",
    "- [Independent checkpoint audit](../results/verification/ungated_fit_analysis_v1.json).",
    "- [Final preservation audit](../results/verification/ungated_fit_final_v1.json).",
    "",
    f"Result SHA-256: `{sha(ROOT / 'result.json')}`.",
    f"Protocol SHA-256: `{sha(ROOT / 'protocol.json')}`.",
    "",
]
Path("research/ungated_fit_results.md").write_text("\n".join(lines), encoding="utf-8")
report_path = Path("research/ungated_fit_results.md")
report_text = report_path.read_text(encoding="utf-8")
subset_lines = [
    "## Descriptive checks on the headline gain",
    "",
    "The simple linear positive-control task contributes heavily to the aggregate",
    "gain. These post-hoc subsets show the remaining effect; they change no frozen",
    "gate, selected rate, training allocation or result. Teacher tasks are excluded",
    "from the generic-only subset. This is not a new promotion criterion.",
    "",
    "| Form | Subset | Gain over plain | Gain over narrow GELU | Gain over full GELU | Gain over full SwiGLU |",
    "|---|---|---:|---:|---:|---:|",
]
for subset in ("without_linear", "generic_only"):
    for form in ("gelu_same", "gelu_matched"):
        ratios = subsets["ratios"][subset][form]
        subset_lines.append(
            f"| {labels[form]} | {subset.replace('_', ' ')} | {100 * (1 - ratios['plain']):+.3f}% | {100 * (1 - ratios['narrow_gelu']):+.3f}% | {100 * (1 - ratios['full_gelu']):+.3f}% | {100 * (1 - ratios['full_swiglu']):+.3f}% |"
        )
subset_lines += [
    "",
    "In particular, an overall win over full GELU must not be read as a comparably",
    "large advantage on the generic nonlinear targets. Oscillatory errors remain",
    "around the zero predictor for all forms. These limits remain part of the",
    "decision to test resources, rather than a claim of complex-pattern mastery.",
    "[Subset records](../results/verification/ungated_fit_subsets_v1.json).",
    "",
    "",
]
report_text = report_text.replace(
    "## Matched comparison and interpretation",
    "\n".join(subset_lines) + "## Matched comparison and interpretation",
)
report_path.write_text(report_text, encoding="utf-8")

plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
short = ["Plain", "GELU same", "GELU matched", "Narrow SG", "Narrow GELU", "Full SG", "Full GELU"]
for ax, source, title in zip(
    axes,
    (a["task_ratios_to_plain"], a["ratios_to_zero"]),
    ("Held-out MSE / plain", "Held-out MSE / zero predictor"),
):
    values = np.array([[source[t][f] for f in forms] for t in tasks])
    vmax = max(2.0, float(values.max()))
    im = ax.imshow(
        values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax), aspect="auto"
    )
    ax.set_xticks(range(7), short, rotation=30, ha="right")
    ax.set_yticks(range(7), [t.replace("_", " ") for t in tasks])
    ax.set_title(title + " (lower is better)")
    for i in range(7):
        for j in range(7):
            v = values[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                color="white" if v < 0.25 or v > 1 + 0.65 * (vmax - 1) else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Paired error ratio")
fig.suptitle(
    "H073: full-size ungated comparison; 65,536 train rows; fixed positive-control requirements",
    fontsize=13,
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_fit.{ext}", dpi=160)
plt.close(fig)
print(json.dumps({"status": "PASS", "report": "research/ungated_fit_results.md"}))
