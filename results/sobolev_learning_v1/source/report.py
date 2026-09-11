"""Generate the H109 report from completed, independently audited results."""

import json
import re
import statistics as st
from pathlib import Path

root = Path("results/sobolev_learning_v1")
s = json.loads((root / "summary.json").read_text())
a = json.loads((root / "audit.json").read_text())
inference = json.loads((root / "inference_only.json").read_text())
assert a["passed"] and len(a["scores"]) == 648
groups = {(g["teacher"], g["method"]): g for g in s["groups"]}
names = {
    "value_greedy": "Value only",
    "value_select_sobolev_fit": "Derivative readout",
    "sobolev_greedy": "Derivative selector + readout",
    "random": "Random subset",
}
lines = [
    "# H109: value-only learning erodes the derivative-aware initialization gain\n",
    "**Close both fixed derivative-initialization plus value-learning recipes.**\nH108's useful zero-SGD calibration result survives as its own finding, but its\nadvantage mostly disappears after all narrow-FFN weights learn from new value\ntargets. Neither candidate passes the fixed comparative learning gate, and no\nmethod meets the complete absolute quality gate. The broader goal is unmet.\n",
    f"The screen completes **144 fresh fits, 86,400 updates and 72 selections** in\n{s['elapsed_seconds']:.2f} seconds. All 216 endpoint states, 648 selection/output/derivative\nmetrics and 18 exactly recaptured datasets pass independent verification.\nThe original pre-protocol path-type failure is retained; no training was repeated.\n",
    "![Output fidelity improves while candidate derivative fidelity deteriorates](figures/sobolev_learning.png)\n",
    "## What changed and what was controlled\n",
    "The four initializers are exact H108 primary-width exports: value-greedy/value\nreadout, the same neurons with a derivative-aware readout, derivative-aware\nselection and readout, and random neurons with a value readout. The first is the\nprimary matched control. These are ordinary dense GELU/SwiGLU FFNs with an output\nbias, not new activations. GELU h448 has 344,448 parameters and SwiGLU h298 has\n343,680, respectively **70.80% / 70.87% fewer** than each full FFN.\n",
    "We sample 960 nonoverlapping 128-token windows, excluding every H105/H107\ncollection window, split across sampling seeds71/83/97. The same windows feed\nboth existing seed-17 teachers at layers0/3/7. Each dataset has32,768 training,\n4,096 selection and4,096 reporting rows. Old calibration initializes the models;\nthe fitting/reporting inputs are fresh. They remain from the teachers' training\ncorpus: this is not a new corpus, language holdout or teacher-seed replication.\n",
    "Raw FFN inputs are captured through the BF16 decoder; smooth FP32 teacher\noutputs are recomputed with TF32 off. Each model trains on the same saved batch\nstream, at constant AdamW LR0.001 or0.003 for600 updates, batch256, betas(0.9,0.95),\nepsilon1e-8, zero decay and global gradient clip1. Only output MSE is optimized;\nno derivatives are used after initialization. Both rates start from fresh copies\nof the same weights with new optimizer states. Inputs/targets stay on CPU.\n",
    "The positive H108 calibration RMS sy scales both predictions and targets,\nso training MSE equals raw output MSE/sy². The adapter introduces no trainable\nparameter and leaves raw deployment unchanged. Four qualification groups verify\nthat algebra, parameter gradients, one-step Adam identity, raw outputs, analytic\nJVPs versus autograd and exact counts before real-data access.\n",
    "Selection compares the original initialization with both final rates, using\nselection output MSE only. **All72 selections choose LR0.001 after600 steps.**\nEvery endpoint is retained. Reporting derivatives use two new independent\nRademacher directions per input, scaled by old calibration input SDs. They are\nnever used for optimization or endpoint selection. Full FFNs receive30 LR0\nprofile updates per dataset (540 total), allocating Adam/gradient memory while\nverified weights remain unchanged. These are resource controls, not fresh teachers.\n",
    "## Initial and learned errors\n",
    "| Teacher | Method | Initial output MSE | Learned output MSE | Initial derivative error | Learned derivative error |\n|---|---|---:|---:|---:|---:|",
]
for g in s["groups"]:
    lines.append(
        f"| {g['teacher']} | {names[g['method']]} | {g['initial']['reporting_mse']['mean']:.6f} | {g['reporting_mse']['mean']:.6f} | {g['initial']['derivative_relative_mse']['mean']:.6f} | {g['derivative_relative_mse']['mean']:.6f} |"
    )
lines += [
    "\nOutput MSE is divided by calibration output variance; derivative squared error\nis divided by reporting teacher JVP energy. Table entries are arithmetic means\nover nine depth-by-sampling cells. Paired decisions use geometric-mean ratios.\n",
    "| Candidate / teacher | Initial derivative gain vs value only | Learned derivative gain | Learned output gain | Fixed component gate |\n|---|---:|---:|---:|---|",
]
for method, decision in s["components"].items():
    for teacher, d in decision["teachers"].items():
        ratios = d["paired_ratios"]
        lines.append(
            f"| {names[method]} / {teacher} | {100 * (1 - ratios['0']['derivative_relative_mse']):.2f}% | {100 * (1 - ratios['selected']['derivative_relative_mse']):.2f}% | {100 * (1 - ratios['selected']['reporting_mse']):.3f}% | Fail |"
        )
lines += [
    "\nBoth candidates miss the fixed5% derivative improvement and every-seed rule\non both teachers. Value error stays within the1% allowance; updates are within\n10% of value-only and all quantities remain finite. Matching rates does not\nrestore the gain: the selected comparison is exactly the equal-LR0.001 comparison,\nand LR0.003 derivative changes are also smaller than1%. No lucky-rate explanation\nrescues the initializer. Neither control/candidate earns language insertion.\n",
    "| Candidate / teacher | Output change from its own initialization | Derivative-error change from its own initialization |\n|---|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    for method in ("value_select_sobolev_fit", "sobolev_greedy"):
        r = groups[teacher, method]["selected_over_initial_ratios"]
        lines.append(
            f"| {names[method]} / {teacher} | {100 * (r['reporting_mse'] - 1):+.2f}% | {100 * (r['derivative_relative_mse'] - 1):+.2f}% |"
        )
lines += [
    "\nThese are paired geometric-mean changes. Lower value error does not imply\nbetter derivatives. A simple counterexample is f_e(x)=e sin(x/e²): its uniform\nvalue magnitude is at most e, while its maximum derivative magnitude is1/e.\nThat elementary example proves a general non-implication, not a theorem about\nthis fixed FFN class or the cause of its training trajectory. The experiment\nmeasures the divergence directly; it does not prove vanishing/exploding network\ngradients. Ambient probes also differ from actual upstream perturbations through\nRMSNorm and the surrounding decoder; no downstream NLL conclusion follows.\n",
    "## Sampling variation and gradients\n",
    "| Candidate / teacher | Seed71 output / derivative | Seed83 output / derivative | Seed97 output / derivative | Output median; variance | Derivative median; variance |\n|---|---:|---:|---:|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    for method in ("value_select_sobolev_fit", "sobolev_greedy"):
        g = groups[teacher, method]
        seeds = [
            f"{g['seed_means'][str(seed)]['reporting_mse']:.5f} / {g['seed_means'][str(seed)]['derivative_relative_mse']:.5f}"
            for seed in (71, 83, 97)
        ]
        lines.append(
            f"| {names[method]} / {teacher} | {' | '.join(seeds)} | {g['reporting_mse']['median']:.5f}; {g['reporting_mse']['sample_variance']:.5g} | {g['derivative_relative_mse']['median']:.5f}; {g['derivative_relative_mse']['sample_variance']:.5g} |"
        )
lines += [
    "\nVariances span all nine cells and include depth differences; they are not\nconfidence intervals from independent teacher training. Full summaries retain\nmean, median, variance, range and every seed/depth for all four methods.\nAll144 fits have finite parameters, Adam moments, losses and gradients; **no\nupdate triggers clipping**. Preclip norms range approximately0.025-0.567 across\nthe grid. This excludes gross numerical blowup here, but says little about\nlong-depth conditioning or convergence. Per-parameter gradients and activation\nstatistics at0/50/150/600 are retained in the compressed diagnostic record.\n",
    "## Actual memory and runtime\n",
    "| Teacher | Local fitting peak MiB | Full fitting peak MiB | Training saving | Inference peak after training MiB | Full inference MiB |\n|---|---:|---:|---:|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    g, f = groups[teacher, "value_select_sobolev_fit"], s["full_references"][teacher]
    lines.append(
        f"| {teacher} | {g['peak_allocated_bytes']['mean'] / 2**20:.3f} | {f['training_peak_bytes'] / 2**20:.3f} | {100 * (1 - g['peak_allocated_bytes']['mean'] / f['training_peak_bytes']):.2f}% | {g['inference']['peak_allocated_bytes']['mean'] / 2**20:.3f} | {f['inference_peak_bytes'] / 2**20:.3f} |"
    )
lines += [
    "\nThe ordinary value-only control has the same fitting and inference allocation\nas the derivative candidates. Mean median update time is approximately2.65ms\nfor GELU and2.94ms for SwiGLU. These gains are width savings shared with the\ncontrols; they do not establish an efficiency benefit from derivative fitting.\n",
    "The original inference profiles run in the process that has already executed\nbackward passes. A separate, explicitly posthoc fresh process with no gradients\nor optimizer gives the following deployment-only values. This does not replace\nthe frozen gate inputs, and the allocator/context difference is not assigned\na proved low-level root cause. All90 fresh inference profiles use the same\nselected weights and batch256,10 warmups and5x20 synchronized calls.\n",
    "| Teacher | Fresh inference candidate / full MiB | Memory saving | Candidate / full median-block mean ms |\n|---|---:|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    students = [
        r
        for r in inference["students"]
        if r["teacher"] == teacher and r["method"] == "value_select_sobolev_fit"
    ]
    full = [r for r in inference["full_teachers"] if r["teacher"] == teacher]
    memory, full_memory = [
        st.mean(r["peak_allocated_bytes"] for r in records) for records in (students, full)
    ]
    timing, full_time = [
        st.mean(r["inference_ms"] for r in records) for records in (students, full)
    ]
    lines.append(
        f"| {teacher} | {memory / 2**20:.3f} / {full_memory / 2**20:.3f} | {100 * (1 - memory / full_memory):.2f}% | {timing:.4f} / {full_time:.4f} |"
    )
lines += [
    f"\nThe complete preparation/fitting pipeline peaks at **{s['pipeline_peak_cuda_bytes'] / 2**20:.3f}MiB**,\nincluding prerequisite teacher capture/calibration and fresh capture/targets.\nFresh capture costs{s['fresh_capture_seconds']:.2f}s and target generation{s['fresh_target_seconds']:.2f}s.\nThe reused H108 statistics plus all selectors cost{s['prior_statistics_and_all_selectors_seconds']:.2f}s;\nper-method readout times and prerequisite capture are retained in the original\nrecords. Shared all-selector preparation is a conservative accounting quantity,\nnot the minimum cost of value-only calibration. Original teacher pretraining is\nnot included. Peak allocated tensors and reserved memory are distinct from whole\nprocess/driver VRAM; CPU data/statistics also consume host RAM.\n",
    "## Decision, provenance and limits\n",
    "Both teachers fail the absolute mean value0.05 and derivative0.10 gates,\nincluding the seed aggregates. Local training/inference memory, timing and\nparameter gates pass, but cannot compensate for lost quality. There is no\nautomatic language experiment, larger rate/step budget or changed derivative\nweight. Close this fixed initializer-plus-value-learning branch.\n",
    "This is a useful negative learning result: initialization can encode local\nsensitivity information that output-only optimization subsequently discards.\nH108 remains a qualified zero-SGD component; it does not become a general\ntraining solution. Any future derivative-constrained learning or actual\nupstream-sensitivity test would be a distinct hypothesis with its own cost and\nqualification, not an earned continuation of this recipe.\n",
    "The first launch stopped before protocol creation, qualification, data capture\nor updates because string paths were supplied to a Path-only hash helper. Its\nsource, log, exception and exit status remain under preflight_failure and the\nrecovery manifest. Corrected call sites were then frozen before the first\nscientific execution. All144 actual fits finish on that execution without retry.\n",
    f"The independent auditor exactly recaptures all18 datasets and smooth value\ntargets, checks all216 endpoint states against hashes, all72 exact H108\ninitializations, shared streams,144 training records/initial losses and648\nmetrics. Student values/JVPs are recomputed from raw tensors at batch257 with\nautograd derivatives. Maximum score discrepancy is{a['max_score_difference']:.3g}.\nThe audit performs zero updates and passes on its first execution.\n",
    "The maintained model tree and tests are unchanged from the preceding116-test\npass; four isolated qualification groups and this audit provide new evidence.\nNo novelty priority, new activation, SOTA superiority, long-run convergence,\nwhole-model memory, broad task transfer or full goal completion is established.\n",
    "## Evidence\n",
    "- [Frozen plan](sobolev_learning_plan.md) and [source guide](../results/sobolev_learning_v1/source/README.md).\n- [Complete endpoints](../results/sobolev_learning_v1/result.json.gz), [CSV](../results/sobolev_learning_v1/metrics.csv.gz), [summary](../results/sobolev_learning_v1/summary.json), [diagnostics](../results/sobolev_learning_v1/diagnostics.json.gz).\n- [Independent audit](../results/sobolev_learning_v1/audit.json.gz), [fresh-process inference](../results/sobolev_learning_v1/inference_only.json), [final receipt](../results/verification/sobolev_learning_final_v1.json).\n- [Prior Sobolev training](https://arxiv.org/abs/1706.04859) and [GRAIL](https://proceedings.mlr.press/v328/tang26a.html); this study does not claim their principles as new.\n",
    "Raw data, model tensors, per-step histories and source ZIP remain local and\nignored. Compact metadata and hashes are not a tensor backup.\n",
]
# Keep prose readable when numbers directly follow words in template fragments.
text = "\n".join(lines)
text = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", text)
text = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", text)
# Avoid altering paths, identifiers or URLs with numeric components.
# Restore those exact artifact identifiers after prose spacing.
for token in (
    "H108",
    "H109",
    "H105",
    "H107",
    "FP32",
    "BF16",
    "TF32",
    "AdamW",
    "RMSNorm",
    "JVP",
    "WikiText-2",
    "v1",
    "v328",
    "tang26a",
    "1706.04859",
):
    spaced = re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", token)
    text = text.replace(spaced, token)
text = re.sub(r"(\d) e(-\d)", r"\1e\2", text)
Path("research/sobolev_learning_results.md").write_text(text + "\n", encoding="utf-8")
print("Wrote research/sobolev_learning_results.md")
