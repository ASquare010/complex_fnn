"""Render the H108 findings from audited endpoint statistics."""

import json
from pathlib import Path

root = Path("results/sobolev_selection_v1")
summary, audit, ablation = (
    json.loads((root / name).read_text())
    for name in ("summary.json", "audit.json", "ablation.json")
)
assert audit["passed"]
groups = {(r["teacher"], r["budget"], r["method"]): r for r in summary["groups"]}
budgets = ("primary", "half", "three_quarter")
lines = [
    "# H108: derivative-aware calibration is useful; complete compression remains unqualified\n",
    "**Retain derivative-aware calibration as a promising component.** It improves\nboth output reconstruction and directional derivatives for both teacher families\nat every tested width. The effect survives all three sampling-seed aggregates.\nThe three complete recipes still fail the absolute quality/memory allocation\ngates, so none earns language insertion. The original architectural and broader\nVRAM/quality goals remain unmet.\n",
    f"The screen produces **432 compressed FFNs with zero SGD updates** in\n{summary['elapsed_seconds']:.2f} seconds. Independent auditing verifies all **864\noutput/derivative scores**, all 432 exports/readouts, reconstructed sufficient\nstatistics for 18 datasets and 432 sampled real greedy marginal gains. The\npredeclared ablations show that the readout objective alone captures about\n88-90% of the combined reduction in derivative error.\n",
    "![Output error, derivative error, ablations and memory](figures/sobolev_selection.png)\n",
    "## Mechanism and scope of the proof\n",
    "The deployed model retains selected rows of a teacher's up/gate matrices,\nfits its down projection and adds an output bias. It is an ordinary dense GELU\nor SwiGLU FFN, with no derivative operation or selection metadata in inference.\nAll deployed weights remain trainable; this experiment only fits the readout by\nlinear algebra. It does not test learning a new activation or training from scratch.\n",
    "Let H be centered/standardized hidden features and Y centered outputs in\ntraining-variance units. Let K and T be directional derivatives in the same\ncoordinates, without centering. The subset readout minimizes\n\n\\[L_S=\\|H_SB-Y\\|_F^2/N+\\beta\\|K_SB-T\\|_F^2/N+10^{-4}\\|B\\|_F^2.\\]\n\nBeta is zero for value fitting or 0.1 divided by the calibration mean square\nof T for derivative-aware fitting. This assigns the derivative target 10% of\nthe value target's energy. Intercept compensation is unpenalized, and every\nnormalizer is fitted on calibration rows only.\n",
    "For the regularized Gram G and cross-target matrix C, conditioning on selected\nfeatures S leaves scalar variance g_j and cross-target vector c_j for another\nfeature j. Completing the square shows that adding j decreases the optimal\nobjective by exactly ||c_j||²/g_j. The [frozen plan](sobolev_selection_plan.md)\ngives the full Schur-complement equations. Positive ridge makes the Gram\npositive definite. Greedy maximal one-step gain is not globally optimal subset\nselection and does not guarantee generalization.\n",
    "Independent Rademacher directions satisfy E[vv']=I, hence the expected\nsquared response of J diag(sx) v is ||J diag(sx)||_F². The input scales sx\ncome from calibration SDs. This motivates a sampled sensitivity metric, not a\nworst-case Jacobian bound or proof against deep-network gradient collapse.\nThirteen qualification groups compare analytic JVPs to autograd and finite\ndifferences, validate exports/counts, enumerate isotropy exactly on a fixture,\nand check every small-fixture greedy gain against all alternative least-squares\nfits, including rank-deficient features.\n",
    "## Data and controlled comparison\n",
    "The 18 H107 input datasets are reused: two existing seed-17 teachers, depths\n0/3/7 and sampling seeds 71/83/97. Only the first 8,192 calibration inputs and\nlast 4,096 reporting inputs are used; the former selection partition is unused.\nThese are previously inspected development inputs from the teachers' training\ncorpus, not a fresh confirmation or independent teacher-training seeds.\n",
    "H108 recomputes smooth FP32 teacher values and derivatives, with TF32 disabled.\nIt does not differentiate rounded BF16 arithmetic or reuse H107's BF16 labels.\nConsequently its MSE is not directly comparable to H107's different targets,\ncalibration size and fitting budget. All eight H108 methods share the same data,\nnormalizers, fixed ridge and parameter budget. One direction per calibration\nrow and two independent directions per reporting row are used. Full input\nJacobians are never materialized.\n",
    "Controls are random subsets, output-column weight norms, fluctuation scores,\nconditional standardized-feature variance and value-greedy selection. The\ncandidate uses Sobolev-greedy selection and readout. Two cross-combinations\nisolate the selector and readout objective. No beta/ridge sweep or SGD is used.\n",
    "## Numerical results\n",
    "| Teacher | FFN parameter reduction | Exact parameters | Value-only MSE | Derivative-aware MSE | Value-only derivative error | Derivative-aware derivative error |\n|---|---|---:|---:|---:|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    for budget in budgets:
        base, candidate = (
            groups[teacher, budget, "value_greedy"],
            groups[teacher, budget, "sobolev_greedy"],
        )
        lines.append(
            f"| {teacher} | {candidate['parameter_reduction']:.2%} | {candidate['parameters']:,} | {base['value_nmse']['mean']:.6f} | {candidate['value_nmse']['mean']:.6f} | {base['derivative_relative_mse']['mean']:.6f} | {candidate['derivative_relative_mse']['mean']:.6f} |"
        )
lines += [
    "\nValue MSE is normalized by calibration output variance. Derivative error is\nnormalized by the reporting teacher's mean squared directional derivative.\nMeans cover nine depth-by-sampling cells. The comparison gate uses paired\ngeometric-mean ratios, not a ratio of arithmetic means.\n",
    "| Width budget | Value improvement: GELU / SwiGLU | Derivative improvement: GELU / SwiGLU | Comparative component gate |\n|---|---:|---:|---|",
]
for budget in budgets:
    ratios = [
        summary["component_decisions"][budget]["teachers"][t]["ratios_to_value_greedy"]
        for t in ("gelu", "swiglu")
    ]
    values = " / ".join(f"{1 - r['value_nmse']:.2%}" for r in ratios)
    gradients = " / ".join(f"{1 - r['derivative_relative_mse']:.2%}" for r in ratios)
    lines.append(f"| {budget} | {values} | {gradients} | Pass |")
lines += [
    "\nAll six teacher/width groups meet the fixed 5% derivative gain and at-most-1%\nvalue-regression rule, including each seed's aggregate. In fact value error\nimproves in each seed aggregate. No cheaper static selector dominates the\ncandidate in both errors. This is a replicated local component result.\n",
    "| Teacher / budget | Seed 71 value / derivative | Seed 83 value / derivative | Seed 97 value / derivative | Value median; sample variance | Derivative median; sample variance |\n|---|---:|---:|---:|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    for budget in budgets:
        r = groups[teacher, budget, "sobolev_greedy"]
        seeds = " | ".join(
            f"{r['seed_means'][str(s)]['value_nmse']:.5f} / {r['seed_means'][str(s)]['derivative_relative_mse']:.5f}"
            for s in (71, 83, 97)
        )
        lines.append(
            f"| {teacher} / {budget} | {seeds} | {r['value_nmse']['median']:.5f}; {r['value_nmse']['sample_variance']:.5g} | {r['derivative_relative_mse']['median']:.5f}; {r['derivative_relative_mse']['sample_variance']:.5g} |"
        )
lines += [
    "\nNine-cell variances include depth heterogeneity and are not confidence intervals\nfrom independent teacher training. The summary retains mean, median, sample\nvariance, range, each depth and each seed for every method. The CSV retains all\n432 endpoints and resources, including failed controls.\n",
    "## Which part caused the gain?\n",
    "Keeping the value-greedy neuron set and changing only its readout objective\nrecovers 88.33-90.14% of the combined absolute derivative-error improvement\nacross the six groups. The share is descriptive: selector/readout interactions\nmean the separate fractions need not sum to one. The derivative-aware selector\nitself adds a smaller increment. At the primary GELU budget, readout-only even\nhas slightly better value MSE than the combined method.\n",
    "This supports retaining the simpler derivative-aware readout as the main\ncomponent. It does not prove that the elaborate selector is necessary. The\nmechanism is consistent with using additional derivative information to constrain\nthe readout, but neither these ablations nor a derivative error establishes the\ncause of downstream language quality. No downstream language test was performed.\n",
    "## Actual inference memory and the failed absolute gate\n",
    "| Teacher / budget | Inference peak MiB | Saving vs full | Inference ms | Full-reference mean ms |\n|---|---:|---:|---:|---:|",
]
for teacher in ("gelu", "swiglu"):
    full = ablation["full_reference"][teacher]
    for budget in budgets:
        r = groups[teacher, budget, "sobolev_greedy"]
        peak = r["peak_allocated_bytes"]["mean"]
        lines.append(
            f"| {teacher} / {budget} | {peak / 2**20:.3f} | {1 - peak / full['peak_allocated_bytes']['mean']:.2%} | {r['inference_ms']['mean']:.4f} | {full['inference_ms']['mean']:.4f} |"
        )
lines += [
    "\nThe primary models meet the original 70% parameter-count threshold and save\nroughly one third of local forward tensor allocation, but their reconstruction\nand derivative errors are far above the fixed absolute quality gates. Half-width\nalso fails quality. Three-quarter width brings both teachers below 5% output\nerror, but SwiGLU derivative error remains 11.71%, above the 10% gate, and both\nfamilies save only about 6.6-6.8% inference allocation, below the required 10%.\nNeither candidate nor eligible value-greedy control earns language insertion.\n",
    "Counts include an output bias: 2dh+d for GELU and 3dh+d for SwiGLU. Nominal\nhalf/quarter reductions are slightly less than 50%/25% after counting that bias.\nCompared methods have identical dense shapes at each teacher/width; derivative\ncalibration adds no inference parameters, gathers, JVPs or kernels. Dense matrix\nMACs per token are 2dh/3dh; scalar activation/bias costs remain measured in timing.\n",
    f"The screen's combined calibration preprocessing peaks at\n{summary['preprocessing_peak_cuda_bytes'] / 2**20:.3f} MiB allocated. Including the\nprerequisite full-teacher capture gives a pipeline peak of\n{summary['pipeline_peak_cuda_bytes'] / 2**20:.3f} MiB. This capture was reused, not\nrerun. Shared value-and-derivative statistics cost {summary['preprocessing_seconds']:.2f}\nseconds; all selectors cost {summary['all_selection_seconds']:.2f} seconds; all\nreadout solves cost {summary['all_readout_seconds']:.2f} seconds. A value-only\nselector does not need the derivative preprocessing, so this shared screen cost\nis not its minimum deployment preparation cost.\n",
    "Memory is peak PyTorch tensor allocation, with reserved values retained\nseparately, not total process/driver VRAM. Inference uses batch 256, ten warmups\nand five synchronized timing blocks of twenty calls, under the recorded\nPython allocator/environment. Absolute timings vary across cases on this laptop;\nall raw blocks are retained. These are native eager measurements, not compiled\nserving throughput. No optimizer is allocated and no training-memory saving,\nconvergence or long-depth gradient guarantee is claimed.\n",
    "## Independent verification and preserved failure\n",
    f"The auditor recomputes calibration statistics with centered full matrices and\nautomatic JVPs, independent of the screen's streamed analytic formulas. It\nchecks every exported row/readout and reconstructs 864 output/derivative scores\nwith a different student batch partition. The maximum score discrepancy is\n{audit['max_score_difference']:.3g}. Eight actual prefixes of each of three greedy\norders are checked per dataset against direct conditioned solves: 432 marginal\nchecks, maximum gain discrepancy {audit['max_gain_difference']:.3g}. These real\nchecks sample prefixes; the small qualification fixtures check every step and\nevery competing addition.\n",
    "The first audit failed on one near-zero teacher output: changing GEMM batch\n512 to 257 produced a 2.086e-6 difference against an absolute tolerance of\n2e-6. The failure, original source and logs remain intact. A separately recorded\nrecovery checks teacher references at their original batch 512 while keeping\nall student rescoring at 257, automatic derivatives and every tolerance unchanged.\nThat complete audit passes. No original compressed model, selected subset,\nfrozen source or recorded metric was replaced.\n",
    "The maintained architecture remains two model folders and five variants.\nNo maintained source or test changed in this round; the preceding 116-test result\nremains referenced separately. The 13 new isolated qualification groups and\nthis independent audit provide the new validation rather than inflating the\nmaintained suite's count.\n",
    "## Prior work, decision and next evidence\n",
    "[Orthogonal least squares](https://doi.org/10.1109/72.80341) supplies the\nfeature-selection precedent; [Sobolev training](https://arxiv.org/abs/1706.04859)\nexplicitly learns from target derivatives. [GRAIL](https://proceedings.mlr.press/v328/tang26a.html)\nprecedes Gram/ridge post-compression reconstruction and\n[FLAP](https://ojs.aaai.org/index.php/AAAI/article/view/28960) precedes fluctuation\npruning/bias compensation. The exact greedy-gain identity follows from ordinary\nSchur complements. This is evidence about a specific combination and its\nablations, not a claim to invent pruning, derivative distillation or a new theorem.\n",
    "Retain derivative-aware calibration as **PROMISING COMPONENT**. Complete\nrecipes remain unqualified; no beta sweep, additional probes or automatic\nlanguage run follows. A distinct justified next question is whether the simpler\nderivative-aware readout gives a better initialization for actual narrow-FFN\nlearning on fresh data, with a value-only counterpart and all preparation costs\ncharged. That would need its own frozen fitting protocol. The current result\ndoes not answer it, satisfy the 70%-compression quality goal, or establish broad\ncapability, scale, independent teacher-seed generalization or novelty.\n",
    "## Evidence\n",
    "- [Frozen plan and equations](sobolev_selection_plan.md).\n- [Source guide](../results/sobolev_selection_v1/source/README.md).\n- [Complete compressed results](../results/sobolev_selection_v1/result.json.gz), [CSV](../results/sobolev_selection_v1/metrics.csv.gz), [summary](../results/sobolev_selection_v1/summary.json), [ablations](../results/sobolev_selection_v1/ablation.json).\n- [Independent audit](../results/sobolev_selection_v1/audit.json.gz) and [final receipt](../results/verification/sobolev_selection_final_v1.json).\n\nRaw tensor statistics, original teacher inputs and all compressed checkpoints\nremain local and ignored. Their hashes are recorded; compressed metadata is\nnot a backup of those tensors.\n",
]
Path("research/sobolev_selection_results.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote research/sobolev_selection_results.md")
