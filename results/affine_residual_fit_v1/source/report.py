"""Build the readable H107 report only after independent audit and diagnostics."""

import json
import statistics as st
from pathlib import Path

ROOT = Path("results/affine_residual_fit_v1")


def read(name):
    return json.loads((ROOT / name).read_text())


def run():
    summary, audit, diagnostic, memory = (
        read(n)
        for n in ("summary.json", "audit.json", "diagnostic_summary.json", "inference_memory.json")
    )
    assert audit["passed"] and diagnostic["all_finite"]
    stats = {(r["teacher"], r["form"]): r for r in summary["statistics"]}
    forms = (
        "affine_gelu",
        "affine_swiglu",
        "gelu",
        "relu",
        "leaky_relu",
        "prelu",
        "silu",
        "swiglu",
    )
    names = {
        "affine_gelu": "Affine + GELU h256",
        "affine_swiglu": "Affine + SwiGLU h170",
        "gelu": "GELU h448",
        "relu": "ReLU h448",
        "leaky_relu": "LeakyReLU h448",
        "prelu": "PReLU h448",
        "silu": "SiLU h448",
        "swiglu": "SwiGLU h298",
    }
    lines = [
        "# H107: affine residual FFNs fail the matched-budget comparison\n",
        "**Both fixed recipes are rejected.** They use over 70% fewer FFN parameters\nand roughly 42-45% less local training allocation than full teacher profiles,\nbut reconstruct the teacher less accurately than strong narrow controls and\nfail the raw-inference timing gate. Neither earns language insertion, longer\ntraining, kernel work or integration. The broader VRAM/quality goal is unmet.\n",
        f"All **288 fits, 172,800 updates and 144 selections** completed in\n{summary['elapsed_seconds'] / 60:.2f} minutes on one RTX 4070 Laptop GPU. Independent\nverification recaptures all 18 datasets, recomputes **864 scores**, checks 144\ninitializations and validates all selected exports. Separate native Windows\npostprocessing failures and their successful recovery are retained below.\n",
        "![Capacity, learning, memory and latency](figures/affine_residual.png)\n",
        "## What was tested\n",
        "[H106's capacity calculation](affine_residual_capacity_results.md) qualified\nGELU h256 and SwiGLU h170 corrections after an explicit full-rank affine path:\n\n\\[f(x)=Wx+b+D\\phi(Ux+a).\\]\n\nThe gated form replaces phi(Ux+a) by SiLU(Ux+a) times (Vx+c). The full affine\npath avoids H105's global output-rank restriction, but its d-by-d matrix consumes\nparameters that an ordinary FFN can spend on additional nonlinear features.\nThe test asks whether decoupling the linear path compensates for those lost\nfeatures. It does not assume that a favorable matrix bound proves trainability.\n",
        "Two existing seed-17, 3,200-update teachers supply depth 0/3/7 inputs/outputs.\nAll H105 windows are excluded. Seeds 71/83/97 receive disjoint new groups of\n320 nonoverlapping 128-token windows: 256 for calibration, 32 for selecting\ncheckpoints and 32 for reporting. These are sampling/optimization replications,\nnot three independent teacher training seeds. The windows come from the original\nWikiText-2 training cache, which the teachers already saw during training.\nNo claim of unseen-domain or official language-test performance follows.\n",
        "Teacher capture uses BF16; stored BF16 outputs are the labels. Students use\nFP32 with TF32 disabled. Training-only coordinate means/SDs normalize inputs;\na training-only global RMS normalizes centered outputs. Deployment folds these\nconstants into weights and biases. MSE below is divided by the training output\nvariance. H106 uses different windows and its recorded FP32 reference variance,\nso the capacity and fitting plots are not a paired numerical gap estimate.\n",
        "Every model receives a shared prefix of the same permuted teacher feature\nbank, followed by CPU FP64 ridge fitting of its readout. Candidate designs jointly\nfit inputs and hidden features; ordinary controls fit hidden features with the\nsame standardized-column ridge and unpenalized intercept. No reporting-label\noracle from H106 enters an initialization. All parameters then train using the\nsame sample stream, batch 256, 600 AdamW updates, clipping at norm 1, and rates\n0.001/0.003. Selection considers initialization and both final checkpoints using\nselection MSE only. Every endpoint is retained, including unsuccessful rates.\n",
        "## Reconstruction at the same parameter budget\n",
        "| Form | Parameters | Saving vs full FFN | GELU-teacher mean MSE | SwiGLU-teacher mean MSE |\n|---|---:|---:|---:|---:|",
    ]
    for form in forms:
        g, s = stats["gelu", form], stats["swiglu", form]
        lines.append(
            f"| {names[form]} | {g['parameters']:,} | {g['parameter_reduction']:.2%} | {g['reporting_mse']['mean']:.6f} | {s['reporting_mse']['mean']:.6f} |"
        )
    lines += [
        "\nEach mean covers three depths and three sampling seeds. No form reaches the\n0.05 mean allocation threshold for either teacher. SiLU is the best ordinary\ncontrol on the GELU teacher; SwiGLU is best on the SwiGLU teacher. Their relative\nadvantages are useful control evidence, not a promotion under a changed gate.\n",
        "| Candidate | Teacher | Paired MSE ratio to best control | Raw inference / narrow GELU | Update / narrow GELU |\n|---|---|---:|---:|---:|",
    ]
    for form in forms[:2]:
        for teacher, best in (("gelu", "silu"), ("swiglu", "swiglu")):
            d = summary["decisions"][form]["teachers"][teacher]
            lines.append(
                f"| {names[form]} | {teacher} | {d['paired_geomean_ratios'][best]:.4f} | {d['raw_inference_ratio_to_gelu']:.3f} | {d['update_ratio_to_gelu']:.3f} |"
            )
    lines += [
        "\nRatios above one are worse. Paired reconstruction ratios are geometric means\nover matching depth/seed cells. Both candidates lose to the strongest control\nin every seed aggregate and at every tested depth. Their quality, all-control,\nall-seed, depth and inference gates fail for both teachers. Parameter count,\nfinite training, local training allocation and the 1.25x update-time ceiling pass.\nAll comparisons against all six controls are in the machine-readable summary.\n",
        "| Teacher / form | Seed 71 mean | Seed 83 mean | Seed 97 mean | Nine-cell median | Nine-cell sample variance |\n|---|---:|---:|---:|---:|---:|",
    ]
    for teacher in ("gelu", "swiglu"):
        for form in ("affine_gelu", "affine_swiglu", "gelu", "silu", "swiglu"):
            r = stats[teacher, form]
            values = " | ".join(f"{r['seed_mse'][str(seed)]:.6f}" for seed in (71, 83, 97))
            lines.append(
                f"| {teacher} / {names[form]} | {values} | {r['reporting_mse']['median']:.6f} | {r['reporting_mse']['sample_variance']:.6g} |"
            )
    lines += [
        "\nThe nine-cell variance includes depth heterogeneity; it is not a teacher-seed\nconfidence interval. Mean, median, sample variance, range, every depth mean and\nevery endpoint are retained for all eight forms in the summary/CSV.\n",
        "## Memory and compute\n",
        "| Form | Mean local training peak MiB | Mean raw inference peak MiB | Parameters accounted |\n|---|---:|---:|---|",
    ]
    for form in forms:
        train = (
            st.mean(stats[t, form]["peak_allocated_bytes"]["mean"] for t in ("gelu", "swiglu"))
            / 2**20
        )
        infer = (
            st.mean(r["peak_allocated_bytes"] for r in memory["students"] if r["form"] == form)
            / 2**20
        )
        lines.append(f"| {names[form]} | {train:.3f} | {infer:.3f} | All weights and biases |")
    for teacher in ("gelu", "swiglu"):
        train = summary["full_profiles"][teacher]["peak_allocated_bytes"]["mean"] / 2**20
        infer = (
            st.mean(
                r["peak_allocated_bytes"]
                for r in memory["full_teachers"]
                if r["teacher"] == teacher
            )
            / 2**20
        )
        lines.append(
            f"| Full {teacher} reference | {train:.3f} | {infer:.3f} | 1,179,648 raw weights; normalization biases in training profile |"
        )
    lines += [
        f"\nThese are peak **PyTorch tensor allocations**, not total process/driver VRAM.\nThe complete compression pipeline peaks at **{summary['pipeline_peak_cuda_bytes'] / 2**20:.3f} MiB**,\nincluding the full teacher during collection. All students require that common\ncapture step. The roughly 24 MiB fitting numbers do not establish a corresponding\nend-to-end pipeline saving. Ordinary narrow controls already use about the same\nlocal training allocation. Reserved-memory values are retained separately.\n",
        "The original full FFNs have no biases. Algebraically equivalent normalized\nfull profiles add 1,920 GELU or 2,432 SwiGLU bias parameters, about 0.16%/0.21%\noverhead, to absorb input/output normalization. They take 30 zero-learning-rate\nAdamW steps per dataset, with 10 timing warmups, to allocate gradients and\noptimizer states. The 18 profiles add 540 resource-only updates, not teacher\ntraining or a matched quality experiment. Inference references use the original\nbias-free full FFNs. Post-run no-gradient inference-memory profiles use batch\n256, 10 warmup and 10 measured forwards, with no optimizer or dataset on GPU.\nThese additional measurements do not alter the frozen timing/quality decisions.\n",
        "The affine-GELU and ordinary-GELU models both require 344,064 dense matrix\nMACs per token. The two parameter-matched SwiGLU forms each require 343,296.\nThus the candidate does not reduce leading matrix arithmetic at this budget;\nit trades nonlinear features for a separate affine matrix and output addition.\nScalar activations, biases, elementwise products, optimizer work and kernel\nlaunches are excluded from that MAC count and included in measured timing.\nNo fused-kernel or large-batch speed claim is made.\n",
        f"Capture takes {summary['collection_seconds']:.2f} seconds in total and CPU\ninitialization takes {summary['initialization_seconds']:.2f} seconds. Both rates\nare charged even when a single endpoint is selected. Timings use synchronization\nand warmup, but remain sequential measurements on one laptop GPU. The prototype\npipeline includes CPU-resident data and the existing CUDA runtime's allocation\noverhead. It is not a complete Transformer training/serving memory benchmark.\n",
        "## Optimization and verification\n",
        f"All 288 fits retain finite weights and Adam moments. The largest recorded\npreclip gradient norm is {diagnostic['max_preclip_norm']:.4f}; the largest run-level\nclipping fraction is {diagnostic['max_clip_fraction']:.2%}. This is bounded-run\nevidence under clipping, not a proof against vanishing/exploding gradients at\narbitrary depth. Loss histories, activation distributions and parameter-gradient\nsnapshots at initialization and training checkpoints remain available.\n",
        "Only the PReLU control learns a scalar activation parameter; its initial\nand both final slopes are saved for each dataset. The candidates use fixed\nGELU/SwiGLU and learn their projection matrices. No learned-curve novelty is\nimplied. The compact diagnostic archive includes 50-step loss/gradient summaries\nand snapshots; unaggregated histories remain unchanged locally.\n",
        "Ten qualification groups check all eight parameter counts, finite-difference\ninput and parameter gradients, FP64 normalization-folded outputs and input\ngradients, independent augmented least-squares ridge solutions, normalized\nteacher equivalence and shared row prefixes. The audit then independently\nrecaptures all 18 complete pair datasets, checks training-only statistics and\nsampling streams, verifies 144 real ridge stationarity conditions and reconstructs\nevery checkpoint using direct tensor algebra with a different batch partition.\n",
        f"All 864 selection/report scores agree within the stated numerical tolerance;\nthe largest absolute discrepancy is {audit['max_score_difference']:.3g}. All 144\nselections and raw exports pass. Export verification covers every reporting and\nselection row. Source/teacher/data/checkpoint hashes and all update records are\nchecked. The maintained suite still has 116 passing tests; no active model or\nfactory entry was added.\n",
        "The first audit process failed during Torch import with a native Windows\naccess violation, before any dataset was checked. The first plot process failed\nin Matplotlib font lookup. Their original logs, exit codes and source hashes\nare retained. A separately recorded fresh-process recovery uses the same UV\nPython 3.12.9 and installed packages with PYTHONMALLOC=malloc and\nPYTHONHASHSEED=107. The independent audit and plot complete there. This does\nnot prove the cause of the Windows failures. No fit, scientific endpoint,\nsource from the frozen training snapshot or decision threshold was replaced.\n",
        "## Decision and remaining question\n",
        "Close these exact affine h256/h170 recipes. The capacity calculation was\nuseful for eliminating impossible small widths, but insufficient for selecting a\ncompetitive learned compressor. Spending a dense matrix's parameters on the\nexplicit linear path did not compensate for reduced nonlinear width here. That\nlast statement is an empirical explanation consistent with this comparison,\nnot a theorem excluding all affine-residual architectures.\n",
        "A next design must address measured memory with a quality advantage over\nthe strongest ordinary control and must account for its initialization/capture\nrequirements. Uniform local reconstruction remains far from the fixed gate;\nthis result does not justify automatic language training or a claim of universal\ncompression failure. Language NLL, longer schedules, independently trained\nteachers, larger models and unrelated data remain untested for these prototypes.\n",
        "The [FFN linear-recoverability study](https://arxiv.org/abs/2606.19379) directly\nprecedes affine FFN fitting and warns that local reconstruction and perplexity\ncan dissociate. [Wide & Deep](https://arxiv.org/abs/1606.07792) precedes the basic\nlinear-plus-nonlinear family. No architectural or activation novelty is claimed.\n",
        "## Reproducible artifacts\n",
        "- [Frozen H107 plan](affine_residual_fit_plan.md) and [H106 proof/results](affine_residual_capacity_results.md).\n- [Source/reproduction guide](../results/affine_residual_fit_v1/source/README.md).\n- [All endpoints, lossless JSON](../results/affine_residual_fit_v1/result.json.gz), [CSV](../results/affine_residual_fit_v1/metrics.csv.gz), [summary and decisions](../results/affine_residual_fit_v1/summary.json).\n- [Independent audit](../results/affine_residual_fit_v1/audit.json.gz), [diagnostics](../results/affine_residual_fit_v1/diagnostics.json.gz), [inference memory](../results/affine_residual_fit_v1/inference_memory.json).\n- [Final verification receipt](../results/verification/affine_residual_final_v1.json).\n\nRaw tensors and checkpoints stay local and ignored. Compressed metadata preserves\nthe original bytes; it is not a substitute for missing tensors in a fresh clone.\n",
    ]
    Path("research/affine_residual_fit_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote research/affine_residual_fit_results.md")


if __name__ == "__main__":
    run()
