"""Fixed-gate analysis; publish every recipe and distinguish mechanism from candidate."""

import csv
import gzip
import json
import math
import statistics as stats
from pathlib import Path

import torch

from results.spectral_fitting_v1.source.model import FORMS
from src.core.reproducibility import write_json

ROOT = Path("results/spectral_fitting_v1")
TASKS = ("quadratic", "cubic", "product", "piecewise")
SEEDS = (17, 29, 43)
TINY = ("tiny_gelu_bounded", "tiny_swiglu_bounded", "tiny_cubic_bounded", "tiny_relu2_bounded")


def read(path):
    return json.loads(Path(path).read_text())


def summary(values):
    return {
        "mean": stats.mean(values),
        "median": stats.median(values),
        "variance": stats.variance(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def run():
    result, audit = read(ROOT / "result.json"), read(ROOT / "audit.json")
    assert audit["passed"] and audit["score_count"] == 816
    selected = [r for r in result["rows"] if r["selected"]]
    assert len(selected) == 204
    lookup = {(r["task"], r["form"], r["seed"]): r for r in selected}
    per_task = {
        form: {
            task: summary([lookup[task, form, seed]["reporting_mse"] for seed in SEEDS])
            for task in TASKS
        }
        for form in FORMS
    }
    resources = {
        form: {
            key: summary([r[key] for r in selected if r["form"] == form])
            for key in ("peak_allocated_bytes", "median_update_ms", "inference_ms")
        }
        for form in FORMS
    }
    # Estimation precedes training and releases its GPU tensors. Charge every
    # recipe its full prepass, including the peak, rather than amortizing the grid.
    for form in FORMS:
        rows = [r for r in selected if r["form"] == form]
        resources[form]["pipeline_peak_allocated_bytes"] = summary(
            [max(r["peak_allocated_bytes"], r["preprocessing"]["peak_cuda_bytes"]) for r in rows]
        )
        resources[form]["preprocessing_seconds"] = summary(
            [r["preprocessing"]["seconds"] for r in rows]
        )
        resources[form]["preprocessing_peak_bytes"] = summary(
            [r["preprocessing"]["peak_cuda_bytes"] for r in rows]
        )
        durations = []
        for row in rows:
            history = (ROOT / "cells" / row["label"] / "history.jsonl").read_text().splitlines()
            durations.append(sum(json.loads(h)["update_ms"] for h in history) / 1000
                             + row["preprocessing"]["seconds"])
        resources[form]["prepass_plus_update_seconds"] = summary(durations)
    zero = {}
    for seed in SEEDS:
        data = torch.load(ROOT / "data" / f"s{seed}.pt", weights_only=True)
        zero[seed] = {task: data["targets"][task][69632:].square().mean().item() for task in TASKS}
        del data

    def ratio(form, ref, seeds=SEEDS):
        return math.exp(
            stats.mean(
                math.log(lookup[t, form, s]["reporting_mse"] / lookup[t, ref, s]["reporting_mse"])
                for t in TASKS
                for s in seeds
            )
        )

    decisions = {}
    for activation in ("gelu", "cubic"):
        candidate = f"factor_{activation}_bounded"
        controls = (*TINY, f"factor_{activation}_raw", f"factor_{activation}_random")
        ratios = {
            f: ratio(candidate, f) for f in (*controls, "full_gelu_bounded", "full_swiglu_bounded")
        }
        seed_ratios = {f: [ratio(candidate, f, (s,)) for s in SEEDS] for f in TINY}
        per_task_caps = {
            t: per_task[candidate][t]["mean"] / min(per_task[f][t]["mean"] for f in TINY)
            for t in TASKS
        }
        counts = lookup[TASKS[0], candidate, SEEDS[0]]["parameters"]
        gates = {
            "ninety_percent_parameter_reduction": all(
                counts <= 0.1 * lookup[TASKS[0], f, SEEDS[0]]["parameters"]
                for f in ("full_gelu_bounded", "full_swiglu_bounded")
            ),
            "cubic_halves_zero_each_seed": all(
                lookup["cubic", candidate, s]["reporting_mse"] <= 0.5 * zero[s]["cubic"]
                for s in SEEDS
            ),
            "within_one_percent_fulls": all(
                ratios[f] <= 1.01 for f in ("full_gelu_bounded", "full_swiglu_bounded")
            ),
            "two_percent_over_controls": all(ratios[f] <= 0.98 for f in controls),
            "wins_each_seed_over_tinies": all(v < 1 for vs in seed_ratios.values() for v in vs),
            "per_task_cap": all(v <= 1.05 for v in per_task_caps.values()),
            "memory_saving": all(
                resources[candidate]["peak_allocated_bytes"]["mean"]
                <= 0.75 * resources[f]["peak_allocated_bytes"]["mean"]
                for f in ("full_gelu_bounded", "full_swiglu_bounded")
            ),
            "update_time": resources[candidate]["median_update_ms"]["mean"]
            <= 1.25 * resources["narrow_gelu_bounded"]["median_update_ms"]["mean"],
            "inference_time": resources[candidate]["inference_ms"]["mean"]
            <= 1.25 * resources["narrow_gelu_bounded"]["inference_ms"]["mean"],
        }
        decisions[candidate] = {
            "gates": gates,
            "ratios": ratios,
            "per_task_caps": per_task_caps,
            "seed_ratios": seed_ratios,
            "verdict": "EARNS_CONFIRMATION" if all(gates.values()) else "REJECTED_AT_THIS_BUDGET",
        }
    mechanisms = {
        f"factor_{a}_{m}": ratio(f"factor_{a}_{m}", f"factor_{a}_random")
        for a in ("gelu", "cubic")
        for m in ("raw", "bounded")
    }
    alignment = {
        form: {
            task: {
                phase: summary([
                    r[phase] for r in audit["factor_alignment"]
                    if r["selected"] and r["label"].startswith(f"{task}_{form}_s")
                ]) for phase in ("initial_recovery", "final_recovery")
            } for task in TASKS
        } for form in FORMS if form.startswith("factor_")
    }
    output = {
        "per_task": per_task,
        "resources": resources,
        "decisions": decisions,
        "mechanism_ratios_vs_random": mechanisms,
        "factor_alignment": alignment,
        "zero_errors": zero,
        "audit_passed": True,
        "neural_runs": 408,
        "score_audits": 816,
        "breakthrough": False,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    write_json(ROOT / "summary.json", output)
    compact = [
        {
            k: r[k]
            for k in (
                "label",
                "task",
                "form",
                "seed",
                "rate",
                "selected",
                "parameters",
                "selection_mse",
                "reporting_mse",
                "peak_allocated_bytes",
                "median_update_ms",
                "inference_ms",
                "clip_fraction",
            )
        }
        for r in result["rows"]
    ]
    with gzip.open(ROOT / "metrics.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    lines = [
        "# H103/H104: feature discovery helps; factorization must earn its cost",
        "",
        "This study investigates H100's cubic-learning failure with training-label spectral",
        "initialization and 66,048-parameter FFNs. No new activation or novelty is claimed.",
        "",
        "## Decisions",
        "",
        "The bounded-initialized cubic factorization is a useful **synthetic lead**,",
        "but fails its fixed promotion gates. With 66,048 parameters (94.41% fewer",
        "than either full FFN), it has lower mean error on all four tasks than the",
        "equally initialized full GELU and SwiGLU controls. Its paired aggregate",
        "MSE is 41.01% / 15.97% lower respectively, and 57.48% below tiny cubic.",
        "",
        "The failures matter: on cubic alone, tiny cubic is better (0.095797 versus",
        "0.141608 MSE), a 47.82% regression exceeding the 5% cap. Native inference",
        "takes 0.308 ms versus narrow GELU's 0.145 ms, exceeding the 25% allowance.",
        "These gates stay fixed. The GELU factorization also fails full-model quality",
        "and cubic-learning gates. Neither recipe earns automatic language training.",
        "",
        "Training-only peak allocation falls about 51-52% versus the full models,",
        "but preprocessing raises the candidate's pipeline peak to **30.77 MiB**.",
        "Including that cost, savings are **25.02% versus full GELU and 22.68% versus",
        "full SwiGLU**, not 50%. Equally initialized narrow/tiny controls have the same",
        "preprocessing peak, so the factorization does not reduce pipeline VRAM below",
        "them at this batch size. This is CUDA tensor allocation, excluding driver",
        "context and other processes; it is not the GPU usage displayed by nvidia-smi.",
        "",
    ]
    for name, row in decisions.items():
        failed = [k for k, v in row["gates"].items() if not v]
        lines += [f"- **{name}: {row['verdict']}**. Failed gates: {', '.join(failed) or 'none'}."]
    lines += [
        "",
        "The primary architectural/VRAM goal remains active. These synthetic, Gaussian",
        "low-rank target families are deliberately favorable to subspace discovery and",
        "polynomial features; they cannot establish language-model or broad task superiority.",
        "",
        "## Independent signal screen (H103)",
        "",
        "The bounded estimator recovers 88.59%, 88.93% and 88.84% of the cubic input",
        "subspace on three independent datasets. Permuted labels recover 7.63%, 8.34%",
        "and 8.30%; raw moments recover 64.13%, 64.72% and 63.83%. All fixed allocation",
        "gates pass. A separate audit reconstructs all 36 moments with a different chunk",
        "partition and checks eigenpairs, subspace recovery and data hashes.",
        "",
        "For Gaussian X and labels depending only on Q^T X, the centered weighted",
        "covariance lies in span(Q). For degree-k normalized Hermite targets, its raw",
        "population eigenvalues are proportional to 2k: the cubic energy has a signal",
        "even though the first label/input cross-moment vanishes. The",
        "[proof and assumptions](spectral_discovery_plan.md) do not guarantee finite-sample",
        "recovery, positive eigengaps or optimization. The bounded transform changes the",
        "spectrum; its improvement here is measured, not assumed.",
        "",
        "## Fitting outcomes (H104)",
        "",
        "Means over three independent dataset/training seeds after selecting one of two",
        "rates using a separate selection split. Lower reporting MSE is better.",
        "",
        "| Form | Parameters | Quadratic | Cubic | Product | Piecewise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for form in FORMS:
        lines.append(
            f"| {form} | {lookup[TASKS[0], form, 17]['parameters']:,} | "
            + " | ".join(f"{per_task[form][t]['mean']:.6f}" for t in TASKS)
            + " |"
        )
    lines += [
        "",
        "All 408 runs, 816 checkpoint scores and 204 rate selections are independently",
        "checked. The audit reconstructs datasets and spectral subspaces and exactly",
        "reconstructs every recorded model initialization. Factorized forms share U,",
        "hidden biases and zero readouts across initializer modes; only A differs.",
        "",
        "[All-rate compact CSV](../results/spectral_fitting_v1/metrics.csv.gz),",
        "[means/medians/variances, gates and ratios](../results/spectral_fitting_v1/summary.json),",
        "[checkpoint audit](../results/spectral_fitting_v1/audit.json).",
        "The complete maintained suite passes 113 tests after the documented runtime",
        "recovery. [Final verification receipt](../results/verification/spectral_final_v1.json).",
        "",
        "![Task fitting results](figures/spectral_fitting.png)",
        "",
        "## Resource costs",
        "",
        "Training data remains on CPU, with the same current-batch transfers for all forms.",
        "Peak allocation includes the model, gradients, optimizer, current batch and",
        "training diagnostics, and excludes later evaluation/export. Reserved memory is",
        "recorded separately. Update time includes gathering/transferring batches; model",
        "forward/backward and inference are separately timed. These are standalone FFN",
        "numbers, not Transformer end-to-end VRAM or autoregressive generation latency.",
        "",
        "| Form | Train peak MiB | Pipeline peak MiB | Update ms | Inference ms | Prepass + updates s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for form in FORMS:
        r = resources[form]
        lines.append(
            f"| {form} | {r['peak_allocated_bytes']['mean'] / 2**20:.2f} | "
            f"{r['pipeline_peak_allocated_bytes']['mean'] / 2**20:.2f} | "
            f"{r['median_update_ms']['mean']:.3f} | {r['inference_ms']['mean']:.3f} | "
            f"{r['prepass_plus_update_seconds']['mean']:.3f} |"
        )
    lines += [
        "",
        "Each spectral recipe also requires a 65,536-example label/input pass and a",
        "384-by-384 moment eigensolve. Per-estimator wall time and peak CUDA memory",
        "are recorded under initializers; they are **not divided by the number of grid",
        "runs**. Random controls do not use the label prepass. Comparisons against",
        "equally initialized controls isolate factorization more fairly, but no matched",
        "total-compute superiority is claimed. Data generation and source/checkpoint I/O",
        "are included in study wall time, not model update time.",
        "",
        "Pipeline peak is max(preprocessing peak, training peak), because these phases",
        "run sequentially and preprocessing GPU tensors are released. The last column",
        "charges the complete prepass plus all 300 measured update intervals. It excludes",
        "data generation, checkpoint I/O and diagnostics outside those intervals, so it",
        "is not total study wall time. The frozen memory gate concerns training only;",
        "the stricter pipeline accounting is reported additionally without changing it.",
        "",
        "## Mechanism and gradient limits",
        "",
        "Cubic error falls from 1.015055 with random factor initialization to 0.141608",
        "with bounded initialization: about 86% less. Full SwiGLU also benefits, falling",
        "from 1.083979 to 0.152723. Giving this initializer to the candidate alone would",
        "therefore greatly exaggerate an architectural advantage.",
        "",
        "On the cubic task, the bounded factor begins with about 89% recovered input",
        "subspace. The cubic model finishes near 97-98%; the GELU model loses alignment",
        "and finishes near 65-71%. These checkpoint diagnostics support a feature",
        "discovery/retention interpretation, but do not isolate its causal components.",
        "Random controls do not pay for training-label access: this is an initializer",
        "intervention with additional data processing, not a compute-matched win.",
        "",
        "Every recorded weight, optimizer moment, loss and gradient is finite across",
        "408 fits. This only establishes finite short-run behavior under clipping;",
        "the cubic derivative is unbounded, and no nonvanishing-gradient, general",
        "stability or long-horizon learning theorem follows.",
        "",
        "## Experimental recipe and limitations",
        "",
        "Fresh inputs: 65,536 train / 4,096 selection / 4,096 reporting rows, d=384 and",
        "hidden teacher rank 16. Each seed (17/29/43) regenerates data, rotations and",
        "sampling, so reported variance combines data and optimizer variation. All",
        "initializers use training labels only; teacher directions appear solely in",
        "oracle qualification and post-training diagnostics. Rank 32 is fixed.",
        "",
        "Each run: 300 updates of batch 256, AdamW (.9,.95), eps 1e-8, zero decay,",
        "clip 1 and constant LR .001 or .003. Readout LR uses the same width calibration",
        "rule for every model. GPU FP32, TF32 disabled, four CPU threads; UV and one",
        "RTX 4070 Laptop GPU. Timing excludes the first 50 updates. Cubic derivatives",
        "are unbounded; clipping/activation/gradient histories are retained, not hidden.",
        "",
        "The candidate is A(384->32), U(32->128), activation, D(128->384), totaling",
        "66,048 parameters including biases. All matrices train. Polynomial oracle",
        "construction and gradient checks establish scoped capacity; training is never",
        "initialized with that construction. The initial 22 qualification checks cover",
        "counts, zero outputs, probe variance, CUDA finiteness, factor gradients and",
        "three polynomial capacity witnesses. Pairing is additionally checked in audit.",
        "",
        "Constant-norm one-hot language labels give exactly zero energy signal. Applying",
        "this mechanism to language would require a different source of training signal,",
        "such as teacher-output fitting, and full accounting of teacher cost. No such",
        "application, mixed-precision result, long-run convergence or broader corpus",
        "claim is established here. The previous memory study and its failures stand.",
        "",
        "## Reproduction and prior work",
        "",
        "[H103 plan](spectral_discovery_plan.md), [H104 frozen plan](spectral_fitting_plan.md),",
        "[model](../results/spectral_fitting_v1/source/model.py),",
        "[shared trainer](../src/core/function_fitting.py). Source hashes and environment",
        "are frozen in each protocol. Raw datasets and all checkpoints stay local and",
        "are ignored in Git; a clean clone must rerun before rescoring them.",
        "Full result JSON is retained losslessly in `result.json.gz`; the original",
        "JSON remains local. Restore it before running historical readers in a clone.",
        "",
        "```powershell",
        "uv run --no-sync python -m results.spectral_fitting_v1.source.audit",
        "uv run --no-sync python -m results.spectral_fitting_v1.source.analyze",
        "uv run --no-sync python -m results.spectral_fitting_v1.source.plot",
        "```",
        "",
        "Fresh execution uses the study module in a checkout with empty experiment output",
        "directories. Existing completed directories are refused, preserving old evidence.",
        "",
        "The first post-audit analysis produced numeric tables but no figure, and its",
        "following pytest process reported a Windows access violation under Python",
        "3.12.12. The [failure record](../results/spectral_fitting_v1/postprocess_failure.json)",
        "is retained. Numeric analysis/testing recovery uses existing UV Python 3.12.9",
        "with the same installed packages. Plotting is isolated from Torch in a separate",
        "process after the first recovery also failed inside Matplotlib layout. Original",
        "training, frozen source and the",
        "816-score checkpoint audit remain unchanged. This is not a training retry.",
        "",
        "[Second-order Stein estimation](https://papers.neurips.cc/paper/7190-estimating-high-dimensional-non-gaussian-multiple-index-models-via-steins-lemma.pdf),",
        "[Gaussian multi-index gradient flow](https://proceedings.mlr.press/v258/simsek25a.html),",
        "[The Generative Leap](https://arxiv.org/abs/2506.05500) and",
        "[layerwise neural feature learning](https://arxiv.org/abs/2511.15120) are relevant",
        "precedents. The small factorization and polynomial activation are established",
        "ideas. The contribution here is a controlled experiment and its limitations,",
        "not a certified novel primitive or a breakthrough.",
        "",
    ]
    Path("research/spectral_fitting_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()
