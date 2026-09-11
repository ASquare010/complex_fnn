"""All fixed gates, every rank and error/VRAM accounting from audited H105 output."""

import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

ROOT=Path("results/real_subspace_v1")


def read(name):
    return json.loads((ROOT/name).read_text())


def stats(values):
    return {"mean":st.mean(values),"median":st.median(values),"variance":st.variance(values),"values":values}


def run():
    result,audit,diagnosis=read("result.json"),read("audit.json"),read("rank_diagnosis.json")
    assert audit["passed"] and audit["score_count"]==378
    rows=result["rows"]
    methods=("pca","weight_input","linear_aware","energy","white_energy","white_residual","white_permuted")
    lookup={(r["teacher"],r["layer"],r["seed"],r["rank"],r["method"]):r for r in rows}
    groups={}
    for activation in ("gelu","swiglu"):
        groups[activation]={}
        for method in methods:
            groups[activation][method]={}
            for rank in (32,64,128):
                selected=[r for r in rows if (r["teacher"],r["method"],r["rank"])==(activation,method,rank)]
                groups[activation][method][str(rank)]={k:stats([r[k] for r in selected]) for k in
                    ("normalized_mse","inference_ms","peak_allocated_bytes","pipeline_peak_cuda_bytes","parameters","parameter_reduction")}
    decisions={}
    for method in ("energy","white_energy","white_residual"):
        decisions[method]={"teachers":{}}
        for activation in groups:
            candidates=[r for r in rows if (r["teacher"],r["method"],r["rank"])==(activation,method,64)]
            ratios={ref:math.exp(st.mean(math.log(r["normalized_mse"] / lookup[
                activation,r["layer"],r["seed"],64,ref]["normalized_mse"]) for r in candidates))
                for ref in ("pca","linear_aware","white_permuted")}
            layer_errors={str(layer):st.mean(r["normalized_mse"] for r in candidates if r["layer"]==layer) for layer in (0,3,7)}
            gates={"mean_error_at_most_five_percent":st.mean(r["normalized_mse"] for r in candidates)<=.05,
                "five_percent_better_than_pca":ratios["pca"]<=.95,
                "five_percent_better_than_linear_aware":ratios["linear_aware"]<=.95,
                "five_percent_better_than_null":ratios["white_permuted"]<=.95,
                "all_layer_errors_at_most_ten_percent":max(layer_errors.values())<=.10,
                "finite_and_factorized":all(r["finite"] and r["fold_error_fp64"]<=1e-9 for r in candidates)}
            decisions[method]["teachers"][activation]={"gates":gates,"ratios":ratios,"layer_errors":layer_errors}
        decisions[method]["verdict"]="EARNS_FITTING" if all(v for t in decisions[method]["teachers"].values() for v in t["gates"].values()) else "REJECTED_FIXED_PROJECTION_RECIPE"
    ranks={a:{str(r):stats([v["fp32_normalized_error_lower_bound"] for v in diagnosis["rows"] if (v["teacher"],v["rank"])==(a,r)]) for r in (32,64,128)} for a in groups}
    summary={"status":"COMPLETE","groups":groups,"decisions":decisions,"rank_error_floors":ranks,
        "audited_scores":378,"recaptured_pair_sets":18,"elapsed_seconds":result["elapsed_seconds"],"neural_updates":0,"breakthrough":False}
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with gzip.open(ROOT/"metrics.csv.gz","wt",newline="") as handle:
        keys=["teacher","layer","seed","rank","method","parameters","normalized_mse","inference_ms","peak_allocated_bytes","pipeline_peak_cuda_bytes"]
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in rows)
    lines=["# H105: synthetic label-energy discovery does not transfer to these FFNs","",
        "All three fixed energy/projection recipes are **rejected**. The Gaussian",
        "mechanism from H103/H104 does not beat simpler input PCA or activation-aware",
        "linear response on these real Transformer FFN inputs. No neural fitting is",
        "allocated to this branch, and no language-quality improvement is claimed.","",
        "The screen completes 378 local comparisons in {:.2f} seconds, with zero SGD".format(result["elapsed_seconds"]),
        "updates. An independent process recaptures all 18 input/output datasets through",
        "the full decoder and checks every projection score, matrix, basis and parameter",
        "count. Score tolerance is 1e-7 absolute plus 1e-5 relative, using a different",
        "batch partition and explicit factor algebra; this is not a claim of bitwise scores.","",
        "## Primary rank-64 result","",
        "Normalized output MSE divides squared error by teacher output variation about",
        "its training mean. Lower is better; zero is the original FP32 FFN function.","",
        "| Method | GELU mean error | SwiGLU mean error |","|---|---:|---:|"]
    for method in methods:
        lines.append(f"| {method} | {groups['gelu'][method]['64']['normalized_mse']['mean']:.4f} | {groups['swiglu'][method]['64']['normalized_mse']['mean']:.4f} |")
    lines += ["","Values average three depths and three calibration seeds per teacher.",
        "Whitening and fitting affine residual energy do not repair the failed energy",
        "signal here. Passing a shuffled-label comparison alone is insufficient: the",
        "useful target is a competitive small FFN, not merely nonzero supervision.","",
        "![Every tested rank](figures/real_subspace.png)","",
        "## Why a cheap input initializer is insufficient","",
        "A full affine fit still leaves appreciable output error, especially for SwiGLU.",
        "The projection must preserve nonlinear behavior, not just high-variance inputs.",
        "The activation-aware linear baseline is stronger than the energy proposals,",
        "but it too exceeds the 5% local reconstruction budget at rank 64.","",
        "Post-screen, we measured an output-rank obstruction. Every tested model has",
        "a final r-dimensional linear output subspace. For a finite centered teacher",
        "output matrix, the squared singular-value tail is the best possible error of",
        "any affine rank-r output prediction, regardless of its input-side nonlinearity.",
        "This is the [Eckart-Young identity](https://doi.org/10.1007/BF02288367),",
        "not a novelty-priority claim. Reported bounds are FP64 numerical evaluations",
        "of the analytical result, not interval-arithmetic certificates.","",
        "The saved outputs use BF16 teacher execution. We subtract their measured",
        "distance to fresh FP32 outputs using the reverse triangle inequality. If V",
        "is the FP32 output variance, t is the BF16 squared singular-value tail divided",
        "by N*d*V, and delta is BF16/FP32 MSE divided by V, the FP32 lower bound is:","",
        "    normalized MSE >= max(0, sqrt(t) - sqrt(delta))^2","",
        "The oracle SVD uses reporting outputs solely for this post-screen bound; no",
        "model or hyperparameter is fitted from it. It cannot be reported as a learned",
        "model. It concerns this local function and this dataset, not Transformer",
        "residual outputs, next-token NLL, or what a network can learn from scratch.","",
        "| Output rank | GELU mean error floor | SwiGLU mean error floor |","|---|---:|---:|"]
    for rank in (32,64,128):
        lines.append(f"| {rank} | {ranks['gelu'][str(rank)]['mean']:.4f} | {ranks['swiglu'][str(rank)]['mean']:.4f} |")
    lines += ["","The rank-64 mean floors are about **11% for GELU and 21% for SwiGLU**.",
        "Consequently, the frozen 5% threshold was unattainable for this output-rank",
        "class on these datasets even with an ideal input-side function. This was",
        "discovered after the screen; it is not an initialization-specific failure.",
        "Energy's additional regression versus simpler controls remains measured.",
        "Future compression screens should check the output-rank floor before a grid.","",
        "This bound quantifies a limitation of a global output bottleneck. It does not",
        "reject structured full-rank maps, grouped bottlenecks, or learned models that",
        "change the surrounding network. The original architectural goal remains open.","",
        "## Memory, timing and actual cost","",
        "Rank-64 GELU/SwiGLU use 247,680 / 248,192 parameters, about 79% fewer than",
        "the corresponding 1,179,648-weight full FFNs. Original hidden widths remain",
        "1536 / 1024, so this does not imply a similar activation-memory reduction.","",
        "| Teacher / method | Rank-64 inference ms | Local inference peak MiB | Calibration pipeline peak MiB |","|---|---:|---:|---:|"]
    for activation in groups:
        refs=[d["teacher_resources"] for d in result["diagnostics"] if d["teacher"]==activation]
        lines.append(f"| {activation} / original FFN | {st.mean(v['inference_ms'] for v in refs):.3f} | {st.mean(v['peak_allocated_bytes'] for v in refs)/2**20:.2f} | — |")
        for method in ("pca","linear_aware","energy","white_energy","white_residual"):
            r=groups[activation][method]['64']
            lines.append(f"| {activation} / {method} | {r['inference_ms']['mean']:.3f} | {r['peak_allocated_bytes']['mean']/2**20:.2f} | {r['pipeline_peak_cuda_bytes']['mean']/2**20:.2f} |")
    lines += ["","Collection retains the full pretrained teacher on GPU. The pipeline peak takes",
        "the maximum of collection and compressed inference, so low student storage",
        "does not erase teacher/calibration costs. Statistics are fitted on CPU FP64.",
        "All methods share collected pairs; per-collection times and full estimator",
        "times are retained. This screen computes all estimators as a diagnostic grid,",
        "not as a claimed optimal single-method preprocessing implementation.","",
        "Native inference is batch 256, FP32, after warmup with CUDA synchronization.",
        "Local inference peaks use 512-row evaluation batches and include parameter",
        "storage and temporary tensors. They exclude CPU pairs, CUDA driver/context",
        "and other processes. No training-memory, generation-latency or total-training",
        "cost claim follows. Previous teacher training is reused and was not free.","",
        "## Data, correctness and limitations","",
        "The teachers are existing audited seed-17, 3,200-update WikiText-2 full GELU",
        "and SwiGLU models. Layers 0, 3 and 7 are tested. Each calibration seed draws",
        "256 nonoverlapping 128-token training-cache windows (32,768 tokens), then",
        "64 disjoint reporting windows (8,192 tokens). Seeds vary sampling, not teacher",
        "training; neighboring tokens within a window are correlated. The teacher",
        "has already trained on this corpus. This is not unseen-corpus evidence.","",
        "Capture uses BF16 model execution; all means, bases and weights use calibration",
        "data only. Evaluation compares FP32 models on captured FFN inputs, with",
        "BF16/FP32 target drift separately reported. No validation/test tokens or NLL",
        "evaluation are used. Ranks 32/64/128 are all reported without selecting a winner.","",
        "Nineteen initial qualification checks verify exact factorization, input",
        "gradients, full-rank identity, the linear-response optimum and capture parity",
        "with full decoder execution. Finite forward outputs do not certify stable",
        "training gradients; this screen has no optimizer trajectory.","",
        "[Frozen plan and algebra](real_subspace_plan.md),",
        "[all-rank statistics and gates](../results/real_subspace_v1/summary.json),",
        "[all 378 rows](../results/real_subspace_v1/metrics.csv.gz),",
        "[independent audit](../results/real_subspace_v1/audit.json),",
        "[rank diagnosis](../results/real_subspace_v1/rank_diagnosis.json).", "",
        "Source is under results/real_subspace_v1/source; the reusable capture helper",
        "is src/core/ffn_capture.py. The active FFN factory and training defaults are",
        "unchanged. Data/checkpoints stay local; full results are also packed losslessly.","",
        "```powershell","uv run --no-sync python -m results.real_subspace_v1.source.audit",
        "uv run --no-sync python -m results.real_subspace_v1.source.diagnose",
        "uv run --no-sync python -m results.real_subspace_v1.source.analyze",
        "uv run --no-sync python -m results.real_subspace_v1.source.plot","```","",
        "[ASVD](https://arxiv.org/abs/2312.05821), [SVD-LLM](https://arxiv.org/abs/2403.07378)",
        "and [IO-SVD](https://arxiv.org/abs/2605.15626) are relevant established compression",
        "precedents. This study does not reproduce their full systems or claim SOTA.",""]
    Path("research/real_subspace_results.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"decisions":decisions,"rank_floors":ranks},indent=2))


if __name__=="__main__":
    run()
