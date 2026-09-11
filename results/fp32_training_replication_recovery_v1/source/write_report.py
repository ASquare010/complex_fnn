"""Render complete seed tables and descriptive statistics from audited evidence."""

import csv
import gzip
import io
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
NAMES = {
    "bf16_default_native": "BF16 native",
    "fp32_default_native": "FP32 native",
    "fp32_default_chunks": "FP32 chunks",
}


def read(name):
    return json.loads((ROOT / name).read_text())


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def run():
    for stage in ("study", "audit", "analyze"):
        assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
    result, audit, summary, protocol = [
        read(n) for n in ("result.json", "audit.json", "summary.json", "protocol.json")
    ]
    rows = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode())
        )
    )
    candidates = [r for r in rows if r["policy"] == "fp32_default_chunks"]
    pairs, endpoints = [], []
    for f in protocol["fixtures"]:
        peers = {r["policy"]: r for r in rows if r["fixture"] == f["label"]}
        a = peers["fp32_default_chunks"]
        quality = all(float(a["nll_ratio_" + ref]) <= 1.01 for ref in ("bf16", "native"))
        endpoints.append(
            [
                f["dataset"],
                f["seed"],
                *[f"{float(peers[p]['final_nll']):.6f}" for p in protocol["policies"]],
                f"{100 * (float(a['nll_ratio_bf16']) - 1):+.4f}%",
                f"{100 * (float(a['nll_ratio_native']) - 1):+.4f}%",
                "PASS" if quality else "FAIL",
            ]
        )
        pairs.append(
            [
                f["dataset"],
                f["seed"],
                f"{100 * (1 - float(a['memory_ratio_bf16'])):.3f}%",
                f"{100 * (1 - float(a['memory_ratio_native'])):.3f}%",
                f"{float(a['wall_ratio_bf16']):.4f}",
                f"{float(a['wall_ratio_native']):.4f}",
            ]
        )
    seed_table = table(
        [
            "Corpus",
            "Seed",
            "BF16 NLL",
            "FP32 native NLL",
            "FP32 chunks NLL",
            "Δ vs BF16",
            "Δ vs FP32 native",
            "Quality",
        ],
        endpoints,
    )
    paired_table = table(
        [
            "Corpus",
            "Seed",
            "Allocation saved vs BF16",
            "Saved vs FP32 native",
            "Time / BF16",
            "Time / FP32 native",
        ],
        pairs,
    )
    resource_table = table(
        [
            "Corpus",
            "Seed",
            "Policy",
            "Job allocated MiB",
            "Job reserved MiB",
            "Update median ms",
            "CUDA forward / backward / optimizer ms",
        ],
        [
            [
                r["scope"],
                r["seed"],
                NAMES[r["policy"]],
                f"{float(r['peak_job_mib']):.3f}",
                f"{float(r['peak_reserved_mib']):.3f}",
                f"{float(r['wall_update_ms']):.3f}",
                " / ".join(
                    f"{float(r[k]):.3f}"
                    for k in ("event_forward_ms", "event_backward_ms", "event_optimizer_ms")
                ),
            ]
            for r in rows
        ],
    )
    statistics_table = table(
        [
            "Corpus",
            "Policy",
            "NLL mean",
            "Median",
            "Sample variance",
            "Mean peak allocated MiB",
            "Mean of seed update medians ms",
        ],
        [
            [
                g["scope"],
                NAMES[g["policy"]],
                f"{g['metrics']['final_nll']['mean']:.6f}",
                f"{g['metrics']['final_nll']['median']:.6f}",
                f"{g['metrics']['final_nll']['sample_variance']:.8f}",
                f"{g['metrics']['peak_job_mib']['mean']:.3f}",
                f"{g['metrics']['wall_update_ms']['mean']:.3f}",
            ]
            for g in summary["groups"]
        ],
    )
    decisions = "\n".join(
        "- **"
        + d["scope"]
        + ": "
        + d["decision"]
        + ".** "
        + (
            "Every frozen gate passes across its three seeds."
            if d["qualifies_scoped_800_update_component"]
            else "Failed gates: " + ", ".join(k for k, v in d["gates"].items() if not v) + "."
        )
        for d in summary["decisions"]
    )
    max_pair = max(r["error"]["global_relative_l2"] for r in audit["paired_initial_errors"])
    max_tensor = max(r["error"]["max_tensor_relative_l2"] for r in audit["paired_initial_errors"])
    max_loss = max(r["loss_relative_error"] for r in audit["paired_initial_errors"])
    replays = [r["replay"] for r in audit["rows"] if r["replay"] is not None]
    assert {r["peak_phase"] for r in rows} == {"|".join(f"update_{s}" for s in range(2, 801))}
    clip_min, clip_max = (
        min(float(r["clip_fraction"]) for r in rows),
        max(float(r["clip_fraction"]) for r in rows),
    )
    all_time = sum(c["training_loop_wall_seconds"] for c in result["cases"])
    savings = [100 * (1 - float(r["memory_ratio_bf16"])) for r in candidates]
    time_ratios = [float(r["wall_ratio_bf16"]) for r in candidates]
    report = f"""# H117 — fresh three-seed replication of the FP32 memory recipe

**The fixed two-corpus claim {"passes in this scope" if summary["two_corpus_component_qualified"] else "fails"}.**
FP32 decoder/classifier chunking saves **{min(savings):.2f}–{max(savings):.2f}% full-job tensor allocation**
versus BF16 native, with {min(time_ratios):.4f}–{max(time_ratios):.4f} times its warmed update cost.
Memory savings alone do not qualify the complete recipe. Every seed must meet
the predeclared final-quality limit against both references.

{decisions}

This is an execution-policy replication on an unchanged narrow GELU model,
with **zero parameter reduction** between arms. It establishes no new activation,
new FFN, general superiority or breakthrough. The broader research goal remains open.

![Every seed and convergence trajectory](figures/fp32_training_replication.png)

## Experimental question and fixed controls

[H116](fp32_decoder_resource_results.md) passed a 50-update resource screen on
correlated saved states. H117 asks whether its promising default-attention FP32
recipe survives **800 updates from random initialization**, three seeds and two
corpora. [Original plan](fp32_training_replication_plan.md) and
[startup recovery plan](fp32_training_replication_recovery_plan.md) were frozen
before their respective execution. Quality, memory and runtime gates did not change.

The grid contains **18 fresh runs / 14,400 optimizer updates / 58,982,400 training
target presentations**. Seeds are 101, 113 and 127. All three arms within each
corpus/seed use the identical CPU step-zero model, empty Adam state and sampled
batches. The policy order rotates/reverses as recorded in the protocol. No
best-seed or best-checkpoint selection occurs.

- Model: width 384, FFN hidden 456, eight layers, six heads, vocabulary 4096,
  batch 8, context 512; **9,099,648 total / 2,801,664 FFN parameters** in every arm.
- Policies: BF16 native decoder/classifier; FP32 native decoder/classifier;
  FP32 decoder with checkpointed 512-token FP32 classifier chunks. All use
  default SDPA and whole-block checkpointing. Native FP32 isolates chunking;
  native BF16 is the complete practical reference.
- Training: fixed LR 0.0006, AdamW betas (0.9, 0.95), epsilon 1e-8, matrix decay 0.1,
  existing non-decay groups, norm clip 1; FP32 parameters and moments throughout.
  No precision-specific tuning or width calibration.
- Data: unchanged hashed WikiText-2 and TinyStories caches with separate
  train-only BPE tokenizers. Full development validation is scored at0/200/400/800
  with one streamed BF16/default scorer. Attention context and target weighting
  match across arms. No test set is used and raw corpus NLLs are not pooled.
- Hardware: one RTX 4070 Laptop GPU, approximately 8 GiB; PyTorch 2.14.0+cu132,
  Python 3.12.9, Windows 11, four CPU threads, TF32 disabled. The UV-managed
  interpreter uses the existing locked environment. No concurrent GPU worker.

The fixed acceptance rule requires, **for every seed**, NLL no more than1% above
both references, at least 15% lower allocated job peak, no more than 25% longer
median update time, stable timing and finite values. Initial paired gradients,
independent replays, native scores and state/batch integrity must also pass.
A favorable average cannot override a failed seed.

## What can be proved, and what must be measured

For fixed hidden states and classifier weights, partition the valid token set
into disjoint chunks. Let M be the total valid-target count and S_j the summed
cross-entropy for chunk j. Then in exact arithmetic:

```text
L = (sum over all valid tokens of token loss) / M
  = (sum over chunks j of S_j) / M
gradient(L) = (sum over chunks j of gradient(S_j)) / M.
```

The equalities follow by partitioning a finite sum and linearity of
differentiation. They require the same hidden states, parameters, valid-target
mask and global denominator; averaging unequal chunk means would be wrong.
Checkpoint recomputation preserves this function when it reproduces the same
forward calculation. The CPU-double qualification tests that implementation,
including ignored targets and partial chunks.

At this batch/context/vocabulary, a single dense FP32 logits tensor contains
4096×4096 values and occupies 64 MiB. A 512-token chunk contains 512×4096 values
and occupies 8 MiB: an **87.5% reduction in that one tensor's size**. Decoder
states, parameters, optimizer moments, gradients, other loss buffers and
workspaces remain. Consequently this calculation does not prove an 87.5%
whole-job saving; only the measured allocation tables support that claim.

Neither finite-sum identity proves bitwise GPU equivalence or convergence to
the same learned model. Rounding, reduction order and nonlinear optimizer
trajectories require separate numerical and learning tests. These are elementary
properties of token-separable loss, not a new theorem or unexplored architecture.

## Quality: every final endpoint

Lower NLL is better. Positive changes are worse. Only step800 is the primary
endpoint; earlier checkpoints describe convergence and cannot rescue a failed run.

{seed_table}

Three seeds provide limited statistical power. The reported seed variation is
descriptive, not a confidence interval or an equivalence proof. The failure of
a fixed recipe does not prove that all chunking or all FP32 training must fail.

## Measured memory and runtime

{paired_table}

Every allocation below is an observed **per-job tensor peak**, including
construction, CUDA corpus caches, optimizer state, the initial gradient probe,
training, validation, diagnostics and serialization. Normal library workspaces
are charged. Driver/context memory is outside PyTorch's allocator measurement.
Allocated and reserved memory are distinct; they are not added together.
The frozen qualification uses allocated memory, not reserved memory.

{resource_table}

Training updates 2–800 tie for the allocated job peak in all 18 runs.
All **28,998 memory intervals** are recorded before resets. Nineteen study
boundaries and25 independent-audit boundaries verify zero allocated/reserved
tensor storage between jobs. Audit memory is not mixed into a training-job peak.

Twenty updates warm up; all remaining780 contribute timing with no outlier
removal. Timed wall updates include forward/backward/clipping/Adam and CUDA
synchronization; batch hashing, scalar diagnostics and disk I/O are outside.
The optimizer CUDA column includes clipping. Component medians do not necessarily
sum to the median full update. The largest three-block timing-stability ratio is
{max(float(r["timing_stability"]) for r in rows):.6f}, below the1.25 limit.

Study elapsed time is **{result["elapsed_seconds"]:.2f}s ({result["elapsed_seconds"] / 60:.2f}min)**,
excluding initial-state creation, CPU qualification and independent audit.
Complete training-loop wall time sums to{all_time:.2f}s and includes intermediate
validation/checkpoints plus logging; it is not pure optimizer time. These are
training measurements, with no new inference-latency result.

## Variation, convergence and diagnostics

All groups contain three seeds. Sample variance uses denominator n−1.
The compact summary also contains mean/median/variance/range for memory,
runtime, clipping and each paired ratio; the tables retain every run.

{statistics_table}

The figure shows mean validation trajectories with ±1 sample standard deviation.
All72 validation points and14,400 update records are available as compressed CSV.
Layer-gradient statistics are recorded at updates1/20/200/400/800, plus the
initial unclipped probe; final sampled activations use the common BF16 forward.
All recorded losses, gradient norms and sampled diagnostics are finite, as are
final parameters/gradients/Adam states. Clipping fractions range from
{100 * clip_min:.3f}% to{100 * clip_max:.3f}% across runs. This does not establish a global
absence of vanishing/exploding gradients. Activations are fixed GELU, so there
are no learned activation-curve coefficients in this experiment.

## Independent audit and failure accounting

Independent audit {"passes" if audit["passed"] else "fails its fixed numerical gates"}.
It regenerates six initial CPU states exactly, verifies all54 trained
model/Adam/sampler checkpoints and18 saved initial-gradient hashes, reproduces
all14,400 sampled batches and computes60 complete native validation scores.
The largest native/streamed score relative error is
**{summary["maximum_native_score_relative_error"]:.3e}** against a1e-6 limit.

All 12 independent FP32 initial-gradient replays are recorded.
Bitwise-equal replays: {audit["bitwise_gradient_replays"]} of 12. Maximum replay global relative L2 is
{summary["maximum_gradient_replay_global"]:.3e}; maximum per-tensor error is
{max(r["error"]["max_tensor_relative_l2"] for r in replays):.3e}. The fixed limits are1e-5 and1e-4.
No BF16 gradient-replay qualification or full800-update trajectory replay is claimed.

At matched initialization, FP32 chunks versus native FP32 have maximum loss
relative error{max_loss:.3e}, global-gradient error{max_pair:.3e} and per-tensor
error{max_tensor:.3e}, below1e-6/0.002/0.02 respectively. Initial local gradient
agreement is insufficient evidence of endpoint agreement after800 updates.
The present experiment lacks duplicate800-update runs under identical policies,
so it cannot uniquely attribute an endpoint gap to chunking instead of ordinary
floating-point trajectory variability. The observed fixed endpoint gate still applies.

The first attempt failed before generating any initial checkpoint or taking any
training update. Environment metadata initialized CUDA before the initializer's
CPU-only guard. It completed24 CPU qualification backwards and then stopped.
The original source, traceback, logs and exit1 remain preserved. A prospectively
frozen recovery moved CPU initialization ahead of environment metadata and
qualification; the scientific recipe, seeds and limits stayed unchanged.
No training was repeated. This startup ordering bug is distinct from earlier
Windows native access violations; no general native-runtime cure is claimed.

After successful training, audit and analysis, the first plotting process hit
a Windows access violation during a Matplotlib import (raw exit 3221225477).
It produced no figure. The failed log, source and result inputs were frozen
in `plot_recovery_protocol.json` before one separate CPU process imported the
unchanged plotting code successfully. The rendered figure was visually checked.
No training, gradient replay, scoring, tolerance or scientific input was changed
or repeated. The cause of this native import failure remains unresolved.

Total accounting:14,418 main backwards (14,400 updates plus18 initial probes),
24 successful CPU-qualification backwards,24 from the failed startup and12 audit
backwards = **14,478 backwards across both attempts**. The recovery has18 completed
fresh runs. The reused `continuation_steps` checkpoint field is800; source step0,
regenerated initialization and empty Adam prove these runs began fresh.

## Decision and next discriminator

{decisions}

Keep the measured memory observation and close any failed fixed quality claim.
Do not promote a new default or allocate another large training sweep merely
because the short screen passed. A useful next diagnostic would compare paired
and within-policy trajectory variability under a prospectively fixed budget,
before changing precision, tolerances or optimizer settings. This report allocates
no such run. It also does not establish a new theoretical guarantee: the loss
is token-separable in exact arithmetic, while finite-precision accumulation and
nonlinear optimization can change the trained endpoint.

Classifier memory reduction is established prior work; see
[Cut Your Losses](https://arxiv.org/abs/2411.09009). The relevant numerical limits
are described in [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
These sources do not establish the cause of our specific endpoint difference.
H117 is an empirical replication, not a novelty claim.

## Reproducibility and retained artifacts

- [Frozen protocol](../results/fp32_training_replication_recovery_v1/protocol.json),
  [audit protocol](../results/fp32_training_replication_recovery_v1/audit_protocol.json),
  [complete summary](../results/fp32_training_replication_recovery_v1/summary.json).
- [Source guide](../results/fp32_training_replication_recovery_v1/source/README.md),
  [metrics](../results/fp32_training_replication_recovery_v1/metrics.csv.gz),
  [updates](../results/fp32_training_replication_recovery_v1/updates.csv.gz),
  [curves](../results/fp32_training_replication_recovery_v1/curves.csv.gz).
- Lossless compressed result/audit/logs and a
  [final verification receipt](../results/verification/fp32_training_replication_final_v1.json)
  preserve source hashes, arithmetic,78 tensor-artifact hashes and18 raw metric hashes.
- Initial/trained checkpoints, raw gradients, raw per-run histories and dataset
  caches remain local and ignored. They are not deleted. Independent tensor
  replay requires those artifacts; compact results alone cannot reconstruct them.
- All61 maintained code/config/test/lock hashes remain unchanged: two model
  folders, five variants and eight recipes. The earlier116-test pass is preserved
  separately; this round does not rerun or inflate that maintained-suite count.

The broad VRAM/quality/runtime goal and architectural parameter-efficiency
requirements remain unmet beyond the limited components explicitly qualified.
"""
    # Separate prose words from numbers without touching paths, IDs or URLs.
    report = re.sub(
        r"\b(at|than|step|and|remaining|the|to|All|all|updates|computes|a|are|full|error|below|after|duplicate|completed|exit|plus|has|is)(?=\d)",
        r"\1 ",
        report,
    )
    report = report.replace("accounting:14", "accounting: 14").replace(
        "backwards,24", "backwards, 24"
    )
    report = report.replace("arithmetic,78", "arithmetic, 78").replace("earlier116", "earlier 116")
    Path("research/fp32_training_replication_results.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            dict(
                decisions=summary["decisions"],
                elapsed_seconds=result["elapsed_seconds"],
                max_timing_stability=max(float(r["timing_stability"]) for r in rows),
                max_pair_error=max_pair,
                maximum_native_score_relative_error=summary["maximum_native_score_relative_error"],
                seed_count=3,
                median_candidate_time_ratio_bf16=st.median(time_ratios),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
