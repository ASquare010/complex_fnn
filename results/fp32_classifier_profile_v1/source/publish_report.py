"""Render readable H114 findings directly from the audited numeric summary."""

import csv
import gzip
import io
import json
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")
summary = json.loads((ROOT / "summary.json").read_text())
result = json.loads((ROOT / "result.json").read_text())
rows = list(
    csv.DictReader(io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode()))
)
scopes = ("narrow_wikitext2", "narrow_tinystories", "full_gelu", "full_swiglu")
names = (
    "Narrow GELU / WikiText-2, T512",
    "Narrow GELU / TinyStories, T512",
    "Full GELU / WikiText-2, T128",
    "Full SwiGLU / WikiText-2, T128",
)


def group(scope, policy):
    return next(
        g["metrics"] for g in summary["groups"] if g["scope"] == scope and g["policy"] == policy
    )


table = []
for scope, name in zip(scopes, names, strict=True):
    r, c = group(scope, "block_bf16"), group(scope, "chunks_fp32")
    table.append(
        f"| {name} | {r['peak_job_mib']['median']:.2f} | {c['peak_job_mib']['median']:.2f} | {100 * (1 - c['memory_ratio']['median']):.2f}% | {100 * (c['wall_ratio']['median'] - 1):+.2f}% | {100 * (c['final_nll_ratio']['min'] - 1):+.4f}% to {100 * (c['final_nll_ratio']['max'] - 1):+.4f}% |"
    )
controls = []
for scope, name in zip(scopes, names, strict=True):
    for policy in ("block_bf16", "chunks_bf16", "block_fp32", "chunks_fp32"):
        g = group(scope, policy)
        controls.append(
            f"| {name.split(' / ')[0]} ({'Tiny' if 'tinystories' in scope else 'Wiki'}) | {policy} | {g['peak_job_mib']['median']:.2f} | {g['wall_median_ms']['median']:.2f} | {g['event_median_ms']['median']:.2f} | {g['global_gradient_relative_error']['median']:.3g} |"
        )
decisions = []
for d in summary["decisions"]:
    failed = ", ".join(d["failed_gates"]) or "none"
    decisions.append(
        f"- `{d['scope']}`: original measured gates {'pass' if d['passes_original_measured_gates'] else 'fail'}; failed gates: {failed}. Gradient replay audit holds advancement."
    )
full = []
for row in rows:
    if row["scope"].startswith("full") and row["policy"] in ("block_bf16", "chunks_fp32"):
        full.append(
            f"| {row['scope']} | {row['policy']} | {float(row['training_peak_mib']):.2f} | {float(row['peak_job_mib']):.2f} |"
        )
max_stability = max(c["timing_stability_ratio"] for c in result["cases"])
gradient_gates = []
for scope, name in zip(scopes, names, strict=True):
    selected = [r for r in rows if r["scope"] == scope and r["policy"] == "chunks_fp32"]
    values = [float(r["global_gradient_relative_error"]) for r in selected]
    gradient_gates.append(
        f"| {name} | {min(values):.6f} to {max(values):.6f} | {sum(v > 0.002 for v in values)}/{len(values)} |"
    )
text = f"""# H114 — memory benefit survives; gradient replay remains unresolved

FP32 classifier chunks reduce measured whole-job allocation by **{100 * (1 - group("narrow_wikitext2", "chunks_fp32")["memory_ratio"]["median"]):.2f}% on WikiText-2** and
**{100 * (1 - group("narrow_tinystories", "chunks_fp32")["memory_ratio"]["median"]):.2f}% on TinyStories** in the narrow, context-512 models. The short continuations
finish with finite states and stable warmed timings. This is a useful scoped
resource observation, **not a qualified training recipe**: the original global
gradient-fidelity gate fails in every scope. Independent gradient replay also
fails both an exact audit and a subsequent numerical audit. Advancement
to fresh language training is held. The full-width, context-128 cases also fail
the required 15% whole-job saving. No parameter reduction or breakthrough follows.

![All fixture results](figures/fp32_classifier_profile.png)

## Controlled experiment

The [prospective plan](fp32_classifier_profile_plan.md) fixes 14 saved models and
four execution policies per model. All twelve H112 step-800 states are included:
two corpora, three training seeds and two previous training policies per seed.
Those two states are correlated fixtures, not six independent seeds. Two H078
step-3200 full controls add GELU/SwiGLU breadth, each with seed 17 only.

Each execution starts from identical source weights and Adam state and performs
50 sequential updates: 20 warmup and 30 measured. Total: **2,800 optimizer updates,
10,649,600 targets and 2,856 profile backwards**, completed once in
**{result["elapsed_seconds"]:.2f} seconds**. This is checkpoint continuation, not fresh training.
Batch/context stay 8/512 for narrow and 16/128 for full models. All decoders use
BF16 and whole-block checkpointing; only the classifier precision/chunking changes.
The vocabulary is 4096, width 384, eight layers and six attention heads.

Narrow models retain 9,099,648 total / 2,801,664 FFN parameters; full controls
retain 15,735,168 total / 9,437,184 FFN parameters. All weights, gradients and Adam
moments are FP32. The candidate adds **zero trainable parameters**. Dense FFNs,
activation functions and all 61 maintained files remain unchanged.

All cases use LR 0.0006, AdamW (0.9,0.95), epsilon 1e-8, matrix decay 0.1,
zero scalar decay and gradient clipping at 1. H078's saved terminal LR 0.00012
is deliberately replaced identically across its four continuations; optimizer
moments and steps are retained. A new common sampler seed (training seed +40000)
is used, with every batch hash preserved. No source checkpoint is overwritten.

## Whole-job resource and short-quality observations

Allocation includes model/gradients/Adam, CUDA corpus caches, normal workspaces,
probe, warmup, updates, evaluations, diagnostics and checkpoint preparation.
PyTorch's allocated counter excludes GPU driver/context memory. Reserved memory
is reported separately in the machine-readable table. Times are median paired
ratios across fixtures; quality uses independent native BF16 validation forward.
Positive NLL change means worse. This reused development validation is not a
new holdout or the official test set.

| Scope | Native BF16 job MiB | FP32 chunks job MiB | Allocation saving | Median update cost | Final NLL change range |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All 5,992 memory intervals are recorded before counter resets, fixing H113's
missing overall-peak accounting. First 20 updates count toward memory and learning
even though excluded from timing summaries. Every case passes the fixed warmed
timing-stability bound; the largest max/min ratio of three ten-update block means
is **{max_stability:.4f}**, below 1.25. No timing outlier was removed.

CUDA events enclose forward, backward and optimizer phases; only the end is
synchronized. Wall time includes clipping/Adam and the final synchronization,
while excluding sampling/hashes/evaluation/disk logging. Complete case elapsed
time is also recorded. Event time measures elapsed stream time, including
scheduling/submission gaps, not pure kernel activity. See the
[PyTorch CUDA guidance](https://docs.pytorch.org/docs/2.14/notes/cuda.html) and
[event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html).

## All four policies

Gradient errors below compare original saved initial probes against each
fixture's native FP32 classifier, with the decoder still BF16. They are measured
single-probe comparisons, **not fully independently replay-qualified gradients**.
The [summary](../results/fp32_classifier_profile_v1/summary.json) includes mean,
median, sample variance, range and per-seed aggregates. The
[56-case table](../results/fp32_classifier_profile_v1/metrics.csv.gz) and
[2,800-update table](../results/fp32_classifier_profile_v1/updates.csv.gz) retain all points.

| Model/corpus | Classifier policy | Job MiB | Median wall ms | Median event ms | Median initial gradient relative L2 |
|---|---|---:|---:|---:|---:|
{chr(10).join(controls)}

The original full-model global-gradient tolerance is 0.002 relative L2 against
the same model with a native FP32 classifier. Although FP32 chunks improve
the median error relative to BF16 classifier execution, they do not meet this
absolute requirement in every fixture. This is an original prospective gate
failure, independently of the later replay-audit hold. Per-tensor and initial
loss gates pass. H113's local classifier precision result does not extend to
the complete BF16 decoder gradient at the required tolerance.

| Scope | FP32 chunk global-gradient error range | Fixtures exceeding 0.002 |
|---|---:|---:|
{chr(10).join(gradient_gates)}

## Why the full-width memory result is weak

| Full control | Policy | Training peak MiB | Whole-job peak MiB |
|---|---|---:|---:|
{chr(10).join(full)}

The final layer-diagnostics phase sets both full-GELU peaks and the FP32-chunk
full-SwiGLU peak. Thus a smaller classifier training allocation does not establish
an equivalent whole-job reduction. Persistent parameter/gradient/Adam storage
also grows from about 138.85 MiB in narrow models to 240.10 MiB in full models.
Phase identification is measured; no individual diagnostic operator has been
isolated as a causal explanation. No bytes are subtracted to improve the gate.

## Numerical mechanism and its limit

With N tokens, vocabulary V and chunk C, a materialized FP32 logit tensor uses
4NV bytes, versus 4CV for one chunk, before cross-entropy intermediates. At
N=4096,V=4096,C=512 that is 64 MiB versus 8 MiB. Chunking sums the same loss in
real arithmetic. Its floating-point reduction and backward path need not be
identical. This elementary count explains why increasing classifier precision
can coexist with lower memory; it does not prove a whole-job saving.

[H113](classifier_precision_results.md) independently verified the classifier
cast/accumulation mechanism and FP64 derivative comparisons. H114's 16 small
double-precision complete-model checks also pass. Neither establishes bitwise
reproducibility of the full BF16 decoder backward. No new theorem, activation or
priority claim is made. Classifier-memory techniques already exist, including
[Cut Your Losses](https://arxiv.org/abs/2411.09009); these ordinary PyTorch chunks
are not that fused implementation.

## Failed replay audits are part of the result

The original audit stopped on a gradient hash mismatch after checking exact
weights, tokens, targets and scalar loss. A six-backward diagnostic of the first
native-BF16 case found five exact replays and one discrepancy with maximum
per-tensor relative L2 2.061e-6. It changed optimizer presence and preceding
evaluation together, so it did not establish a cause.

The [explicit recovery](fp32_classifier_profile_audit_recovery.md) introduced a
posthoc replay tolerance of 1e-5 global / 1e-4 per tensor, leaving every
prospective candidate gate unchanged. That recovery passed five probes and then
failed on `wikitext2_s61_loss_chunks__block_fp32`: global relative L2
**0.0005106002**, maximum tensor relative L2 **0.0009818187**, maximum absolute
difference **3.051758e-5**, with identical scalar loss. This larger discrepancy
shows the first diagnosis was not representative. The cause is unresolved;
there is no second tolerance relaxation or best-repeat selection.

The [completion audit](fp32_classifier_profile_completion_audit.md) performs zero
backwards: all **70 native scores, 56 saved gradient hashes, final model/Adam
states and 2,800 training batches** verify. Maximum streamed/native NLL drift is
**{summary["independent_audit"]["maximum_evaluation_relative_drift"]:.3g}**. That audit validates scores,
states and saved-gradient integrity; it does not certify gradient replay.

Both failed audit sources, protocols, logs and errors remain unchanged. The
diagnostic and failed recovery add six backwards each, with zero updates. The
first failed audit's exact backward count was not logged (one to four). Do not
claim an exact combined audit backward total. No scientific training was retried.
These are Python assertion failures, separate from earlier unresolved native
import crashes; H114 training completed without a native failure.

During reporting, an analyzer stalled before producing exports and was stopped;
an unchanged run with a traceback watchdog completed. A premature report/plot
attempt failed because those exports were absent. The original plot error and
exit remain preserved, and plotting succeeded after analysis. These ordering
and postprocessing failures are recorded in
[the postprocessing record](../results/fp32_classifier_profile_v1/postprocessing_record.json);
they caused no scientific rerun. The analyzer stall's cause is unknown.

## Decision

{chr(10).join(decisions)}

Retain the narrow-model memory observation as a component result. Close the
fixed full-model gradient-fidelity claim in all scopes and the fixed full-width
whole-job saving claim. **Hold fresh LM allocation for all
scopes until backward reproducibility is understood.** A separately frozen
matched-state diagnosis is the next useful discriminator; this round does not
automatically allocate another training grid or change maintained defaults.

Execution used UV-managed Python 3.12.9, PyTorch 2.14.0+cu132, the RTX4070 Laptop,
four CPU threads and TF32 off, with one GPU worker at a time. Fresh models share
one CUDA process; 57 case boundaries verify zero allocated/reserved bytes after
unused workspace cleanup. Workspaces count normally within cases. The
[source guide](../results/fp32_classifier_profile_v1/source/README.md) and
[final receipt](../results/verification/fp32_classifier_profile_final_v1.json)
separate completed science, failed gradient checks and narrower successful audit.
The prior 116-test maintained suite was not rerun because those files are unchanged.
The broader VRAM/quality and parameter-efficient architecture goals remain open.
"""
Path("research/fp32_classifier_profile_results.md").write_text(text, encoding="utf-8")
print("Research report generated from complete audited records")
