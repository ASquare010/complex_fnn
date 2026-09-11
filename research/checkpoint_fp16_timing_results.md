# H159: FP16 checkpoint storage under complete optimizer updates

**PASS complete-update resource/timing gate.** All 36 segments completed: 1,080 updates, 8,847,360 training targets,
72 study validation scores and 36 independent native final-state scores.
Independent clipping/first-Adam/moment, gradient, batch and native-scoring audit:
**True**. The broader goal remains open.

Across these six fixtures, FP16 saved 23.77-23.99% of native peak GPU allocation.
Complete CUDA medians ranged from 0.60% faster to 0.56% slower than native and
3.25-4.13% faster than offload4. These are observed paired short-run timings;
fresh-training convergence and sustained throughput remain untested.

## Every fixture, against both controls

The candidate is buffered classifier loss plus eight FP16 GPU checkpoint inputs.
Ordinary uses native loss and GPU FP32 inputs; buffer4 uses the same buffered loss
with four FP32 checkpoint inputs offloaded to pinned CPU memory. Parameter count
is 9,099,648 for every arm. No learned activation or parameter reduction is added.

| Corpus | Seed | Reference | GPU allocation saved | Complete CUDA ratio | Wall ratio | Final NLL ratios, repeats0/1 | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
| wikitext2 | 101 | ordinary | 23.77% | 1.0025 | 1.0025 | 1.000007 / 1.000001 | none |
| wikitext2 | 101 | buffer4 | 0.00% | 0.9587 | 0.9587 | 1.000003 / 1.000000 | none |
| wikitext2 | 113 | ordinary | 23.77% | 1.0029 | 1.0029 | 0.999996 / 0.999998 | none |
| wikitext2 | 113 | buffer4 | 0.00% | 0.9623 | 0.9622 | 1.000002 / 1.000000 | none |
| wikitext2 | 127 | ordinary | 23.77% | 1.0032 | 1.0033 | 1.000003 / 1.000000 | none |
| wikitext2 | 127 | buffer4 | 0.00% | 0.9613 | 0.9614 | 1.000005 / 0.999999 | none |
| tinystories | 101 | ordinary | 23.99% | 0.9991 | 0.9990 | 1.000011 / 0.999994 | none |
| tinystories | 101 | buffer4 | 0.00% | 0.9635 | 0.9635 | 0.999995 / 1.000005 | none |
| tinystories | 113 | ordinary | 23.99% | 1.0056 | 1.0056 | 1.000001 / 1.000018 | none |
| tinystories | 113 | buffer4 | 0.00% | 0.9675 | 0.9675 | 1.000002 / 1.000007 | none |
| tinystories | 127 | ordinary | 23.99% | 0.9940 | 0.9940 | 1.000012 / 1.000004 | none |
| tinystories | 127 | buffer4 | 0.00% | 0.9657 | 0.9657 | 0.999998 / 1.000003 | none |

Ratios are candidate/reference. Timing uses the combined 40 measured updates per
arm and fixture after 10 warmups in each 30-update segment. The native limit is 1.15;
the offload limit is 1.05 for both CUDA and wall medians. GPU allocation uses the
maximum whole-segment peak across both repeats, capped at 0.90 native and 1.02
offload4. Each repeat's final NLL must be <=1.01 of its corresponding reference.
All fixtures and all stability checks must pass; aggregates do not rescue failures.

## Absolute resource measurements

| Arm | Whole-segment GPU allocated peak range MiB | Maximum allocator-owned pinned host MiB |
|---|---:|---:|
| ordinary | 658.267-664.446 | 64.000 |
| buffer4 | 500.330-506.509 | 64.000 |
| fp16 | 500.330-506.509 | 64.000 |

Memory includes model, Adam state, resident data, construction, evaluation,
training, first-gradient copies, state serialization and diagnostic phases. It is
allocator memory, excluding driver/context and unrelated processes, and is not
isolated inference memory. This shared-process study may retain offload host
caches; its cap is 128 MiB. The isolated host advantage was qualified separately in
[H158](checkpoint_host_isolation_results.md), including native's five-byte peak.
H157's failed absolute-zero host gate remains failed.

## Timing stability

Every segment's two 10-update half-medians and every arm's between-repeat medians
must have max/min <=1.15 for both CUDA and wall time. Failures below are descriptive;
no repeat removal, clock normalization or retrospective threshold changes occur.

All within-segment and between-repeat stability gates pass.

Passive 200 ms telemetry covers every timed segment if its telemetry gate passes.
Per-segment states, clock/temperature/power samples, all paired repeat ratios, and
mean/median/sample variance are retained in telemetry.csv and summary.json. This
is not a long-duration throughput measurement; sensor variation alone cannot
prove the cause of a slowdown.

## Controlled execution and independent evidence

Use the six H156 ordinary step800 model/Adam/sampler states, two corpora,
seeds 101/113/127. Each segment resets to the same fixture. ABC CBA order reverses
to CBA ABC on alternating fixtures. Batch 16, context512, validation batch8,
FP32, TF32 off, four CPU threads, ordinary attention, 8.125MiB cuBLAS workspace,
AdamW LR0.0006/betas0.9,0.95/eps1e-8, original decay and gradient clipping1.

The H142 training loop is imported unchanged. Timers encompass forward/backward,
FP16 casts and FP32 restoration, checkpoint recomputation, CPU offload transfers,
clipping and Adam. Sampling, gradient clearing, logging, validation and saving are
outside update timing. First-step gradient/state copies occur only during warmup.
No completed case is rerun. All 36 segments have clean GPU boundaries; the 36
native audit evaluations have separate clean boundaries, 74 boundary records total.

The FP16 codec is unchanged from H157. It saves 240 block inputs and unpacks 240
per segment; no full activation snapshots are retained. H157 established local
approximate-VJP semantics, not exact differentiation of the FP32 forward. H159
compares full saved first gradients against native per repeat under the same
separately frozen FP16 caps (global.002, max tensor.02, cosine.99999). Exact
controls retain 1e-5 global/1e-4 tensor limits; loss relative error<=1e-6 for all.

Independent NumPy verifies clipping and first Adam/moment updates using each arm's
actual saved gradients, with unchanged H142 tolerances. This checks the optimizer
implementation; it does not assert that approximate gradients yield the same
parameter trajectory. Native scoring independently verifies all 36 final states,
and saved sampler states regenerate all 1,080 batches. First/final model and moment
hashes, counters 801/830, finite states, stored artifacts and source hashes verify.
The audit adds zero backwards or optimizer updates. Full results remain in audit.json.

## Decision

The candidate earns fresh multi-seed long training. That next test must assess convergence and sustained timing from initialization before any quality-preserving claim.
This short continuation cannot establish fresh-training quality, stronger complex
pattern learning, novel activation geometry, fewer parameters, SOTA or a breakthrough.
H156 remains the existing long-training qualification of the exact offload helper.

[Prospective plan](checkpoint_fp16_timing_plan.md),
[machine-readable summary](../results/checkpoint_fp16_timing_v1/summary.json),
[independent audit](../results/checkpoint_fp16_timing_v1/audit.json).
