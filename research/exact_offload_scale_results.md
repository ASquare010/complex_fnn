# H161: larger-model exact memory-helper probe

**PASS numeric/resource smoke gates.** Process RSS promised by the plan was not collected; full measurement
coverage is therefore incomplete. Pinned allocator counters do not measure total
process RAM. No long-training quality or breakthrough qualification is claimed.

The same 22,163,968-parameter FP32 Transformer is used in both arms (2.44x H156's
9,099,648 parameters). Width512, hidden608, 12 layers, eight heads, context512,
batch16, WikiText2 only. Compare native classifier loss with unchanged buffered
loss and last-four-block CPU checkpoint-input offload; whole-block recomputation
is enabled in both. No learned activation or parameter reduction is introduced.

| Seed | Allocated GPU saved | CUDA update ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---:|---:|---:|---|
| 401 | 18.10% | 1.0643 | 1.0643 | 1.000000 | none |
| 409 | 18.10% | 1.0069 | 1.0069 | 0.999981 | none |
| 419 | 18.10% | 1.0442 | 1.0442 | 1.000000 | none |

| Seed | Arm | Peak allocated MiB | Peak reserved MiB | Pinned peak MiB | Median CUDA ms | Final NLL |
|---|---|---:|---:|---:|---:|---:|
| 401 | ordinary | 925.58 | 1024.00 | 0.00 | 290.07 | 6.826442 |
| 401 | helper | 758.08 | 848.00 | 64.00 | 308.73 | 6.826443 |
| 409 | helper | 758.08 | 848.00 | 64.00 | 319.10 | 6.855114 |
| 409 | ordinary | 925.58 | 1024.00 | 0.00 | 316.91 | 6.855242 |
| 419 | ordinary | 925.58 | 1024.00 | 0.00 | 322.30 | 6.838170 |
| 419 | helper | 758.08 | 848.00 | 64.00 | 336.54 | 6.838170 |

Across-seed final validation NLL (three seeds; descriptive statistics):

| Arm | Mean | Median | Sample variance |
|---|---:|---:|---:|
| ordinary | 6.83995105 | 6.83816953 | 0.00020974175 |
| helper | 6.83990877 | 6.83816954 | 0.00020776432 |

## Evidence and interpretation

Six fresh30-update runs, three seeds, 180 accepted updates/186 backwards. Every
initial model hash and paired batch sequence matches. All final states are
reloaded and independently scored using native full-classifier cross entropy;
the measured loop uses FP32 chunked classification for full validation. Initial
full gradients are compared in NumPy FP64 under the original exact tolerances.
Memory includes model/data construction, initial probe, both full validation
passes and all optimizer updates. Per-update forward/backward/optimizer timing,
loss, gradient norm, and mean/median/sample variance are in the raw summary.
All six retained model checkpoints have unchanged parameter count.

The fixed gates are >=10% allocated-memory saving, <=15% median CUDA/wall
slowdown, <=1% final-NLL increase, <=128MiB pinned allocation, and the planned
within-run stability/interruption limits, plus numerical audits. Thirty updates
only screen execution and resource behavior; they cannot establish convergence,
robust speed across sessions, generalization quality, or unrelated-task benefit.
No selected seed is hidden and no aggregate rescues a failing seed.

## Failure and repair accounting

The original case00 completed30 updates but failed post-run cleanup before its
resource summary was written. Its gradients, final state, history and logs remain
untouched. A separate one-matrix, zero-training diagnostic reproduced8.125MiB of
retained cuBLAS workspace after empty_cache; clearing the workspace released it.
Recovery changes only post-measurement cleanup through a scoped process adapter,
as in the repository's existing boundary helper. All original source hashes,
measurements and gates remain intact. Recovery uses a separate output folder.
Total spent:210 updates/217 backwards, plus that one diagnostic matrix product.
The missing first-attempt resource summary cannot be reconstructed and is not used.

## Mechanism and next decision

At FP32, weights, gradients and two Adam moment arrays alone require16P bytes
when resident. Here that is338.20MiB, versus138.85MiB at H156 scale. Four saved
block inputs contain4*B*T*d FP32 values:64MiB here versus48MiB before. Thus fixed
checkpoint-input offload does not scale quadratically with the parameter state;
classifier-buffer savings and the actual peak phase also determine the result.
This accounting explains why width/parameter counts cannot substitute for measured
whole-job memory. It is not a universal lower bound for other optimizer/storage
schemes, and component savings do not necessarily add at the measured peak.

The recorded numeric/resource gates support a later sustained test at this scale. Repair missing RSS instrumentation before extending claims; do not promote short NLL as quality evidence.
The broader architectural and complex-pattern research goal remains open.

[Prospective plan](exact_offload_scale_plan.md),
[cleanup recovery](exact_offload_scale_recovery_plan.md),
[machine-readable evidence](../results/exact_offload_scale_v1/recovery/summary.json).
