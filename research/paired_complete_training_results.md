# H134: complete training with an 8-MiB workspace

**FAIL complete short-training gate.** Independent numerical/native-score audit: **True**.
All eight corpus/control/repeat comparisons below must pass individually.
These are short continuations from trained step-800 fixtures, not fresh training
or a multi-seed quality result. Prior failed gates remain recorded.

| Corpus | Repeat | Control | GPU allocation saved | Complete CUDA ratio | Wall ratio | NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
| wikitext2 | 0 | ordinary | 24.66% | 1.1582 | 1.1581 | 0.99999849 | event, wall |
| wikitext2 | 0 | native | 14.51% | 0.9961 | 0.9962 | 1.00000000 | none |
| wikitext2 | 1 | ordinary | 24.66% | 1.1648 | 1.1647 | 0.99999962 | event, wall |
| wikitext2 | 1 | native | 14.51% | 0.9995 | 0.9994 | 1.00000000 | none |
| tinystories | 0 | ordinary | 25.04% | 1.1726 | 1.1725 | 0.99999919 | event, wall |
| tinystories | 0 | native | 14.76% | 0.9331 | 0.9331 | 1.00000000 | none |
| tinystories | 1 | ordinary | 25.04% | 1.1585 | 1.1584 | 0.99999520 | event, wall |
| tinystories | 1 | native | 14.76% | 0.9875 | 0.9875 | 1.00000000 | none |

Candidate is H128 buffer-reuse classifier plus checkpoint-input CPU offload,
strict deterministic default attention and explicit 8-MiB external cuBLAS
workspaces. Matched native uses the same configuration with native loss.
Ordinary uses native loss, GPU-resident checkpoint inputs and ordinary default
execution (observed 8.125-MiB workspace). Parameters remain 9,099,648.
This uses the existing [PyTorch workspace API](https://docs.pytorch.org/docs/stable/backends);
it is not a novel activation or FFN architecture.

## Runs and measured scope

Each corpus executes reuse, ordinary, native, native, ordinary, reuse in fresh
processes. Both mirrored repeats must meet memory <=0.90, complete-update CUDA
and wall <=1.15, NLL <=1.01, candidate pinned host peak <=128 MiB, and within-run
halves timing ratio <=1.15. No best-run selection or averaging away failures.
Repeats share one fixture/seed; they do not estimate independent-seed variance.

| Run | Corpus | Arm | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing halves ratio | Telemetry samples |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | wikitext2 | reuse | 306.181 | 64.000 | 80.149 | 80.200 | 4.9633912 | 1.0022 | 9 |
| 1 | wikitext2 | ordinary | 406.399 | 0.000 | 69.202 | 69.248 | 4.9633988 | 1.0044 | 8 |
| 2 | wikitext2 | native | 358.149 | 64.000 | 80.460 | 80.506 | 4.9633912 | 1.0442 | 9 |
| 3 | wikitext2 | native | 358.149 | 64.000 | 80.382 | 80.436 | 4.9633912 | 1.0063 | 8 |
| 4 | wikitext2 | ordinary | 406.399 | 0.000 | 68.976 | 69.021 | 4.9633931 | 1.0025 | 7 |
| 5 | wikitext2 | reuse | 306.181 | 64.000 | 80.345 | 80.389 | 4.9633912 | 1.0283 | 9 |
| 6 | tinystories | reuse | 300.001 | 64.000 | 80.995 | 81.043 | 2.6276819 | 1.0026 | 8 |
| 7 | tinystories | ordinary | 400.220 | 0.000 | 69.071 | 69.118 | 2.6276840 | 1.0063 | 7 |
| 8 | tinystories | native | 351.970 | 64.000 | 86.806 | 86.854 | 2.6276819 | 1.1200 | 9 |
| 9 | tinystories | native | 351.970 | 64.000 | 80.003 | 80.047 | 2.6276819 | 1.0020 | 9 |
| 10 | tinystories | ordinary | 400.220 | 0.000 | 68.197 | 68.240 | 2.6276945 | 1.0009 | 7 |
| 11 | tinystories | reuse | 300.001 | 64.000 | 79.005 | 79.048 | 2.6276819 | 1.0012 | 8 |

Whole-update CUDA elapsed time spans forward, backward, clipping and Adam,
including checkpoint recomputation and offload transfers. Synchronized wall
time ends before post-step memory inventory. Timing excludes batch generation,
gradient clearing, validation, serialization and inventory, while reported
whole-job peak CUDA allocation includes them. First ten updates are warmup;
all raw steps, means, medians, sample variance and timing stability are retained.
Reserved CUDA and pinned-host peaks are reported separately in machine-readable
results. CUDA allocation excludes driver/context and other processes; pinned
host peak is not total CPU RSS. This harness differs from H130's phase-barrier
instrumentation, so no cross-study absolute-runtime causal claim is valid.

The loop is derived from H120 by asserted textual instrumentation edits only:
remove three phase inventory barriers and stop whole-update timers before the
remaining post-step inventory. Optimizer, clipping, data, checkpointing and
validation numerical operations are unchanged. The derived source is frozen. An independent syntax-tree check reconstructs
the declared edits in memory and verifies exact agreement after formatting.

## Verification and telemetry

360 optimizer updates/backwards, 1,474,560 training targets, 24 full study
validation scores and 12 independent native scores. There are 36 tensor
artifacts, 804 memory intervals, and 37 zero allocator boundaries including
the independent audit. All 61 maintained source/config/test files are unchanged.

Inherited NumPy checks verify original clipping and first Adam updates/moments.
Every final checkpoint is scored through independent native code and all batch
hashes are replayed. Native/reuse deterministic runs are compared bitwise for
first/final model, optimizer and sampler state, first raw/clipped gradients,
all 30 loss/norm pairs and before/after validation. The audit result above
includes these checks. Ordinary execution is audited under original numerical
and validation tolerances, not an invalid bytewise reproducibility requirement.

Read-only GPU telemetry samples every 200 ms; every monitor is stopped in a
finally block. Run-level clock, power and temperature ranges/medians and raw
samples are retained. Missing sensors stay missing. Coverage is required within
updates 11-30; this is sparse observational telemetry, not proof of a throttling
or kernel-selection cause. No power or clock settings were changed.

The initial launch crashed during NumPy import with Windows access violation
3221225477, before any case artifacts or updates. The driver stopped. A frozen
recovery plan switched all scheduled and audit processes to PYTHONMALLOC=pymalloc,
retaining original logs and scientific sources. This operational change does not
establish the crash cause; no completed scientific case was repeated.

## Next decision

Test the same candidate under ordinary/default attention policy, alongside native loss with and without checkpoint-input offload. Calibrate full-model numerical tolerances against repeated ordinary native controls, retaining the existing exact operator qualification. This isolates whether strict deterministic execution is the remaining practical runtime cost; do not assume it is the cause. Keep paired complete-update timing, native scoring and memory gates, and do not extend long training yet.
The broad VRAM, quality and parameter-efficiency goal remains open.

[Prospective plan](paired_complete_training_plan.md),
[recovery plan](paired_complete_training_recovery_plan.md),
[summary](../results/paired_complete_training_v1/summary.json),
[receipt](../results/paired_complete_training_v1/receipt.json).
