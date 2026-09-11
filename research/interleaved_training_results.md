# H140: interleaved complete-training comparison

**PASS scoped interleaved timing gate.** Independent numerical/native audit: **True**.
H138's failed long-training qualification remains unchanged. No new activation,
parameter reduction, SOTA or broad-goal achievement is claimed.

| Corpus | Source seed | GPU allocation saved | Complete CUDA ratio | Wall ratio | Failed gates |
|---|---:|---:|---:|---:|---|
| wikitext2 | 101 | 18.69% | 1.0261 | 1.0261 | none |
| wikitext2 | 113 | 18.69% | 1.0289 | 1.0289 | none |
| wikitext2 | 127 | 18.69% | 1.0310 | 1.0311 | none |
| tinystories | 101 | 18.98% | 1.0291 | 1.0291 | none |
| tinystories | 113 | 18.98% | 1.0132 | 1.0131 | none |
| tinystories | 127 | 18.98% | 1.0223 | 1.0223 | none |

All six ordinary H138 step800 model/Adam/sampler states are used. For each, four
independent30-update segments reset to that exact state, in ABBA or BAAB order
alternating by fixture. A=ordinary, B=buffer4. These are repeated continuations,
not120 updates of a continuous trajectory. Twenty updates per segment remain
after ten fixed warmups. Primary ratios compare all40 timed updates per arm.
Every seed must pass; descriptive aggregate means cannot rescue a failure.

## Neighboring temporal pairs

| Corpus | Source seed | Repeat | CUDA ratio | Wall ratio | Final native NLL ratio |
|---|---:|---:|---:|---:|---:|
| wikitext2 | 101 | 0 | 1.0303 | 1.0304 | 1.000000 |
| wikitext2 | 101 | 1 | 1.0238 | 1.0238 | 1.000002 |
| wikitext2 | 113 | 0 | 1.0335 | 1.0335 | 0.999999 |
| wikitext2 | 113 | 1 | 1.0234 | 1.0234 | 1.000002 |
| wikitext2 | 127 | 0 | 1.0311 | 1.0311 | 1.000003 |
| wikitext2 | 127 | 1 | 1.0249 | 1.0247 | 1.000002 |
| tinystories | 101 | 0 | 1.0219 | 1.0219 | 0.999997 |
| tinystories | 101 | 1 | 1.0353 | 1.0352 | 1.000003 |
| tinystories | 113 | 0 | 1.0146 | 1.0146 | 1.000004 |
| tinystories | 113 | 1 | 1.0120 | 1.0120 | 1.000008 |
| tinystories | 127 | 0 | 1.0198 | 1.0198 | 0.999999 |
| tinystories | 127 | 1 | 1.0226 | 1.0227 | 1.000006 |

The frozen gates are peak allocated memory<=0.90, complete CUDA/wall time<=1.15,
both repeat NLL ratios<=1.01, pinned hostpeak<=128MiB, every segment's half-median
stability<=1.15 and each arm's two-repeat median ratio<=1.15. Telemetry coverage
and numerical/native-score/initial-hash/batch-order checks are mandatory.

## Per-segment resources and setup

| Segment | Arm | GPU peak MiB | Complete CUDA ms | Within-segment stability | Setup ms | Entire segment ms | Median active SM MHz | P-states |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | ordinary | 406.40 | 67.18 | 1.0067 | 615.9 | 7325.3 | 2445.0 | P0 |
| 1 | buffer4 | 330.43 | 69.21 | 1.0023 | 604.4 | 6466.1 | 2370.0 | P0 |
| 2 | buffer4 | 330.43 | 69.39 | 1.0002 | 628.0 | 6487.2 | 2430.0 | P0 |
| 3 | ordinary | 406.40 | 67.77 | 1.0039 | 574.4 | 6404.8 | 2295.0 | P0 |
| 4 | buffer4 | 330.43 | 69.66 | 1.0084 | 598.2 | 6542.3 | 2415.0 | P0 |
| 5 | ordinary | 406.40 | 67.40 | 1.0006 | 590.8 | 6394.8 | 2310.0 | P0 |
| 6 | ordinary | 406.40 | 67.48 | 1.0047 | 597.8 | 6530.0 | 2280.0 | P0 |
| 7 | buffer4 | 330.43 | 69.06 | 1.0035 | 588.9 | 6520.7 | 2430.0 | P0 |
| 8 | ordinary | 406.40 | 67.28 | 1.0062 | 593.0 | 6383.5 | 2415.0 | P0 |
| 9 | buffer4 | 330.43 | 69.37 | 1.0067 | 599.0 | 6479.8 | 2340.0 | P0 |
| 10 | buffer4 | 330.43 | 69.66 | 1.0037 | 590.4 | 6510.8 | 2377.5 | P0 |
| 11 | ordinary | 406.40 | 67.97 | 1.0072 | 607.2 | 6538.8 | 2400.0 | P0 |
| 12 | buffer4 | 324.25 | 69.30 | 1.0050 | 592.6 | 5497.5 | 2400.0 | P0 |
| 13 | ordinary | 400.22 | 67.81 | 1.0136 | 581.8 | 5420.8 | 2370.0 | P0 |
| 14 | ordinary | 400.22 | 68.07 | 1.0077 | 584.9 | 5410.9 | 2265.0 | P0 |
| 15 | buffer4 | 324.25 | 70.47 | 1.0118 | 576.0 | 5464.3 | 2385.0 | P0 |
| 16 | ordinary | 400.22 | 68.72 | 1.0216 | 577.6 | 5485.0 | 2340.0 | P0 |
| 17 | buffer4 | 324.25 | 69.72 | 1.0001 | 587.0 | 5524.8 | 2325.0 | P0 |
| 18 | buffer4 | 324.25 | 70.21 | 1.0009 | 572.0 | 5432.8 | 2370.0 | P0 |
| 19 | ordinary | 400.22 | 69.38 | 1.0205 | 583.1 | 5511.7 | 2400.0 | P0 |
| 20 | buffer4 | 324.25 | 70.67 | 1.0023 | 567.1 | 5475.2 | 2340.0 | P0 |
| 21 | ordinary | 400.22 | 69.29 | 1.0464 | 583.3 | 5486.7 | 2355.0 | P0,P3 |
| 22 | ordinary | 400.22 | 68.94 | 1.0156 | 576.2 | 5432.5 | 2265.0 | P0 |
| 23 | buffer4 | 324.25 | 70.50 | 1.0046 | 580.5 | 5513.5 | 2430.0 | P0,P3 |

Setup includes checkpoint/model/optimizer/data construction and initial hashes;
it is not isolated transfer time. Entire-segment wall time includes setup,
validation, saving, sampling and diagnostics; it excludes the following cleanup
boundary. `excluded_wall_ms` in the machine-readable result subtracts all30 update
wall times from that total. The primary training timer includes forward/backward,
clipping/Adam and candidate transfers/recomputation. First-step gradient copies
are in warmup only. No allocator inventories occur inside timed updates.
One model is resident at a time. Whole-job allocation includes setup, validation
and artifact operations, excludes driver/context/other processes; pinned host
memory is tracked separately and is not total CPU RSS. Reserved memory and all
phase inventories remain available in raw artifacts.

## Three-seed descriptive statistics

| Corpus | Ratio | Mean | Median | Sample variance |
|---|---|---:|---:|---:|
| wikitext2 | event | 1.028683 | 1.028895 | 6.10177e-06 |
| wikitext2 | wall | 1.028669 | 1.028865 | 6.20289e-06 |
| wikitext2 | memory | 0.813070 | 0.813070 | 0 |
| tinystories | event | 1.021533 | 1.022284 | 6.40184e-05 |
| tinystories | wall | 1.021514 | 1.022340 | 6.48264e-05 |
| tinystories | memory | 0.810184 | 0.810184 | 0 |

All24 segments execute in one process to avoid repeated process startup. Passive
200ms telemetry is matched only to actual timed update intervals; missing sensors
remain missing. Short temporal pairing reduces separation but does not guarantee
identical device conditions. Neither clocks nor recorded times are normalized.
No hardware settings changed. Segment/repeat stability gates retain all data.

## Evidence and limits

720 updates/backwards,2,949,120 targets,48 study full-validation scores plus24
independent native scores,72 tensor artifacts,1,608 phase records and50 zero
allocator boundaries. Independent NumPy checks verify clipping, Adam parameters
and moments from the first actual step; native evaluation replays all720 batches
and scores each final state. Raw/clipped gradients are checked against ordinary
repeat noise with the unchanged capped tolerances. No extra audit backwards.
All9,099,648 parameters, ordinary FP32/TF32off policy,8.125MiB workspace, original
AdamW/LR/batch/context/checkpointing and fourCPUthreads remain fixed. Python is
UV-managed. An AST audit proves the reused H134 loop differs only in fixture seed
metadata and setup-wall instrumentation. Frozen input/source/library/maintained
hashes and independent audit artifacts are checked before publication.

This measures short complete-update segments at trained states. It does not
replace the H138 convergence study, show sustained deployment throughput, or
validate another scale/domain. The earlier failures are preserved. Original
maintained defaults are unchanged.

## Next decision

Design an explicit opt-in maintained implementation with source-equivalence tests, preserving defaults, then qualify another workload scale. This pass concerns only balanced short segments at saved states; H138 remains failed and sustained deployment throughput remains unproven.

[Prospective plan](interleaved_training_plan.md),
[summary](../results/interleaved_training_v1/summary.json),
[receipt](../results/interleaved_training_v1/receipt.json).
