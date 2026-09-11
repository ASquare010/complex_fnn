# H142: doubled-batch maintained-helper qualification

**PASS scoped batch-16 qualification gate.** Independent numerical/native audit: **True**.
H138's failed long-training qualification remains unchanged. No new activation,
parameter reduction, SOTA or broad-goal achievement is claimed.

| Corpus | Source seed | GPU allocation saved | Complete CUDA ratio | Wall ratio | Failed gates |
|---|---:|---:|---:|---:|---|
| wikitext2 | 101 | 23.77% | 1.0461 | 1.0461 | none |
| wikitext2 | 113 | 23.77% | 1.0482 | 1.0482 | none |
| wikitext2 | 127 | 23.77% | 1.0538 | 1.0538 | none |
| tinystories | 101 | 23.99% | 1.0489 | 1.0488 | none |
| tinystories | 113 | 23.99% | 1.0469 | 1.0468 | none |
| tinystories | 127 | 23.99% | 1.0505 | 1.0505 | none |

All six ordinary H138 step800 model/Adam/sampler states are used. Training batch
size is16 instead of8; context512 and validation batch8 remain unchanged.
The candidate uses the maintained buffer_model_loss and offload_checkpoint_inputs
APIs without editing their source. This tests a larger activation workload, not
a larger model or a different domain. For each, four
independent30-update segments reset to that exact state, in ABBA or BAAB order
alternating by fixture. A=ordinary, B=buffer4. These are repeated continuations,
not120 updates of a continuous trajectory. Twenty updates per segment remain
after ten fixed warmups. Primary ratios compare all40 timed updates per arm.
Every seed must pass; descriptive aggregate means cannot rescue a failure.

## Neighboring temporal pairs

| Corpus | Source seed | Repeat | CUDA ratio | Wall ratio | Final native NLL ratio |
|---|---:|---:|---:|---:|---:|
| wikitext2 | 101 | 0 | 1.0484 | 1.0484 | 0.999999 |
| wikitext2 | 101 | 1 | 1.0432 | 1.0432 | 0.999997 |
| wikitext2 | 113 | 0 | 1.0445 | 1.0446 | 1.000000 |
| wikitext2 | 113 | 1 | 1.0506 | 1.0505 | 0.999999 |
| wikitext2 | 127 | 0 | 1.0577 | 1.0577 | 1.000001 |
| wikitext2 | 127 | 1 | 1.0413 | 1.0413 | 1.000000 |
| tinystories | 101 | 0 | 1.0437 | 1.0437 | 1.000003 |
| tinystories | 101 | 1 | 1.0491 | 1.0490 | 1.000018 |
| tinystories | 113 | 0 | 1.0478 | 1.0478 | 1.000009 |
| tinystories | 113 | 1 | 1.0303 | 1.0303 | 1.000020 |
| tinystories | 127 | 0 | 1.0480 | 1.0480 | 1.000001 |
| tinystories | 127 | 1 | 1.0504 | 1.0504 | 0.999996 |

The frozen gates are peak allocated memory<=0.90, complete CUDA/wall time<=1.15,
both repeat NLL ratios<=1.01, pinned hostpeak<=128MiB, every segment's half-median
stability<=1.15 and each arm's two-repeat median ratio<=1.15. Telemetry coverage
and numerical/native-score/initial-hash/batch-order checks are mandatory.

## Per-segment resources and setup

| Segment | Arm | GPU peak MiB | Complete CUDA ms | Within-segment stability | Setup ms | Entire segment ms | Median active SM MHz | P-states |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | ordinary | 664.45 | 130.60 | 1.0050 | 569.0 | 8904.5 | 2430.0 | P0 |
| 1 | buffer4 | 506.51 | 136.92 | 1.0032 | 587.4 | 8514.8 | 2475.0 | P0 |
| 2 | buffer4 | 506.51 | 137.00 | 1.0046 | 583.4 | 8534.9 | 2490.0 | P0 |
| 3 | ordinary | 664.45 | 131.33 | 1.0077 | 553.4 | 8287.2 | 2445.0 | P0 |
| 4 | buffer4 | 506.51 | 137.63 | 1.0045 | 596.2 | 8569.1 | 2415.0 | P0 |
| 5 | ordinary | 664.45 | 131.76 | 1.0064 | 581.1 | 8384.7 | 2430.0 | P0 |
| 6 | ordinary | 664.45 | 132.11 | 1.0053 | 588.0 | 8398.3 | 2445.0 | P0 |
| 7 | buffer4 | 506.51 | 138.79 | 1.0008 | 584.6 | 8576.8 | 2460.0 | P0 |
| 8 | ordinary | 664.45 | 134.65 | 1.0032 | 596.9 | 8489.0 | 2340.0 | P0 |
| 9 | buffer4 | 506.51 | 142.42 | 1.0081 | 589.8 | 8681.8 | 2317.5 | P0 |
| 10 | buffer4 | 506.51 | 142.42 | 1.0023 | 576.6 | 8644.4 | 2265.0 | P0 |
| 11 | ordinary | 664.45 | 136.77 | 1.0210 | 568.1 | 8500.7 | 2242.5 | P0 |
| 12 | buffer4 | 500.33 | 144.13 | 1.0154 | 562.4 | 7641.2 | 2205.0 | P0 |
| 13 | ordinary | 658.27 | 138.09 | 1.0102 | 562.7 | 7437.0 | 2190.0 | P0 |
| 14 | ordinary | 658.27 | 139.25 | 1.0008 | 579.8 | 7479.4 | 2175.0 | P0 |
| 15 | buffer4 | 500.33 | 146.08 | 1.0030 | 575.6 | 7638.0 | 2175.0 | P0 |
| 16 | ordinary | 658.27 | 139.79 | 1.0088 | 585.3 | 7587.2 | 2235.0 | P0 |
| 17 | buffer4 | 500.33 | 146.48 | 1.0012 | 582.2 | 7671.7 | 2160.0 | P0 |
| 18 | buffer4 | 500.33 | 146.74 | 1.0123 | 576.9 | 7758.1 | 2145.0 | P0 |
| 19 | ordinary | 658.27 | 142.43 | 1.0064 | 566.4 | 7586.2 | 2025.0 | P0 |
| 20 | buffer4 | 500.33 | 148.75 | 1.0127 | 595.9 | 7826.0 | 2040.0 | P0 |
| 21 | ordinary | 658.27 | 141.95 | 1.0081 | 571.9 | 7578.1 | 2040.0 | P0 |
| 22 | ordinary | 658.27 | 142.23 | 1.0306 | 546.7 | 7490.5 | 1935.0 | P0 |
| 23 | buffer4 | 500.33 | 149.40 | 1.0022 | 572.6 | 7744.3 | 2032.5 | P0 |

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
| wikitext2 | event | 1.049377 | 1.048233 | 1.58604e-05 |
| wikitext2 | wall | 1.049377 | 1.048210 | 1.57459e-05 |
| wikitext2 | memory | 0.762303 | 0.762303 | 0 |
| tinystories | event | 1.048753 | 1.048856 | 3.38754e-06 |
| tinystories | wall | 1.048722 | 1.048802 | 3.39575e-06 |
| tinystories | memory | 0.760071 | 0.760071 | 0 |

All24 segments execute in one process to avoid repeated process startup. Passive
200ms telemetry is matched only to actual timed update intervals; missing sensors
remain missing. Short temporal pairing reduces separation but does not guarantee
identical device conditions. Neither clocks nor recorded times are normalized.
No hardware settings changed. Segment/repeat stability gates retain all data.

## Evidence and limits

720 updates/backwards,5,898,240 targets,48 study full-validation scores plus24
independent native scores,72 tensor artifacts,1,608 phase records and50 zero
allocator boundaries. Independent NumPy checks verify clipping, Adam parameters
and moments from the first actual step; native evaluation replays all720 batches
and scores each final state. Raw/clipped gradients are checked against ordinary
repeat noise with the unchanged capped tolerances. No extra audit backwards.
All9,099,648 parameters, ordinary FP32/TF32off policy,8.125MiB workspace, original
AdamW/LR/context/checkpointing and four CPU threads are shared by both arms;
training uses batch 16 and validation batch 8. Python is
UV-managed. AST audits prove the reused H140 loop differs only in training batch size and
target accounting; the independent H120 audit differs only in sampled batch size.
Native validation and numerical Adam/clipping checks remain unchanged. Frozen input/source/library/maintained
hashes and independent audit artifacts are checked before publication.

This measures short complete-update segments at trained states. It does not
replace the H138 convergence study, show sustained deployment throughput, or
validate model-size or domain generality beyond this doubled-batch workload.
The earlier failures are preserved. Original
maintained defaults are unchanged.

## Next decision

The maintained helper now has bounded evidence at batch8 and batch16. Preserve ordinary defaults and H138's failed long-run result. Return to the unresolved structural FFN/parameter-efficiency question using the repository's prior candidate eliminations; do not infer novel neuron geometry or sustained throughput from these memory-only results.

[Prospective plan](batch_scale_training_plan.md),
[summary](../results/batch_scale_training_v1/summary.json),
[receipt](../results/batch_scale_training_v1/receipt.json).
