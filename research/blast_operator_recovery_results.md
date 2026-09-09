# H080 - BLAST operator recovery and qualification

**LOCALLY QUALIFIED; learning remains untested.** All 34 frozen checks pass in
the one explicit recovery. All 32 tensor files verify independently. No
optimizer update, corpus target or full Transformer run was allocated.

BLAST is published prior art. Our comparator uses rank-batched native group
mixing, a local orthogonal initialization and the retained BlockShuffle outer
permutations. This does not reproduce the paper's complete training recipe or
establish a new architecture. [BLAST source](https://arxiv.org/html/2410.21262v1)

| FFN | Hidden width | Projections | FFN weights, 8 layers | Total weights by count |
|---|---:|---:|---:|---:|
| BLAST GELU | 3,200 | 2 | 2,801,664 | 9,099,648 |
| BLAST SwiGLU | 1,984 | 3 | 2,801,664 | 9,099,648 |

Both have 350,208 weights per FFN: 70.3125% FFN and 42.1700% total reduction.
Transformer totals are arithmetic, not executed models. No training time,
memory or quality claim follows.

## Numerical and mathematical evidence

| Check | Verified result |
|---|---|
| Frozen functions, seeds, shapes, precision and thresholds | Unchanged |
| FP64 projection comparisons | 12 cases; 60 output/gradient comparisons |
| Maximum projection comparison error | 1.577e-14 |
| Independently derived gradient maximum error | 8.882e-15 |
| Eager/checkpoint comparisons | 6 CPU FP32 + 6 CUDA BF16 cases; all 114 tensor pairs bitwise equal |
| Independent initializer Gram maximum error | 1.998e-15 |
| Finite differences | PASS; 10 forward calls, zero updates |
| Exact same-width BlockShuffle embeddings | Four projection shapes |
| Hadamard/rank-collapse certificates | Four projection shapes |
| Readable original tensor payloads | All 24 match the recovery exactly |

The [theory](blast_operator_theory.md) proves same-width inclusion with an
explicit Latin-square mapping, costing 3,072 extra coupling weights/projection.
Equal-budget FFNs reduce hidden width by 64, so this does not prove equal-budget
nonlinear-family inclusion. The fixed-partition rank-6 matrix approximation
has squared relative error at least 7/8 on the Hadamard witness. Neither statement
is a learning result or novelty claim.

The initial projections have controlled nonzero singular values. Down retains
a nullspace; activations, learned factors and depth can still have small
Jacobians. No full-network nonvanishing-gradient or convergence guarantee follows.

## Preserved failures and recovery scope

[H079](blast_operator_results.md) records a process pass but incomplete saved
evidence: 25 entirely zero-filled files, 18/34 parseable observation JSONs and
24/32 intact tensor archives. Cause is unknown; original bytes remain unchanged.

H080 repeats the same checks once in a new root. Only the artifact destination
and writer change: flush, fsync, ZIP/hash checks and exact payload readback.
This verifies present artifacts without diagnosing or guaranteeing removal of
the original cause. There are two qualification attempts across H079/H080.
The recovery process takes 15.680 seconds; pytest reports 14.26 seconds.

Audit v1 incorrectly required a CPU-recomputed norm to equal the recorded GPU
norm bitwise. Seventeen scalars differ by at most 6.651e-16 relatively. Audit v2
checks each norm on its original device, retaining exact equality. All numerical
thresholds remain unchanged; its 57 CUDA reductions are metadata checks with
zero model forwards. The failed audit and a pre-execution report-writer syntax
error are preserved separately; neither triggers a qualification/training repeat.

Only a separately frozen learning comparison is earned, with explicit optimizer
calibration and equal tuning against full, calibrated narrow and plain controls.
Full-model resources, language quality, long-budget seeds, convergence, scale,
broader data and published comparisons remain open. H078 stays closed. Active
source remains three model folders, six variants, nine recipes and the unchanged
previously passing 108-test suite.

[Recovery plan](blast_operator_recovery_plan.md),
[measurements](../results/blast_operator_recovery_v1/result.json),
[independent audit](../results/verification/blast_operator_recovery_analysis_v2.json),
[correction record](../results/blast_operator_recovery_v1/analysis_correction.json),
[final preservation audit](../results/verification/blast_operator_recovery_final_v1.json).
