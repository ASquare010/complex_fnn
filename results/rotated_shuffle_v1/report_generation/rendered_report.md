# H070 - Rotated shuffle qualification

**LOCALLY QUALIFIED.** One learned orthogonal stage expands the retained linear
projection family at low parameter cost. All 21 fixed checks pass, including
baseline-preserving initialization and a same-cost redundant control. No fitted
quality, full-model resource result or breakthrough is claimed.

| Form | Extra weights, d384/L8 | FFN weights | Total model weights | FFN reduction |
|---|---:|---:|---:|---:|
| Plain BlockShuffle | 0 | 2,801,664 | 9,099,648 | 70.3125% |
| Same-origin rotation, shift0 | 4,608 | 2,806,272 | 9,104,256 | 70.2637% |
| Cross-origin rotation, shift1 | 4,608 | 2,806,272 | 9,104,256 | 70.2637% |

The extension places one disjoint Givens stage between the fixed intermediate
shuffle and the second factor in each up/value/down map. Every angle starts at
zero. The candidate pairs different input origins and output groups; the control
pairs the same input origin across output groups. They have equal angle counts
and arithmetic. Cosine/sine coefficients are cast to the activation dtype before
vector operations; actual memory and runtime still require measurement.

## What changes mathematically

In canonical output coordinates, each original d384/G8 projection block has rank
at most6. The new stage includes the baseline at zero angles and has an explicit
rank7 block construction. For that fixed matrix witness, every original projection
has Frobenius error at least 1/sqrt(2), or **1/sqrt(13) = 27.7350% relative error**.
Independent reconstruction from saved factors matches the recorded matrix within
1.11e-16 maximum absolute error. The bound follows from the rank6 truncated-SVD
obstruction; it is not a measured language loss or nonlinear FFN approximation
bound. A more expressive projection does not by itself prove strict full-FFN
separation or a learning advantage.

The shift0 control can be absorbed exactly in real arithmetic into the existing
first factor, adding optimization coordinates without expanding the projection
family. Its independent FP64 absorption check has 2.78e-17 maximum matrix error.
This control is needed to distinguish a structural benefit from reparameterizing
the same functions during later training.

The rotation itself has an orthogonal input Jacobian. FP64 matrix error is
2.22e-16; sampled BF16 maximum relative norm changes are 0.0891% forward and
0.1024% backward across the two forms. This does not bound the complete FFN or
Transformer Jacobian: surrounding factors, nullspaces and nonlinearities remain.

![Routing counts and projection witness](figures/rotated_shuffle.png)

## Fixed checks and independent verification

- Twelve zero-angle cases: two forms, CPU FP32/CUDA BF16 and three seeds.
  All 96 output/input/shared-factor gradient comparisons are bitwise exact.
  All 36 new angle-gradient vectors have finite, nonzero norms.
- Four nonzero-angle cases: eager versus standard product checkpointing.
  All 44 output/input/all-parameter gradient comparisons are bitwise exact.
- Five mathematical/software checks: counts/topology, full-size rank witness,
  redundant-control absorption, independent FP64 gradcheck and component isometry.

Both local references use standard non-reentrant product checkpointing; this is
not an assertion that the complete historical native-custom-backward training
trajectory is reproduced. CPU FFN probes use d64/h128/G8 and [2,7,64] inputs;
CUDA probes use d384/h2048/G8 and [16,128,384]. The actual full-size sparse rank
witness is separately stored. Four CPU threads and TF32-disabled CUDA are used.

The single process finishes first attempt: 21 passed in3.97s, 5.16s including
process overhead. Independent inspection reloads every paired raw tensor,
recounts paths and reconstructs the witness from block factors and an explicit
rotation matrix. All 140 comparisons and 36 angle-gradient signals pass.
There are zero optimizer updates, corpus targets or full-model resource workers.


Report rendering encountered a separate CPython `tok_backup` failure while
importing Torch through a hash helper, after scientific qualification and the
independent tensor audit had passed. The cause is unknown. The failed report
source and observed truncated exception are preserved in
[report-generation records](../results/rotated_shuffle_v1/report_generation/).
Replacing that unnecessary report-only dependency with standard-library hashing
allowed rendering to finish. No scientific check or experiment was rerun.

## Correction to the small fitting interpretation

H068's k48/G8 setup preserves the main model's relative widths but has only48
latent paths across64 canonical block pairs: 16 blocks are identically zero,
with the remaining48 rank-bounded by1. At k384/G8 all64 blocks have rank bound6.
The new topology check verifies this directly. H068 remains a valid negative
result for its frozen setting; it was not a reproduction of full-size block
connectivity. Its saved data, source, result, gates and rejection are unchanged.
This caveat does not automatically reopen either rejected activation recipe.

## Prior art and earned next step

The block-low-rank interpretation and more general structured orthogonal products
are established in [Group and Shuffle](https://arxiv.org/html/2406.10019v1#S3).
[Kaleidoscope](https://arxiv.org/abs/2012.14966) learns structured linear maps;
[ButterflyQuant](https://arxiv.org/abs/2509.09679) uses learnable Givens butterfly
angles for quantization. The present proposal is a small use of known components
inside the retained FFN, with an explicit redundant control and scoped witness.
Neither the components nor these elementary constructions establish novelty.

Passing earns a separately frozen fitting/resource comparison against plain,
shift0, calibrated narrow and both full controls. It does not earn language
training. Future fitting must state the actual block routing/rank pattern, in
addition to width ratios. Adequate learning, resource behavior, longer three-seed
qualification, convergence, scale and broader data remain unproven.
The prototype stays beside its evidence; no active model or recipe is added.
H068/H069 and earlier rejected branches remain closed under their tested rules.

- [Frozen plan](rotated_shuffle_plan.md) and [scoped theory](rotated_shuffle_theory.md).
- [Raw result](../results/rotated_shuffle_v1/result.json) and [process](../results/rotated_shuffle_v1/process.json).
- [Independent tensor/matrix audit](../results/verification/rotated_shuffle_analysis_v1.json).
- [Final preservation audit](../results/verification/rotated_shuffle_final_v1.json).

Result SHA-256: `8b241b978d4b52475efbae2109cbb39c355975a58534e738ea37fa74a776d7ec`.
Protocol SHA-256: `235ca1df1696a154e2f5122c5f0c3c1765632bee442675b5ee74dde89d9b6c2b`.
