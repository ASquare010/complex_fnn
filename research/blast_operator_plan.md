# H079 - BLAST comparator: algebra, initialization and operator qualification

Freeze before implementation or numerical qualification. H078 is COMPLETE and
fails all four quality gates at 3,200 steps. The previous goal turn made PROGRESS.
Its fixed GELU recipe stays closed. This stage fills a published-comparator gap
using BLAST's distinct learned group couplings, not an activation/rate repair.
Keep the active three-folder/six-variant/nine-recipe tree unchanged.

## Hypothesis and scope

A BLAST projection has canonical blocks A_ij = U_i diag(s_ij) V_j^T.
We will implement this established factorization with three native batched matrix
products. The middle product batches over rank, avoiding a broadcast tensor with
both block axes and the entire token axis. This is an implementation choice, not
an architectural novelty or a measured speed/memory claim.

Use b=8, r=48, d=384. The two-projection GELU control has h=3200; the three-
projection SwiGLU control has h=1984. Each projection uses r(m+n+b^2) weights.
Both FFNs have 350,208 weights per layer, 2,801,664 over eight layers and
9,099,648 total with the unchanged 6,297,984 non-FFN weights. This matches the
retained compressed budget: 70.3125% FFN / 42.1700% total reduction. Counting a
full Transformer is arithmetic here, not an allocated Transformer execution.
Forward matrix MACs per token equal parameter count; multiply by two for FLOPs.

Canonical projections return contiguous output groups as in the BLAST equation.
The isolated FFN wrapper applies the retained output-unshuffle after every
projection to match BlockShuffle's outer permutations. Its activations are
unchanged GELU or SiLU(up)*gate. Document this local permutation/initialization
adaptation: it is not reproduction of the paper's complete training recipe.

## Local mathematics to verify

1. A canonical two-factor BlockShuffle block has rank at most t=384/8^2=6.
   Express it as L_ij R_ij^T. Map each rank-t component to BLAST latent slice
   ((i+j) mod b)*t through a Latin-square assignment; verify exact embedding at
   the SAME input/output dimensions. Each row/column receives disjoint slices.
   The additional r*b^2 weights cost 3,072 per projection. Reducing hidden width
   by 64 pays this cost. No containment at different equal-budget widths follows.
2. A padded H8 tensor I48 witness (H8 the unnormalized Sylvester Hadamard matrix)
   belongs to BLAST, has all 64 canonical blocks of rank 48 and 384 nonzero
   singular values sqrt(8). The squared relative Frobenius error of ANY fixed-
   partition rank-6-block approximation is at least 7/8, from discarding at least
   42 unit singular values in every block. This is a local matrix obstruction,
   not a nonlinear FFN, learning, task-quality or universal expressivity theorem.
3. Initialize row bases with orthonormal columns, column bases with orthonormal
   rows, and each rank's b-by-b coupling matrix orthogonally. Scale every factor
   by g^(1/3), where g=.02*sqrt(max(m,n))*residual_scale. With r=min(m,n)/b,
   the nonzero singular values are all g. Verify A^T A=g^2 I for up and
   A A^T=g^2 I for down, plus dense-equivalent mean row squared norm. Use residual
   scale 1 for up/gate and 1/4 for down. This is OUR local calibration, not an
   author-reported initialization or a property preserved by AdamW training.
   All-ones couplings instead have global rank at most r; verify exact collapse.

These properties concern fixed canonical partitions and single projections.
BLAST/Monarch containment is established prior art; our explicit mapping and
certificate are an auditable implementation derivation, not novelty priority.
An initial partial isometry does not guarantee full FFN gradients or stability
through attention, activations, depth or optimization. Down has a nullspace.

## Fixed numerical allocation: 34 checks, zero optimizer updates

- One shape/invalid-configuration/parameter-count check for both FFNs.
- Twelve FP64 CPU projection checks: both directions at h1984/h3200 and seeds
  17/29/43. Compare outputs and input/three-factor gradients with independent
  blockwise dense reconstruction at rtol=1e-10, atol=1e-11. Verify deterministic
  name-local initialization, finite nonzero norms, partial-isometry Gram at
  rtol=1e-10/atol=1e-11 and mean row squared norm at relative tolerance 1e-10.
  CPU probe shape [2,7,n]. Save both sides and factors as tensor evidence.
- One small FP64 finite-difference gradcheck at n8/m12/b2/r3, fast mode, with
  eps=1e-6, atol=1e-5 and rtol=1e-3. This derivative check need not meet the
  full-rank initialization condition; it checks generic factor values.
- Twelve whole-FFN eager/checkpoint comparisons: two forms, seeds17/29/43,
  CPU FP32 [2,7,384] and CUDA BF16 [16,128,384] with FP32 parameters, TF32 off,
  four CPU threads. Nonreentrant checkpoint with RNG preservation. Require
  bitwise equal finite outputs/input/all-factor gradients, nonzero norms;
  GELU supplies 8 tensor comparisons per case, SwiGLU 11: total 114 comparisons.
- Four exact integer-valued FP64 BlockShuffle embeddings, one per projection
  direction/width, seed17. Require dense represented matrices and integer-input
  outputs exactly equal after accounting for the outer output permutation.
- Four Hadamard witnesses, one per direction/width. Verify integer Gram,
  all canonical block minors, exact all-ones rank collapse, rational 7/8 bound.

There are no optimizer steps, corpus targets, language workers, Transformers,
compiler runs, time/VRAM benchmarks, rate selection or learning claims. All CUDA
work is in one bounded qualification process. Save factors, both numerical
paths, inputs/gradients and JSON measurements. Finite-difference forward calls
are counted, not described as training. No failure-triggered retry, looser
threshold, implementation change or extra allocation is automatic.

## Qualification decision and preservation

All 34 checks and independent tensor/certificate verification must pass. Passing
earns only a separately frozen learning comparison with full, calibrated narrow
and plain controls and equal tuning effort. Optimizer calibration, full-model
trajectory fidelity/resources and language quality remain unqualified. A failed
check stops this attempt and preserves its partial files. No active integration.

Pin the completed H078 final/result/source/support evidence, its 112 scientific
sources, 67 existing frozen plans and the three current navigation documents.
Add only three isolated scientific sources and a separately hashed theory note.
Record UV environment, git provenance, PID/UTC/log/return code and source hashes.
Independent analysis verifies all stored tensor pairs, initialization/certificates,
counts, outcomes and source archives. Preserve all previous fitting/resource/
language evidence and the prior 108-test active suite without unrelated reruns.
Update the report, README, current state and ledger; no breakthrough or research-
goal completion follows from local qualification.

Primary sources read 2026-09-08:
[BLAST factor definition, rectangular footnote, and Appendix A](https://arxiv.org/html/2410.21262v1),
[official repository](https://github.com/changwoolee/BLAST).
See the [prior count note](ungated_comparator_note.md) and
[H078 result](ungated_duration_results.md).
