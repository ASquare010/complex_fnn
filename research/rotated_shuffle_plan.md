# H070 - Qualify one rotated shuffle and its absorbable control

Freeze before outcomes. H068 activation fitting and H069 balancing motivation
remain rejected. Test one distinct structural extension, with no active model,
factory, recipe or optimizer change. See the [scoped theory](rotated_shuffle_theory.md).

## Definition and fixed counts

Replace each BlockShuffleLinear by the same two factors with one disjoint Givens
rotation stage between its existing shuffle and second factor. Pair first-half
index p with second-half index (p+shift) mod half_width. Candidate shift1 crosses
both factor groupings; shift0 is absorbable into the first factor and is a
same-cost optimization-coordinate control. Angles start exactly zero. No other
mixing stage, angle constraint, amplitude, rate or alternative pairing is tested.

Angles are FP32 model parameters; cosine/sine coefficients are cast to the
projected activation dtype BEFORE vector arithmetic. Double inputs/parameters
retain FP64 for mathematical checks. Do not force a full FP32 activation bank.
The added cost is4,608 weights at d384/L8: FFN2,806,272, total9,104,256 and
70.263671875% FFN reduction versus9,437,184 full FFN weights. Both forms must
satisfy these exact counts. No optimizer update or corpus/resource worker occurs.

## Fixed 21-check qualification

1. Twelve zero-angle cases: shift0/1 x CPU FP32 or CUDA BF16 x seeds17/29/43.
   CPU shape[2,7,64], h128/G8; GPU[16,128,384], h2048/G8. Use name-derived
   existing initialization, down residual scale.25. Candidate/control copy plain
   projection weights exactly. Synthetic CPU input/incoming-gradient generator
   seed93000+training_seed, transferred to GPU where needed. Both baseline and
   extension use standard non-reentrant product checkpointing (not a claim about
   the historical native custom backward's complete training trajectory). Require
   output, input gradient and all six shared factor gradients bitwise equal with
   finite/nonzero norm for each of the three new angle-gradient vectors.
2. Four nonzero cases: shift0/1 x CPU/GPU, seed17. Set each projection's angles to
   linspace(-.3,.3,k/2). Compare eager versus standard product-checkpoint output,
   input and all nine parameter gradients bitwise, all finite. Same shapes/inputs.
3. Five mathematical/software checks: counts plus routing counts at k48/64/384;
   the full-size rank7 witness and rank6 Frobenius obstruction; an explicit FP64
   absorption of shift0 into A; independent FP64 gradcheck for the new component
   at both shifts; orthogonality/input-gradient isometry in FP64 and sampled
   CUDA BF16 norm distortion <=1% for both shifts at full width.

For isometry: FP64384x384 matrix orthogonality max error<=1e-13. CPU random input
and incoming gradient shape[16,128,384], seed94017. Set angles linspace(-pi,pi,192),
then compare per-row input/output and incoming/input-gradient norms: FP64 relative
error<=1e-12, CUDA BF16<=.01. This is a component bound, not an FFN Jacobian bound.
The full-size witness uses the exact sparse indices in the theory, theta0=pi/4.
Require canonical block singular values(1 six times,1/sqrt2,then zeros) within
1e-12 absolute/relative; save matrices, factors and independent calculations.

Four CPU threads, TF32 disabled, UV's existing compile/data extras, one GPU
process under a hidden coordinator. Save input/state hashes, every case's paired
raw tensors and metrics, source/plan/theory snapshot, PID/logs and return code.
Use the already qualified raw-byte hash helper for BF16. No automatic retry,
source mutation or tolerance change after scientific outcomes. All21 checks
must pass. Existing source/config/test/checkpoint bytes must match H069.

## Decision and documentation

Passing earns only a separately frozen fitting/resource comparison against plain,
shift0, calibrated narrow and both full controls. It does not earn language
training, prove a training benefit or finish the gold target. Failure closes
this definition without another pairing, stage or dtype workaround automatically.
Keep the prototype only beside its evidence. Document H068's small-size block
connectivity limitation without changing that frozen plan, result or rejection.
Novelty, actual memory/runtime, matched learning, longer three-seed qualification,
convergence, scale and broader data remain unresolved.
