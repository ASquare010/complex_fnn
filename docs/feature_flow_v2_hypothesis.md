# Feature flow execution revision 2: fuse the grouped recurrence

Registered before implementation/measurement of this revision. Keep the
[mathematical hypothesis](feature_flow_hypothesis.md), counts, initialization,
three steps, controls and all acceptance thresholds unchanged. Version 1's
48-profile screen is terminal and failed resources: roughly 398 MiB allocated
but about 50% longer update time than full dense. No language training occurred.
Its source, checks and profiles remain frozen; do not overwrite the retired recipe.

Execution hypothesis: the repeated small FP32 group GEMMs, group-major/token-major
copies, reaction tensors, separately dispatched updates and backward matrix
reductions are costly at group size 32. Fuse the actual small group operation,
rather than changing the number of steps or switching to an easier control.
The screen did not isolate which cost dominates, so this is a testable explanation,
not a measured attribution.

Use one CUDA warp per token/group for sizes <=32. Each lane computes its scalar
bounded reaction once, broadcasts it with warp shuffles, and accumulates its
output row's exact dense group dot product in FP32. Fuse the residual update.
Transpose the small master M into a contiguous forward-read layout once per
inner call, for coalesced weight reads; include that copy in timings. Input
backward uses original M layout, broadcasts incoming state derivatives, and
fuses M.T multiplication, reaction derivative and identity derivative.

Matrix gradients use deterministic 32-token tiles: eight token lanes by 32
output channels, four tokens per lane. Broadcast each input reaction across
the output lanes, accumulate each output/input coefficient, and reduce token
lanes with shared memory. One FP32 partial buffer [groups,ceil(tokens/32),size,size]
is accumulated across reversed steps, then reduced by Torch over token tiles.
No atomic accumulation, no token-sized derivative bank. Include this temporary
buffer and all reconstructed states in peak measurements. Preserve Torch-owned
memory, the current Torch stream and borrowed existing Torch context.

No early return by individual warp lanes before shuffles; invalid channels must
still participate with zeros. Token tails and coefficient reductions need full
barrier participation. Validate all raw shapes/dtypes/layouts and int32/grid
limits before launch. Master M remains FP32; states/accumulators FP32, no TF32,
FMA fusion or fast-math selection. P/Q remain BF16 AMP; final activation/input
adjoint use native Torch casts once. Sum order can differ from cuBLAS; this is
the same mathematical operation up to recorded floating-point differences.
For groups >32 retain the checked ordinary grouped path and report the fallback;
the registered full-width profiles all use groups of exactly 32.

Implement in separate staged version-2 files. Independent CPU FP64 equations,
all adjoints, finite differences, counts/calibration and five complete-model
causality/locality/save-load/exact optimizer recovery must pass again. CUDA
FP32/BF16 checks include groups 1/8/32 and explicit >32 fallback, incomplete token
tiles, nonzero mixing matrices, all removals and initialized controls. Add actual
2048-token gradient checks. Compare output/input FP32 at atol 1e-6, rtol 2e-4;
matrix-reduction gradients at atol 1e-5, rtol 3e-4. BF16 cast paths allow recorded
rounding tolerance atol .00025, rtol .025 for output/input, while FP32 matrix
gradients retain their stricter reduction tolerances. Full-model BF16 checks
retain atol .004/rtol .02 outputs and .0001/.05 parameter gradients. Record actual
errors; do not loosen criteria using language outcomes.

Run the same eight variants, 48 profiles, three alternating-order rounds per
corpus, 20 warmup/100 timed updates and unchanged primary resource gates. Cached
execution stays ordinary autograd. Only a passing primary may be integrated
and enter the previously registered eleven-variant two-corpus language screen.
No source changes or additional GPU work overlap a live campaign; no relaxing
the slowdown limit or unregistered precision change. Retire another resource
failure before language; finite resource improvement is not a quality result.

Warp-level small matrix products, tiled reductions and recomputation are
established implementation techniques, not a new FFN theory. This revision's
only claim to test is faster execution of the same registered recurrence.
