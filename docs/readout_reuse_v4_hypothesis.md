# Readout reuse execution v4: native group adjoints, ordered scalar backward

Registered after actual v3 failed its complete-model gradient check twice and
the [saved-fixture adjoint trace](readout_reuse_v3_result.md) completed. Retire
v3 before resource/language work. Preserve all prior sources, failed fixtures
and actual/preparation records. Original FFN mathematics, allocation, controls,
initialization, signs, calibration and acceptance gates remain unchanged.

Keep projected-z storage, native linear cast sites and both shared-P master
gradient contributions. Keep v3's scalar forward kernel and native grouped
forward/reconstruction. Replace fused grouped input/correction adjoints with
native Torch grouped BMM adjoints: g_r = g_u*C and grad_C = g_u.T*r within each
group, then apply the same gradient masks through the original C construction.
The diagonal control retains assembled group work; fixed/untied modes retain
their signed readout. Native group backward adds temporary banks/transposes;
it may fail the memory/time gates even if precision is repaired.

The scalar backward must preserve ordinary operation order instead of folding
scale and derivative into one expression. For the registered computation
scale*(z*SiLU(z)-mean), backpropagate gain through any fixed signs first, then
through scale, calculate the direct z-product contribution, calculate the
SiLU input contribution with native SiLU-backward ordering, and sum the two.
Check the local derivative and its BF16 projection cast against ordinary
autograd at the preserved fixture and registered full/small shapes before
assuming an equivalent real-valued formula is equivalent in floating point.
No tolerance relaxation, detached branch, straight-through approximation,
TF32, fast math, FMA fusion or omitted projection contribution.

Reuse Torch's context/current stream and native BF16 casts. Raw scalar kernels
accept only contiguous CUDA FP32 tensors and validated layouts/devices/counts.
No raw grouped matrix kernels are needed in this revision. Unsupported dtype
or group size>32 uses the explicit original ordinary-checkpoint fallback.
No all-layer FP32 response bank is retained; response/readout and their group
adjoints are reconstructed within one backward scope. Include all projected
z/ones buffers, FP32 banks, transposes, casts and dispatch in measured resources.

Training projection accounting remains 29,491,200 FLOPs/token plus scalar/
reduction work, four group products including reconstruction and six linear
products. Parameter and forward counts are unchanged. This is an execution
hypothesis; no speed, memory, language or novelty claim follows from its formula.

Separate v4 prototype/scalar-kernel/check/profile files. Repeat actual v3 CPU
checks and the complete numerical suite with correctly verified module/source
hashes: 192 FFN fixtures, 24 whole models, nonzero corrections, all removals,
incomplete rows, group sizes 2/8/32 and explicit size-34 fallback, native BF16
edgecases. Require both preserved v2/v3 failing models to pass complete output
and all-parameter-gradient comparisons against ordinary execution. Preserve
original tolerances: FP32 1e-5/3e-4; BF16 output .004/.02 and gradients
.0001/.05. Verify the previously different projected-adjoint cast cases as
well, without substituting that narrower check for whole-model validation.

Only terminal complete numerical passes permit source freezing and the original
54-profile screen. Three rounds, alternating order, all nine variants, both
corpora, 20 warmup/100 measured updates, unchanged backbone/data/optimizer.
No overlapping GPU work or live source edits. Primary must pass every original
resource comparison before production equivalence and the original 24-run
language screen. Any failure is retired; no control substitution or gate change.
