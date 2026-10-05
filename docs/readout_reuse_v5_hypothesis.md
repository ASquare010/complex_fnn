# Readout reuse execution v5: native scalar backward

Registered after v4's complete-model gradient failure and saved-fixture trace,
before v5 implementation, profiling or training. Preserve v4 sources and evidence.

Keep v4's FFN mathematics, projected-z storage, native linear cast sites, native
group forward/reconstruction/adjoints, and both shared-projection gradient terms.
Replace only its custom scalar gradient kernel with native Torch operations:
apply fixed signs, apply the even-feature removal mask if enabled, multiply by
the response scale, then add `g * silu(z)` and `aten.silu_backward(g * z, z)`.
The installed native activation backward avoids relying on an equivalent-looking
CUDA expression to reproduce its rounding. Retain the already checked scalar
forward kernel; remove unused scalar-gradient dispatch. Include all additional
intermediate tensors and launches in memory/time measurements.

No new architectural or originality claim: this is an execution change for the
original registered readout-reuse hypothesis. Parameter and projection counts,
backbone, initialization, controls and all acceptance thresholds are unchanged.
The extra native elementwise launches may make v5 too slow; a precision pass is
not an efficiency pass.

Repeat the full CPU suite and all 192 GPU FFN fixtures, 24 whole-model comparisons,
raw grouped adjoints, native BF16 casts and preserved v2/v3 failure regressions.
Also include the actual v4 failed model as a third regression. Inspect previously
differing projected-adjoint casts on the preserved v3 fixture. Original tolerances:
FP32 absolute 1e-5 / relative 3e-4; BF16 outputs .004/.02; gradients .0001/.05.
Any failure retires this execution revision before hardware/language work.

Only a terminal complete numerical pass permits the original 54-profile screen:
nine variants, two corpora, three alternating rounds, 20 warmup and 100 measured
updates. No overlapping GPU experiments or modifications of frozen sources.
Require every original resource comparison: at least 1.2x throughput or 20%
allocated-memory reduction, with no more than 5% worsening of the other resource.
Only a resource survivor proceeds to production equivalence and the original
24-run language screen, unchanged quality/removal/confirmation gates.
