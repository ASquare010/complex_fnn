# Feature flow execution revision 4: reuse reconstructed reactions in backward

Registered after version 3 completed and failed TinyStories versus GELU,
before any implementation or measurement of this revision. Preserve the
[original mathematical hypothesis](feature_flow_hypothesis.md), all counts,
initialization, three internal steps, controls and resource/quality gates.
Version 3 passes five comparisons but needs 7.3% longer updates than TinyStories
GELU. No feature-flow language training has occurred.

Hypothesis: version 3 recomputes each reconstructed state's bounded reaction
and derivative in the reverse pass, repeating sigmoid/exponential/tanh work.
Instead cache the bounded reaction and its analytic derivative as lane-local
register values during forward reconstruction inside the backward kernel.
Reuse those values for matrix-gradient coefficients and the input adjoint.
This preserves the discrete adjoint and FP32 operation order of the reaction;
it changes which intermediates are held in registers. More register pressure
may offset the removed nonlinear evaluations; no dominant cost is established.

Keep version 3's forward kernel, transposed M layout, warp grouping,
deterministic 32-token matrix-gradient tiles and native Torch FP32/BF16 casts.
Backward reconstruction computes psi(h_t) and psi'(h_t) together, applies the
same dense group update, and keeps three pairs of scalar responses/derivatives
per channel. The reverse pass reads those pairs instead of recovering them
from saved h_t. No global token-sized reaction bank, atomics, approximate
activation, fast math, TF32, FMA fusion, smaller group or fewer steps. Raw master
matrices/buffers remain FP32. Include transpose/partials/casts and dispatch in
timing and peak memory. Input/matrix projection work is unchanged; only repeated
scalar evaluations are reduced. Cached control remains ordinary autograd.

Preserve zero/one/three steps, every intervention and the explicit size>32
ordinary-group fallback. Use separate version-4 staged files and preserve all
prior source/records. Validate raw shapes/layouts/dtypes/device/steps and launch
limits. All warp lanes and block threads participate in shuffles/barriers,
including incomplete channel/token tiles.

Repeat independent CPU FP64 equations, all adjoints/finite differences,
calibration/counts and all five complete-model causality/locality/save-load/
exact optimizer-recovery checks. Repeat the full version-3 CUDA suite:
FP32/BF16, group sizes 1/8/32 and explicit size-37 fallback, incomplete tiles,
actual 2048-token gradients, nonzero M, removals, full-model ordinary-autograd
adjoints and native BF16 cast edgecases. Registered tolerances remain FP32
output/input 1e-6/2e-4, M adjoint 1e-5/3e-4 even with BF16 inputs, BF16
output/input .00025/.025, full-model output .004/.02 and parameter gradients
.0001/.05. Record actual errors and preserve any development failure/repair.

Only after all checks pass, freeze sources and run the same eight variants and
48 profiles, three alternating-order rounds per corpus, 20 warmup/100 measured
updates with the same backbone/data/optimizer/BF16 recipe. No source changes or
concurrent GPU work. Every primary resource comparison must pass on both
corpora before integration and the original eleven-variant two-corpus language
screen. Retire failure without switching controls or weakening gates.

Register reuse/recomputation is an execution technique, not novel FFN theory.
No language, inference or sustained-resource advantage follows from passing
numerical checks or the short hardware screen.
