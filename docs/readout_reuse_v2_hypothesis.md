# Readout reuse execution v2: retain projections, reconstruct grouped responses

Registered after execution v1 completed all 54 resource profiles with terminal
exit code 0 and failed every primary comparison. Its 396.1 MiB allocation saves
26–29%, but throughput ratios .757–.804 miss the slowdown limit. No language
training occurred. Preserve v1 sources, errors, records and result.md.

Keep the [mathematical hypothesis](readout_reuse_hypothesis.md) unchanged:
P/C, 1152 detectors, groups of 32, centered calibrated response, alternating
signs, initialization, controls, parameters and forward projection counts.
This is an execution revision, not a new architecture or a looser resource gate.

Hypothesis: retaining BF16 projected z, while reconstructing only response and
readout values in backward, can remove projection recomputation and some
ordinary checkpoint dispatch/storage overhead. Fuse scalar responses with
small grouped products in CUDA. V1's controls suggest grouped mixing is not the
only overhead; they do not isolate a causal bottleneck. The saved z bank increases
memory, and raw-kernel/transpose/partial/cast overhead may still fail the gates.

A complete FFN autograd operation saves x/P/C references, optional independent
Q for the untied control, and projected z. It does not retain token-sized FP32
responses or grouped output across all layers. Forward computes z with native
Torch linear projection under autocast, then FP32 scalar/group operations,
native Torch conversion to the projection dtype, the native linear readout and
its original output gain. Backward reconstructs the response/readout from z,
uses native Torch projection adjoints, and sums both master-P contributions.
The [adjoint audit](readout_reuse_adjoint.md) gives the real-valued equations;
its joint/per-variable CPU finite differences passed with maximum error 4.5e-12.

Precision is part of the recipe. Preserve the projection/output BF16 cast sites,
output gain's BF16 rounding, hidden projection adjoint's BF16 cast, and separate
input/readout BF16 weight-gradient products before their FP32 master-P sum.
No straight-through activation approximation, detached readout gradient, TF32,
fast math or FMA fusion. FP32 master weights, internal response/group buffers
and correction gradients. Raw CUDA kernels accept FP32 only; native Torch casts
handle BF16. Use Torch's current context/stream, never create an independent
CUDA context or issue concurrent GPU work.

One warp handles one token/group forward and input-adjoint products, with
full participation of padded lanes. Deterministic 32-token tiles accumulate
the correction gradient, followed by a fixed reduction; no atomics. Size>32
or unsupported dtype uses the original ordinary-checkpoint path explicitly.
Fixed signed readouts use a scalar response/sign kernel; diagonal controls keep
their original assembled full grouped work and actual smaller parameter count.
The untied same-response control retains its independent Q and its initialized
residual scaling. Cached remains ordinary autograd. Every intervention preserves
its original calibration, signs and gradient masks. Validate shapes/layouts,
dtypes/devices, tile limits and incomplete token/group tiles before launch.

Projection work: forward unchanged at 9,732,096 FLOPs/token across four layers.
Primary training now uses six P-products (two forward/four adjoint), plus four
group products including response/readout reconstruction: 29,491,200 FLOPs/token,
4*(12*512*1152 + 8*1152*32), plus scalar/tile/reduction work. Saved z and all
temporary buffers/casts/transposes/compilation/dispatch count in measured VRAM
and time. This accounting does not establish hardware improvement.

Separate v2 staged prototype/kernel/check/profile files. Repeat every v1 CPU
FP64 equation/gradient/finite-difference, initialization/count/quadrature,
causality/locality and exact optimizer-recovery check. Repeat all v1 CUDA
fixtures and full-model comparisons with the original independent reference,
plus raw forward/input/correction adjoints at group sizes 2/8/32, incomplete
token tiles, actual 2048-token shapes, nonzero corrections and all removals.
Include explicit size-34 fallback and native BF16 cast ties/zeros/subnormals/
infinities/NaN classification. Preserve tolerances: FP32 atol 1e-5/rtol 3e-4;
BF16 output .004/.02 and gradients .0001/.05. Record every actual error and
development failure/repair; never silently widen tolerances.

Only after successful terminal checks, freeze sources and repeat all nine
variants, both corpora, three alternating rounds, 20 warmup/100 measured
updates, identical backbone/data/budget. Retire any resource failure before
language. Primary must pass all six resource comparisons before exact
production equivalence and the original 24-run language screen. No control
promotion, width change, new initialization or quality-gate change. Language,
repeated-seed and originality requirements remain entirely unproven.
