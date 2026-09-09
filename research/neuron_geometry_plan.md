# H088 - Can coupled neuron geometry replace width?

Freeze before numerical execution. H086 produced native sparse-operator evidence
but stopped at unsupported CUTLASS. H087 retired rational BlockShuffle and
committed the compact research tree as37b21ed. That previous goal turn made
PROGRESS. The original >=70% FFN reduction, <=1% language-NLL allowance against
both full controls, narrow-control, runtime, multi-seed, scale and broader-data
requirements remain unchanged and unfulfilled.

## Distinct mechanism and controls

The proposed activation operates on pairs, with fixed center c=(1,0), q=x-c,
r2=q1^2+q2^2, a=2*tanh(theta), t=a*r2/(1+r2), and

    f(x) = c + [[1-t^2,-2t],[2t,1-t^2]] q / (1+t^2).

Use the equivalent residual implementation x+(R(t)-I)q for exact identity at
theta=0. Compute t=a*(1-1/(1+r2)). Share one theta across each of four contiguous
groups of pairs. This is input-dependent coupling; a static learned rotation
could instead be folded into a neighboring dense projection. The center breaks
the odd-function restriction of an uncentered bias-free isolated layer.

The real map preserves distance to c, has inverse obtained by negating theta,
determinant1 and singular values in[sqrt(2)-1,sqrt(2)+1]. See the calculation in
[the direction note](research_direction_2026_09_10.md). These are activation-only
bounds, not linear-weight, full-network, optimization or floating-point guarantees.
Familiar normalizing-flow geometry motivates this map; no novelty is claimed.

Also test the user's simplest local scalar alternatives. Bezier1P is ReLU plus
tanh(theta)*u*(1-u) only on0<u<1, with a single layer-shared theta. Bump2P uses
ReLU plus a*u^2*(1-u)^2+b*u^2*(1-u)^2*(2u-1) on the same interval, where
a,b=2*tanh(theta_a),2*tanh(theta_b), shared across four groups (eight controls).
Both initialize exactly as ReLU. Outside the interval, leave ReLU unchanged.
The quadratic has a derivative jump at1; the bump correction and its derivative
vanish at both endpoints. Neither fixes ReLU's negative-side zero gradient.

For b0(u)=u^2*(1-u)^2, max|b0'|=1/(3*sqrt(3)). For b1=b0*(2u-1), its derivative
is(1-6*s^2+5*s^4)/8 with s=2u-1, so max|b1'|=1/8. The constrained bump's positive
interior slope therefore lies in[1-2/(3*sqrt(3))-1/4,1+2/(3*sqrt(3))+1/4].
These coefficient limits are fixed before data is drawn, not selected afterward.

## Fourteen forms; equal fresh comparison

All inputs/outputs have width384 and no dense projection bias. Full GELU/ReLU
use hidden1536 and full SwiGLU1024:1,179,648 weights each. Narrow GELU/ReLU and
all ungated candidates use hidden456; narrow SwiGLU uses304:350,208 base weights.

1. full_gelu;2. full_relu;3. full_swiglu.
4. narrow_gelu;5. narrow_relu;6. narrow_swiglu.
7. groupsort: min/max of each consecutive pair, no learned activation controls.
8. gelu_offset: GELU(z+2*tanh(theta)), four shared shifts initialized to zero.
9. twist_fixed: four theta buffers initialized to atanh(1/2), so a starts at1.
10. twist_learned: same initial function as twist_fixed, four learned theta.
11. twist_identity: four learned theta initialized to zero; starts as linear.
12. bezier_1p: one zero-initialized learned control.
13. bump_2p: eight zero-initialized learned controls.
14. linear: the same two narrow dense projections with no nonlinearity.

The four learned nonlinear candidates are10-13. Fixed twist, GroupSort and offset
are controls and must be reported even if stronger. Candidate weights are
350,209/350,212/350,216 depending on controls: all still exceed70% FFN reduction
against the full counts. Report exact weights, controls and fixed buffers.

Initialize every projection with Gaussian standard deviation1/sqrt(fan_in),
using the established independent SHA256(seed:name.weight) generator. Every
same-width ungated form has exactly matching initial dense matrices. Curves and
initial functions differ as specified; no cross-family identical-function claim.
Only narrow down-projection LR is multiplied by1536/456 (or1024/304 for narrow
SwiGLU), using the existing fan-in calibration convention. Up/gate and shape
controls use the base LR. Full controls have no multiplier. All forms get the
same base rates0.001/0.003 and optimization seeds17/29/43.

## Qualification before any learning:27 checks

-14 actual model-count/buffer checks, one for each form.
-4 CPU FP64 independent-formula and numerical-gradient checks: twist with one
  and four groups, Bezier1P, Bump2P with four groups. Nonzero controls are tested.
  Formula/analytic-gradient comparisons use atol1e-11,rtol1e-10; central numerical
  gradcheck uses epsilon1e-6,atol1e-5,rtol1e-3,fast_mode=True. Count its forwards.
-2 twist-geometry cases, centers0 and1: inverse, distance, determinant and Jacobian
  singular values at radii0,0.01,0.3,1,3,100; four angles and amplitudes
  -1.9,-1,0,1,1.9. FP64 tolerance1e-10; bound slack1e-10.
-1 exact initialization comparison: Bezier/bump versus ReLU, zero twist versus
  linear, nonzero learned twist versus fixed twist; matching dense matrices.
-4 CUDA BF16 whole-FFN checks, one per learned candidate. Finite nonzero gradients
  and outputs versus an independent formula reference at atol0.02,rtol0.02;
  ordinary and non-reentrant checkpoint outputs/gradients must match exactly.
-1 GroupSort reference/gradient check away from ties, exact equality.
-1 scalar slope and endpoint check against the bounds above, FP64 slack1e-10.

Collect exactly27 checks, then execute once. Save paired output/gradient tensors
with fsync and exact readback. Stop on failure; no automatic repair or retry and
no changed tolerances. Do not allocate fitting until all checks pass.

## Data, fitting and diagnostics

Use fresh CPU input seed9814:73,728 examples drawn uniformly from[-sqrt(3),sqrt(3)]
in384 dimensions. Training65,536; rate selection4,096; reporting4,096. Input
permutation seed9283 and output orthogonal QR rotation seed9282 stay fixed, as in
H083/H085. With cyclic a,b,c,d from the permuted input, use four target families:

- smooth: sin(2a)+b^2+exp(c/2)-d;
- oscillatory: sin(3a)*cos(2b)+0.5*sin(4c+d);
- multiplicative: a*b+2*b*c*d+a^2*d;
- piecewise: ReLU(a+b)-0.7*abs(c-d)+where(a>0,b,c).

Rotate outputs, divide by each output's training population standard deviation
without centering. Draw600 batches of256 training indices per optimization seed,
using seed20000+seed, and reuse these streams for every task/form/rate. This is
sampling with replacement, not ordered epochs. No corpus or official test access.

The fixed grid has336 runs,600 updates each:201,600 updates and51,609,600 training
example presentations. AdamW betas(0.9,0.95),epsilon1e-8,decay0,clip1,constant
base rates. Native FP32 CUDA, TF32 off, four CPU threads, one GPU worker. Stage
one task's input/target tensors on GPU at a time. Record synchronization-based
forward, backward and complete-update times; summarize medians after50 warmups.
Peak allocation includes that dataset cache and is a standalone fitting measure.

Record all600 losses, preclip gradient norms and phase times locally. Save shape
controls, parameter-gradient norms and pre/post-activation distributions at
steps0,50,150,300,600, outside timed updates. Save every final model, optimizer
step counts, finite-state checks and source/data/stream hashes. Model checkpoints
do not claim resume support; full optimizer moments need not be stored. Verify
that every parameter has completed600 optimizer steps before saving.

Select each task/form/seed's LR on the selection split, persist that selection,
then report both rates on the reporting split. Compare selected recipes and
matched-rate effects separately. Report mean, median and sample variance/SD across
seeds, plus all individual endpoints, timing, memory, clipping and learned shapes.
Generate parameter-trajectory and learned-function plots, including a2D twist map.

## Frozen decision and limits

The positive assay requires full GELU to beat the zero predictor by at least2%
on at least two task means. Each candidate must:

- finish finite and retain>=70% FFN weight reduction;
- lower selected geometric-mean reporting MSE by>=2% versus EACH narrow GELU,
  narrow ReLU, narrow SwiGLU, GroupSort and GELU-offset control;
- improve the task-aggregate ratio in every seed against each of those controls;
- have no task-mean MSE more than5% above the best of those control task means;
- for either learned twist, additionally beat fixed twist by>=1% in aggregate.

These gates earn only a separately frozen resource/width-frontier study. They do
not establish the1% language-NLL target, multi-dataset success or practical speed.
Record eager overhead honestly; any optimization must later compare equally
optimized controls. Failed fixed recipes close without an automatic longer run.
If fixed twist or GroupSort wins, report that rather than assigning credit to
learned geometry. Reused analytic task families are a limitation despite fresh
inputs. Three optimization seeds do not equal three independent dataset draws.

## Evidence and source boundaries

Pin commit37b21ed, the H087 release receipt, direction note, plan, current shared
source and isolated prototype sources before execution. Use a durable hidden
launcher with PID/UTC/exit records and source archive readback. H087 intentionally
changed active source; do not rerun older frozen source-equality checks against
today's registry. Preserve old evidence and use its source snapshots when needed.

Independently reconstruct all336 checkpoints and rescore both splits exactly on
the original device, recompute every decision/statistic, check initialization and
shape/buffer counts, and verify stored tensors/provenance. No training rerun.
Keep raw data/checkpoints/histories local; commit compact reports, plots, protocol,
source and endpoint evidence under the H087 artifact policy. Update current state
and ledger, then commit a verified coherent result. No new active model is added
without further qualification.

Primary sources: [GroupSort](https://proceedings.mlr.press/v97/anil19a.html),
[KANB](https://doi.org/10.1109/ACCESS.2025.3600484),
[polynomial/trigonometric activations](https://openreview.net/pdf?id=QywpTFx86x),
[normalizing flows](https://proceedings.mlr.press/v37/rezende15.html).
The direction note distinguishes these established results from this hypothesis.
