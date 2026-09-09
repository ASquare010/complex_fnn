# H092: nonlinear basis expansion with a direct input path

The user's latest direction prioritizes different mechanisms, quick elimination
and refinement only after quality gates pass. The unrun H091 partial-GELU plan
and prototype are archived; they produced no measurements. H090's four failed
recipes remain closed. Active model notes are consolidated into their READMEs.
The gold objective is unchanged and unmet.

## Mechanism

For d=384, project to m=176 coordinates z=U*x. Generate an even and an odd
nonlinear feature from each coordinate, retain the original input, and read out:

    f(x) = D * concat(x, E(U*x), O(U*x)).

U has d*m weights and D has d*(d+2*m): d^2+3*d*m=350,208 weights.
This is exactly the narrow-GELU h456 budget and 70.3125% below full GELU h1536.
The native implementation uses two projection GEMMs plus basis arithmetic and
concatenation. Matrix forward work is 700,416 FLOPs per example, excluding the
basis, copies and backward. Actual memory and latency must still be measured.

Three candidate bases, with no GELU/ReLU/SwiGLU in the candidate feature map:

1. Rational parity: s=z/(1+abs(z)), E=z*s, O=z*s^2.
2. Hermite parity: E=(z^2-1)/sqrt(2), O=(z^3-3*z)/sqrt(6).
3. Trigonometric parity: E=1-cos(z), O=sin(z)-z.

The learned output coefficients independently combine the two features. These
are fixed bases with trainable input/output weights, not 1-8 learned scalar
activation controls. The experiment tests a different allocation of weights.
No extra basis-shape parameters are counted or implied.

The rational pair behaves quadratically/cubically near zero and linearly in
the tails. For u=abs(z), let t=u/(1+u). Then E' = sign(z)*(2*t-t^2) and
O'=t^2*(3-2*t), so both derivative magnitudes are at most one. The trig pair
has derivative bounds one and two. Hermite derivatives are unbounded; unit
Gaussian variance is a property under that distribution, not trained-data fact.

The direct-input feature map F(x)=[x,E(Ux),O(Ux)] satisfies

    ||F(x)-F(y)|| >= ||x-y||.

For the rational pair its upper Lipschitz bound is sqrt(1+2*||U||^2), and for
trig it is sqrt(1+5*||U||^2). The final D can annihilate directions; these are
feature-map bounds, not full-FFN or network-gradient guarantees. Floating-point
casts also do not inherit a strict real-arithmetic lower-distance theorem.

Without the direct input, a two-feature expansion at this budget permits only
m=304 projected directions. Every function of U*x is constant on ker(U), losing
at least 80 input directions. For an isotropic Gaussian identity target its
best possible normalized population MSE is at least 80/384. The direct path
removes this forced global blindness. It does not make every nonlinear target
easy, and the nonlinear branch still has a low-dimensional projection.

## Controls and initialization

Use three same-count pair controls: duplicate rational-even [E,E], linear [z,z],
and antipodal GELU [GELU(z),GELU(-z)]. The duplicate folds to one even basis;
linear folds to a matrix; antipodal GELU has difference z, so its odd component
is redundant with the direct path. Their optimizer coordinates can still matter.
An ordinary direct-path GELU with m=264 has d^2+2*d*m=350,208 weights and controls
the benefit of two bases versus more independently projected scalar features.
Core linear D*x has 147,456 weights and is the initialization control.

Use seed/name-local Gaussian U initialization at std 1/sqrt(d). Initialize the
first d columns of D at std 1/sqrt(d), and every nonlinear readout column to zero.
All eight forms start at the same linear function. This preserves the exact
real function; different GEMM widths may round differently. Test CPU FP64
initial equivalence with atol/rtol 1e-11, and qualify nonzero nonlinear readouts.
Future training must retain full/narrow conventional references and H090's
strong learned-shift control with equal newly allocated tuning budgets.

## Prior art and scope

Direct input links and expanded features are established ideas, including
[RVFL](https://arxiv.org/abs/1907.00350). RVFL's fixed random hidden weights and
closed-form readout differ from this fully trainable proposal.
[CReLU](https://proceedings.mlr.press/v48/shang16.html) uses complementary phases;
[FAN](https://arxiv.org/html/2410.02675v3) combines Fourier features with learned
projections. The [FAN component study](https://arxiv.org/html/2512.14873v3)
reinforces the need for component controls. Hermite bases are established,
including [ICLR 2026 work](https://openreview.net/pdf?id=QywpTFx86x).
The rational pair is built from ordinary softsign and products. No novelty,
general superiority, or universal trainability is claimed.

## Frozen qualification: 25 checks, no optimizer updates

Eight forms: rational, hermite, trig, duplicate, linear, antipodal, raw_gelu,
core_linear. Freeze source and the H090 release anchors before execution.
Use UV, four CPU threads, one GPU worker, TF32 off, no automatic retries.

- Eight parameter-count and zero-secondary-initialization checks.
- Six FP64 pair-formula and first-derivative comparisons, plus numerical
  gradchecks including zero. Analytic tolerance 1e-11; gradcheck epsilon 1e-6,
  atol 1e-5, rtol 1e-3, fast mode; count actual forwards.
- One initial-function check over every form, FP64 1e-11.
- One algebraic-collapse check for duplicate/linear/antipodal with nonzero
  secondary readout, FP64 1e-11.
- One explicit kernel-vector and direct-path witness, FP64 1e-10.
- One rational/trig derivative-bound check on 4,097 values in [-8,8], including
  zero, FP64 1e-11; save derivative tensors.
- Seven CUDA whole-FFN checks, all forms except core linear, with nonzero extra
  readout: FP32 and BF16 independent output/input/up/down-gradient comparisons
  at atol/rtol 1e-5 and 0.02, plus exact non-reentrant checkpoint comparisons.

GPU probes: seed 9842 for transposed 2x3x384 inputs and cotangents; seed 9843 for
nonzero secondary readout, std 0.02. Basis arithmetic uses FP32, retains FP64
when requested, and casts once. Under BF16, the direct input is explicitly cast
to the same feature dtype before concatenation. Reference equations differ
algebraically while sharing this precision policy.

Save paired tensors, exact readback and hashes. Stop on the first failure and
preserve partial evidence. Independently audit saved observations before
allocating a separate small fitting screen. No refinement, large run, registered
model, speed result or training claim follows from these checks alone.
