# H008: cross-group products without more learned weights

Status: EXPERIMENTING. The shared-width comparison missed the quality target.
The minimal coupled implementation and mathematical tests are ready.

## Structural limitation
For grouped up/value projections sharing the same input partition, one FFN is
F(x)=sum_g D_g psi_g(x_g), even after a fixed hidden-channel permutation.
For coordinates i,j in distinct input groups, every output component has
partial_i partial_j F=0 wherever differentiable. Both grouped GELU and grouped
SwiGLU obey this additive separability. This claim is at the FFN input: the
outer RMSNorm and multi-layer Transformer introduce other couplings, so it is
not a separability theorem for the full network.

## Minimal proposed change
Before the value projection only, use
M_lambda x = (x + lambda P x)/sqrt(1+lambda^2),
where P cyclically shifts input groups and lambda=.5 is a fixed initial test.
Then F(x)=D Q [SiLU(Ux) * V M_lambda x], with the same grouped U,V,D and hidden
shuffle Q. The learned parameter count is unchanged. Compare lambda=0 and .5
using identical U,V,D initial tensors and training; do not add a learnable router.

A scalar component SiLU(a)*(a+lambda b)/sqrt(1+lambda^2) has mixed derivative
lambda*SiLU'(a)/sqrt(1+lambda^2), nonzero near a=0. Thus this construction can
represent cross-group interactions excluded by the original grouped sandwich.
It is not claimed to contain every function of the original model for fixed
lambda, nor to dominate a dense FFN.

## Conditioning bound
P is an orthogonal permutation. Triangle and reverse-triangle inequalities give
(1-|lambda|)||x|| <= ||(I+lambda P)x|| <= (1+|lambda|)||x||.
After normalization, for lambda=.5 singular values lie in
[.5/sqrt(1.25), 1.5/sqrt(1.25)] = [.4472,1.3417].
The mixing is invertible and well-conditioned; this does not bound the full
SwiGLU or Transformer Jacobian. No additional trainable parameters are needed,
but a roll/copy, additions and normalization cost runtime.

## Prior art and decisive test
Cross-feature gating is established in [MAXIM](https://arxiv.org/abs/2201.02973),
and structured projection/shuffle designs already exist. The exact sparse
coupling proposal is a research hypothesis, not certified novelty.
Validate the mixed-derivative claim and singular bounds numerically, then use
the same short LM and complex-function protocol if this branch is pursued.


## Frozen first screen
Use structured_swiglu_coupled at d=192,h=512,G=4,L=4, seed 17, 200 steps,
the unchanged micro_v1 protocol and data. Compare the existing exact-weight,
exact-parameter uncoupled structured_swiglu control (NLL 4.64820), narrow
SwiGLU (4.41907), and narrow GELU (4.24112). No sweep over the mixing constant.
Promotion requires at least 1% lower NLL than the better narrow control and
noncatastrophic forward speed/memory under the original protocol. Improvement
over only the weaker grouped control is mechanistic evidence, not sufficient
for a larger LM budget. Preserve negative results.

## Diagnostic function claim, frozen before function runs
For independent a,b,c,d uniform on [-1,1], target y=a*(a+0.5*d). Any additive
function across the four input groups has irreducible population squared error
at least Var(0.5*a*d)=1/36. Since Var(y)=4/45+1/36=7/60, the minimum population
MSE normalized by Var(y) is 5/21. This uses orthogonality of the interaction to
all univariate additive functions, not a finite-training assertion.

The coupled model can represent this target exactly with two hidden features:
SiLU(a)-SiLU(-a)=a, multiplied by (a+0.5*d)/sqrt(1.25), with output weights
sqrt(1.25) and -sqrt(1.25). The existing hidden shuffle routes the features to
two output coordinates; a fixed sum over output coordinates recovers y.
This explicit construction proves a separation on this chosen function, not
a universal inclusion relation or superiority over dense SwiGLU.

A function screen will include this constructed target, an additive control
(a*a+0.5*d*d), and a pure cross product (a*d) that is not covered by the exact
construction. Compare full SwiGLU h=16, narrow SwiGLU h=4, and grouped/coupled
SwiGLU h=16,G=4 at input/output width 4. The compressed models have 48 matrix
weights plus one common scalar bias; the full control has 192+1. Use the same
train 2048/test 4096 samples, FP32, AdamW .003 with no decay, batch 256, 1000
steps, seeds 17,29,43. Archive checkpoints, histories, raw metrics, environment,
source hashes and all data hashes. This small diagnostic cannot establish LM
usefulness; it tests whether optimization realizes the proven interaction.
