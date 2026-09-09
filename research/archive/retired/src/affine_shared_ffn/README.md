# Affine scalar gates over a shared SwiGLU bank

## Hypothesis and inspiration
A shared feature bank may need different scalar responses at different depths.
Add per-layer input scale, shift and output weighting while sharing all matrices.
Inspired by [FiLM](https://arxiv.org/abs/1709.07871) and
[Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672).
No novelty certification or improvement is claimed before experiments.

## Architecture
u=Ux, v=Vx; output D[(1+c_l)*SiLU((1+a_l)*u+b_l)*v].
Matrices U,V,D are reused across four existing layers. a,c=.5*tanh(theta);
b=tanh(theta). All theta start at zero, reproducing strict sharing exactly.
Three hidden-width vectors per layer; no extra projections or attention changes.

## Parameters and compute
At four layers, d=192,h=512: 301056 unique FFN weights and 1679040 model weights.
FFN reduction 74.4792%; full-reference matrix FLOPs are retained. The elementwise
operations are additional and must be timed. Controls use FP32 master weights
with the same BF16 autocast policy.

## Risks and proofs
Bounded positive scales limit deformation but do not prevent whole-network
gradient failure. Shared weight directions remain a constraint. Small BF16
changes may be rounded; residual arithmetic preserves exact zero initialization.
See [protocol](../../../../affine_sharing_plan.md).

## Results, verdict and next experiment
REJECTED for promotion: NLL 3.13918 versus strict sharing 3.14017 is an
unconvincing one-seed change, while forward speed falls from about 85060 to
68597 tokens/s. The quality gap to full SwiGLU 3.06505 remains. Preserve this
negative result; do not fund an ablation sweep without a stronger signal.
