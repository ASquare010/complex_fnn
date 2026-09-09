# Switch from scalar corrections to coupled neuron geometry

The original acceptance goal in [RESEARCH_GOAL](../doc/RESEARCH_GOAL.md) stays
unchanged. The user's September 10 direction adds an explicit falsification
priority: investigate whether a few shared activation parameters replace width,
retire unproductive branches, and keep a compact, committable research record.

## Why change the mechanism

H083/H085 improved BlockShuffle synthetic fitting but lost to narrow GELU.
Offsets helped more than the tested nonlinear curves. H086's sparse comparator
passes native derivatives but cannot complete its first accelerated inference
case. Further scalar correction tuning and backend repair are not the next
allocation. These exact recipes remain closed or unpromoted.

The next proposed neuron consumes a pair of projected features and changes their
direction according to their joint magnitude. It can directly couple two features
inside the activation. This differs from a static orthogonal matrix, which can
be folded into adjacent weights, and from an elementwise curve, which acts on
each projected coordinate separately. Neither difference proves better learning.

## Candidate: centered rational twist

For a feature pair x=(x1,x2), fixed center c=(1,0), q=x-c and r2=q1^2+q2^2, set

    a = 2 tanh(theta)
    t = a r2 / (1+r2)
    f(x) = c + 1/(1+t^2) * [[1-t^2, -2t], [2t, 1-t^2]] q.

Share theta across one or four contiguous groups of feature pairs. The online
formula uses products, sums and reciprocals. Tanh is applied only to the few
shared controls. Implement t=a*(1-1/(1+r2)) to avoid an inf/inf radius ratio.

At theta=0, f(x)=x exactly in the real formula; a residual implementation can
preserve floating-point identity too. This initializes an ungated FFN as a linear
map, not ReLU. A separately controlled nonzero initialization can test whether
starting with curvature matters. A fixed nonzero twist is required to distinguish
the geometry from learning its shape. The center-zero form is odd and cannot
represent even components in a bias-free isolated FFN; center-one is deliberate.
Compare it against a cheap learned-offset control so constant response is not
mistaken for new nonlinear capacity.

## Local stability calculation to verify before training

This is a rotation of q, with angle phi(r)=2 atan(a r^2/(1+r^2)). It preserves
distance to c and has explicit inverse given by negating a: radius is unchanged.
In orthonormal radial/tangential coordinates its Jacobian is a rotation times
the shear [[1,0],[k,1]], where

    k = r phi'(r) = 4 a r^2 / ((1+r^2)^2 + a^2 r^4).

Thus |k|<=|a|<=2. Its singular values are
sqrt(1+k^2/4) +/- |k|/2, bounded by sqrt(2)-1 and sqrt(2)+1.
The determinant is 1. These real-arithmetic activation bounds do not constrain
the linear matrices, the full network, training dynamics, or floating overflow.
The transform is asymptotically a fixed rotation, which might limit useful
nonlinear diversity. That is a reason to test cheaply, not an advantage to assume.

## Benchmarks that would falsify it

First qualify formula, inverse, nonzero-shape gradients and Jacobian bounds.
Then freeze a fresh fitting comparison against full and narrow GELU/ReLU/SwiGLU,
fixed twist, learned offset and the user's minimal local Bezier/bump controls.
Use shared data/order, three seeds, equal learning-rate searches and bounded
updates. Include widths that meet the original compression target and a later
90/80/70% width frontier only if the mechanism earns it. Report all outcomes,
mean/median/variance, shape evolution, gradient norms, runtime and memory.
No Transformer allocation or new active model follows from algebra alone.

## Prior work checked; no novelty claim

- [KANB, IEEE Access2025](https://doi.org/10.1109/ACCESS.2025.3600484) explicitly
  uses learned Bezier-derived activations. Fixed-x control points and very small
  shared parameter counts are experimentally useful restrictions, not sufficient
  evidence of novelty. Its published formula uses a sigmoid parameterization.
- [Polynomial, Trigonometric, and Tropical Activations, ICLR2026](https://openreview.net/pdf?id=QywpTFx86x)
  studies orthonormal bases and variance-preserving initialization at substantial
  language and vision scales. It is a relevant strong activation comparator.
- [Learning Polynomial Activation Functions, PMLR321](https://proceedings.mlr.press/v321/zhang26a.html)
  concerns polynomial optimization via Moment-SOS. It is not evidence that our
  small local polynomial corrections work with AdamW or replace network width.
- [Variational Inference with Normalizing Flows](https://proceedings.mlr.press/v37/rezende15.html)
  and [Flows on Tori and Spheres](https://proceedings.mlr.press/v119/rezende20a.html)
  establish prior work on learned invertible nonlinear geometry. The proposed
  pair map uses familiar radial/rotation ideas. The search does not establish
  novelty of this exact formula, and inventing a name would not establish it.

The user's quadratic simplification is correct on 0<x<1. The full piecewise
function equals ReLU at zero curvature. Its derivative can jump at 1 when
curvature is nonzero; the residual bump has a zero derivative correction there.
Neither removes ReLU's kink at 0 or its zero negative-input derivative.

- [GroupSort, ICML2019](https://proceedings.mlr.press/v97/anil19a.html) is an
  established coupled activation with gradient-norm preservation almost
  everywhere. Its published universal-approximation result uses constrained
  matrices. Include unconstrained pairwise GroupSort as a cheap activation
  comparator without borrowing that whole-network theorem. A paired activation
  must beat a strong paired baseline as well as scalar GELU/ReLU controls.
