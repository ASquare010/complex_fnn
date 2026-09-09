# Equations, expressiveness and scoped stability

**Corrected result:** At the same peak LR 0.0012, the
[three-seed comparison](../../../../affine_rate_replication_results.md) finds only
0.101% lower mean NLL with affine and wins in 2/3 seeds. This fails the frozen
material-benefit gate. The earlier 1.755% advantage compared different rates.
Keep these implemented families and their proofs as research alternatives;
they have not earned promotion over the stronger plain control.

Let G=8 groups and share a scalar activation across the h/G channels in each
group. Projections still mix the inputs, and multiplication by a separate value
projection creates interactions. A scalar activation alone does not mix channels.
Different layers learn independent shapes. Fix correction amplitude A=1/4.

## Shifted quadratic Bezier bank

For K=4, anchors q=(-2,-2/3,2/3,2), define

    c_k = q_k + 0.5 tanh(theta_center[k])
    s_k = 1 + 0.5 tanh(theta_slope[k])
    t_k(z) = sigmoid(s_k (z-c_k))
    a_gk = tanh(theta_control[g,k,0])
    b_gk = tanh(theta_control[g,k,1])
    phi_g(z) = SiLU(z) + A/K sum_k [2(1-t_k)t_k a_gk + t_k^2 b_gk].

This uses the user's single quadratic equation with its first control fixed at
zero. Distinct shifted coordinates prevent the earlier same-coordinate bank
from collapsing to one quadratic. K stays finite, so arbitrary approximation
accuracy is not claimed. Centers move at most 0.5; slopes stay in (0.5,1.5).
Parameters per layer: 2GK+2K = 72. Both base recipes have 2,801,664 FFN weights
at d384/L8; this addition makes 2,802,240, a 70.3064% full-FFN reduction.

**Correction bound (real arithmetic).** The nonnegative basis weights sum to
2t-t^2 <= 1, so each bounded-control quadratic lies in [-1,1]. Averaging yields
|phi-SiLU| <= A. Its t-derivative is 2[(1-t)a+t(b-a)], whose magnitude is <=4.
Since dt/dz <= s/4, the correction derivative is bounded by A*1.5 = 0.375.
SiLU derivative is sigmoid(z)+z sigmoid(z)(1-sigmoid(z)). Using
|z| sigmoid(z)(1-sigmoid(z)) <= |z| exp(-|z|) <= 1/e gives
|phi'| <= 1+1/e+0.375 < 1.743. This is an upper bound, not a positive lower bound.

## Rational residual

For each group, a_j=tanh(theta_coefficient[g,j]), j=0..3, and
beta=1+0.75 tanh(theta_denominator[g]) in (0.25,1.75), define

    phi_g(z) = SiLU(z) + A (a_0+a_1 z+a_2 z^2+a_3 z^3)/(1+beta z^2).

The denominator is >=1 for real z, so there are no real poles. Parameters per
layer: 5G=40; total FFN 2,801,984, a 70.3091% reduction. The correction has
linear tails, unlike the bounded Bezier correction. Its slope can also cancel
part of SiLU, so monotonicity or nonvanishing gradients are not guaranteed.

For r_j=z^j/(1+beta z^2), direct differentiation gives derivative bounds

| Basis | Bound for fixed beta | Uniform bound over allowed beta |
|---|---|---|
| r_0 | 9 sqrt(beta)/(8 sqrt(3)) | < 0.860 |
| r_1 | 1 | 1 |
| r_2 | 9/(8 sqrt(3 beta)) | < 1.300 |
| r_3 | 9/(8 beta) | <= 4.5 |

The first and third maxima occur at beta*z^2=1/3. For r_3 set v=beta*z^2:
v(3+v)/(1+v)^2 <= 9/8, attained at v=3. Thus the residual derivative is bounded
by A*(0.860+1+1.300+4.5) = 1.915 and |phi'| < 3.283.

To avoid large polynomial intermediates, implementation uses w=1/(1+|z|),
u=z*w, D=w^2+beta*u^2 and bases (w^2,u*w,u^2,z*u^2)/D. These are algebraically
the same rational functions, including at zero. This reduces avoidable overflow;
it does not promise finite floating-point output for every representable input.
Independent FP64 formula/gradient tests include zero and both tails.

## A simpler basis interpretation

The Bezier expression is `b*t + (2*a-b)*t*(1-t)`. Each coordinate therefore
combines a logistic step and a logistic-derivative bump. Distinct coordinates
supply distinct step/bump locations; stacking the same coordinate only merges
its two coefficients. This is a useful parameterization of familiar smooth
basis functions, not evidence of a novel function class.

For fixed beta, the rational numerator admits the exact division

    (a0+a1*z+a2*z^2+a3*z^3)/(1+beta*z^2)
      = a2/beta + (a3/beta)*z
        + [(a0-a2/beta) + (a1-a3/beta)*z]/(1+beta*z^2).

Thus its four coefficient directions are an affine correction plus two smooth
rational shapes. Degree three does not imply arbitrary high-frequency
complexity. If it helps, an affine-only trained control is necessary before
crediting the more complicated shape. This decomposition also explains why
learned tails can matter without many additional parameters.

## Initialization and training

Every theta starts at zero. Correction and its input derivative are exactly
zero initially, preserving the base model's function and common weight gradients.
The amplitude controls have nonzero learning signals where their basis is
active. Centers, slopes and rational denominators have zero gradients until
amplitudes move; this delayed learning is intentional and measured.

FP32 pointwise calculation is cast to the incoming activation dtype before
adding native SiLU. FP64 is retained for derivative checks. Theta parameters
use the global learning rate, no weight decay, and no random reinitialization.
Dense width calibration and factor-specific optimizer treatment stay as in their
respective baselines. For BlockShuffle, non-reentrant checkpointing recomputes
activation/product without repeating the projections. Native SiLU-only backward
and the previous fused inference kernels cannot evaluate these new functions.

## What these bounds do not prove

They bound one activation derivative. The gated FFN derivative also contains
projection norms and the other multiplicative branch; attention, residuals and
depth add further factors. There is no full-network stability or convergence
proof. No finite shape family can be assumed optimal for arbitrary data.

Matrix FLOPs remain the base values; pointwise arithmetic, sigmoid evaluations,
saved intermediates and checkpoint recomputation are additional. Report measured
memory and speed. Small parameter overhead is not small execution overhead.


## H037: affine correction and an explicit linear path

`swiglu_narrow_affine_activation` and `blockshuffle_swiglu_affine_activation`
use phi(z)=SiLU(z)+a*z+b, where a=0.25*tanh(theta_gain) and
b=0.25*tanh(theta_bias). Parameters are shared within eight groups, independent
at each depth. Both theta vectors start at zero. This adds 2G=16 weights/layer,
128 over eight layers:2,801,792 unique FFN weights and9,099,776 total at d384/L8.
The full-reference FFN reduction is70.3111436632%. The existing
`affine_shared_ffn` instead transforms SiLU's input and output while sharing
projection banks across depth; it is a different architecture.

For the bias-free layer f(x)=D[SiLU(Ux) elementwise-multiplied by Vx],
SiLU(Ux)=O(||x||), hence f(x)=O(||x||^2) and Df(0)=0 for every finite width.
With affine correction the exact real-arithmetic decomposition is

    f_aff(x) = f(x) + D[a * (Ux * Vx)] + D[b * Vx].

Therefore Df_aff(0)=D diag(b) V. Choose U=0, a=0 and any permissible nonzero
b with D diag(b) V nonzero: the affine family represents a nonzero linear map,
which no bias-free SwiGLU layer of any finite width represents exactly on a
neighborhood of zero. Setting a=b=0 includes the baseline, so this is a strict
layer-family extension. It also applies when the projections use BlockShuffle
factors, provided their product is nonzero. It does not prove separation from
biased SwiGLU, a GELU FFN, or the whole residual Transformer. It is an elementary
structural observation, not a claim of new activation-function theory.

Because |SiLU'| <=1+1/e and |a|<0.25, |phi'|<1.618. The shift b has no input
derivative. No positive lower bound follows. At initialization the individual
shape derivatives are dphi/dtheta_gain=0.25*z and dphi/dtheta_bias=0.25;
they can be learned immediately, although a complete loss gradient can cancel.
This establishes available gradient directions, not immunity to vanishing or
exploding full-network gradients or easy optimization on arbitrary data.

The affine hypothesis follows a uniform-grid fit to the previous rational
checkpoint; its usefulness needs retraining and data-distribution checks under
[the frozen protocol](../../../../affine_activation_plan.md).
Learned activation mixtures and gating changes have established prior art:
[GLU variants](https://arxiv.org/abs/2002.05202) and
[LA/MoA](https://arxiv.org/html/2605.26647v1). Their empirical and theoretical
results do not transfer automatically to this bounded grouped correction.


For beta=1, the rational family contains this affine correction exactly in real
arithmetic: choose its bounded coefficients (a0,a1,a2,a3)=(b/A,a/A,b/A,a/A),
where A=0.25. The numerator becomes (b/A+(a/A)z)*(1+z^2), cancelling the positive
denominator. Thus affine retraining restricts function shapes but also changes
parameterization and optimization geometry; it does not isolate curvature alone.
Floating-point evaluation of the two formulas need not be bitwise identical away
from zero initialization.

## Longer-budget evidence and its limits

The [three-seed WikiText study](../../../../affine_activation_replication_results.md)
finds mean NLL 4.754207 for affine BlockShuffle, 1.755% below selected plain
BlockShuffle, with only 128 extra weights. All frozen quality/memory gates pass.
The recipes use different selected learning rates; the
[completed same-rate control](../../../../affine_rate_replication_results.md)
reduces the measured mean gain to 0.101%. This does not establish an activation-specific training cause.

[Removing the correction after training](../../../../affine_activation_removal_results.md)
costs less than .020% NLL in every seed and preserves quality gates. That result
motivates studying the learned common weights and optimization path. It does
not show that the final scalar shape is necessary for the observed performance.
No all-data optimum, convergence, training-speed improvement or full-network
nonvanishing-gradient guarantee follows. The two richer compiler variants
remain failed training backends despite passing local derivative checks.
