# Local normalization geometry and an FFN scaling symmetry

These are implementation-specific derivations of elementary identities, with
no literature-priority or language-quality claim. The [H057 plan](residual_geometry_plan.md)
uses them to investigate H056's failed short language screen.

## Finite-epsilon RMSNorm derivative

For one d-dimensional token, define s = sqrt(x^T x / d + epsilon) and
N_gamma(x) = diag(gamma) x / s. Here epsilon = 1e-5. Differentiation gives

    J_N(x) = diag(gamma) [I/s - x x^T/(d s^3)].
    J_N(x)^T g = (gamma * g)/s - x * mean(x * gamma * g)/s^3.

This is the norm invocation's local VJP. If x also enters a residual identity
path, its total loss gradient includes that path and is not this expression.

Without learned gamma, the symmetric bracket has eigenvalue 1/s on every
direction perpendicular to nonzero x and epsilon/s^3 in the radial direction x.
At x=0 all eigenvalues equal 1/s. Both are positive for epsilon>0. With arbitrary
gamma, the operator norm is at most max(abs(gamma))/s, and the minimum singular
value is at least min(abs(gamma))*epsilon/s^3. The lower bound can be zero if a
gamma coordinate is zero; it is not a useful whole-network gradient guarantee.

For c>0 the exact relation is

    N_{gamma,epsilon}(c*x) = N_{gamma,epsilon/c^2}(x).

Thus finite-epsilon RMSNorm is not exactly scale invariant with fixed epsilon.
When x^T x/d dominates epsilon, its forward value is approximately scale invariant
and its local tangential derivative scales approximately as 1/c. A learned gamma
and the direction of the incoming adjoint affect actual VJP gain. H057 records
these quantities rather than substituting a batch-average RMS into tokenwise
identities. The real module computes its RMS statistics in FP32 even when the
surrounding matrix operations use BF16; a FP64 mathematical reference is tested
separately from the module's floating-point execution.

The FFN residual map has derivative I + J_F J_N at its input. The identity term
is retained, so multiplying isolated norm gains is not a valid proof of vanishing
gradients through this pre-norm Transformer. Attention and subsequent residual
blocks further change the full derivative. Local attenuation can be measured
without asserting that it caused the loss deficit.

## A function-preserving value/down rescaling

For the actual overcomplete headwise FFN,

    F(x) = B concat_h D_h [SiLU(U_h A_h x) * (V_h A_h x)].

Here A_h selects head h of the rectangular mixed input; the private tensors use
transposed storage in code. For any positive c, replace V_h by c V_h and D_h by
D_h/c, with all A/B/U fixed. Multiplication by V and D is linear, so the factors
cancel exactly for every input in real arithmetic. The SiLU gate is unchanged.
This is true independently of mixing structure and does not require RMSNorm.

For a differentiable scalar loss of the FFN output, the corresponding parameter
gradients satisfy

    grad_{V_new} L = grad_V L / c,
    grad_{D_new} L = c * grad_D L,

while input and other parameter gradients are unchanged. Therefore these raw
parameter-gradient norms can be made arbitrarily small or large along an exact
function symmetry. Their magnitude alone cannot establish lost expressivity,
weak functional sensitivity or vanishing gradients. The representation, logits
and loss do not improve through this rescaling. Adam states/updates and decay
are not automatically invariant; H057 performs no optimizer transformation or
training and makes no optimizer remedy claim.

Power-of-two c=8 is used for a finite-precision qualification, not assumed to
provide universal floating-point identity. Original tensors are restored and
verified, and no changed checkpoint is saved.

## Established context

[Zhang and Sennrich, RMSNorm (2019)](https://arxiv.org/abs/1910.07467) introduces
RMS-based normalization and discusses rescaling behavior and implicit learning-rate
adaptation. The finite-epsilon identities above specify our local implementation.

[Van Laarhoven (2017)](https://arxiv.org/abs/1706.05350) studies weight scale and
effective learning rates with normalization, including the incomplete cancellation
by Adam. [AdamP (2020)](https://arxiv.org/abs/2006.08217) examines momentum-induced
norm growth on scale-invariant weights. These results motivate separating scale
from functional change; they do not diagnose this residual architecture or prove
that one of their remedies will improve H056. No new optimizer is implemented.
