# Rational activation: equations and limits

For each group g, define `a_j=tanh(theta[g,j])`, j=0..3, and
`beta=1+0.75*tanh(theta_denominator[g])`, so `0.25 < beta < 1.75`.
The activation is

`phi_g(z) = SiLU(z) + 0.25 * sum(a_j*z^j, j=0..3)/(1+beta*z^2)`.

The denominator is at least one for every real input. The correction has linear
tails, no real poles, and five learned parameters per group. Coefficients are
shared over channels within a group and across tokens; layers have independent
parameters. This finite family cannot adapt optimally to every possible dataset.

## Derivative bound

For `r_j=z^j/(1+beta*z^2)`, direct differentiation gives:

| Basis | Bound on absolute derivative |
|---|---|
| r0 | `9*sqrt(beta)/(8*sqrt(3)) < 0.860` |
| r1 | `1` |
| r2 | `9/(8*sqrt(3*beta)) < 1.300` |
| r3 | `9/(8*beta) < 4.5` |

For r0/r2 the extrema occur at `beta*z^2=1/3`. For r3, substituting
`v=beta*z^2` gives `v*(3+v)/(1+v)^2 <= 9/8`, with equality at v=3.
The correction derivative is therefore below 1.915. Together with the conservative
SiLU derivative bound `1+1/e`, this gives `abs(phi') < 3.283`.
This is an upper bound, not a positive lower bound: the learned correction can
cancel slopes. It does not prove nonvanishing gradients, monotonicity, or stability
of a full gated layer or deep Transformer.

For fixed beta, polynomial division gives

`sum(a_j*z^j)/(1+beta*z^2) = a2/beta + (a3/beta)*z + ((a0-a2/beta)+(a1-a3/beta)*z)/(1+beta*z^2)`.

Thus the family contains affine directions and two curved directions, rather than
arbitrary high-frequency expressivity. The affine-only control failed its material
improvement threshold in the [matched-rate replication](../../research/affine_rate_replication_results.md).

## Initialization and numerical evaluation

All theta values start at zero, so the function and common weight gradients
initially match ordinary SiLU exactly. Coefficient directions can learn immediately;
the denominator receives zero gradient until numerator coefficients move. Theta
uses the global learning rate and zero weight decay.

The implementation evaluates with `w=1/(1+abs(z))`, `u=z*w`, `D=w^2+beta*u^2`
and bases `(w^2, u*w, u^2, z*u^2)/D`. This avoids unnecessarily large polynomial
intermediates without promising finite output for every floating-point input.
Pointwise work uses FP32 (FP64 for derivative tests); the correction is cast back
to input dtype before addition to native SiLU. Checkpoint recomputation preserves
this order. Independent formula/gradient tests include zero and both tails.

Matrix FLOPs remain those of BlockShuffle. Pointwise work, saved intermediates,
and recomputation are additional costs; the measured memory failure remains open.
No novel activation family or general learning guarantee is claimed.
