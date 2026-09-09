# H072 - Ungated width allocation and bias-free parity limits

This qualifies an existing conventional control, not a new activation family.
BlockShuffle already implements `gated=False`; no active model uses that path.
The ledger and saved readable sources contain no matching GELU BlockShuffle
screen. Earlier H037's added affine path and H071's rotations remain rejected.

## Spend the same parameter budget with two projections

For d384, G8 and hidden h >= d, each structured projection has d(d+h)/G weights.
A SwiGLU layer uses three such projections; GELU uses two. Equating the counts
at SwiGLU h2048 gives h_GELU = 1.5*(d+2048)-d = 3264.

- SwiGLU h2048: 350,208 weights per FFN.
- GELU h2048, same hidden width: 233,472 weights per FFN.
- GELU h3264, matched weights: 350,208 weights per FFN.

The matched form has 59.375% more hidden features and 70.3125% FFN reduction
against either full dense control. At L8 it has 2,801,664 FFN / 9,099,648 total
Transformer weights. Same-width GELU has 1,867,776 FFN / 8,165,760 total weights
and 80.208333% FFN reduction. Dense narrow GELU h456 matches 350,208 exactly.
Four grouped matrix operations replace six, at equal total matrix MACs for the
matched forms; nonlinear, allocation and launch costs differ. Speed and memory
require measurements. No learned parameters, new operator or activation is added.

## Exact parity identities

Let phi(t)=t*Phi(t) be exact GELU and s(t)=t*sigmoid(t) be SiLU. Symmetry gives
phi(t)-phi(-t)=t and s(t)-s(-t)=t. For any bias-free linear U,V,D, define
G(x)=D phi(Ux) and S(x)=D[s(Ux)*(Vx)]. Products are coordinatewise.
The odd/even parts of F are F_o=(F(x)-F(-x))/2 and F_e=(F(x)+F(-x))/2.

    G_o(x) = D U x / 2
    S_e(x) = D[(Ux)*(Vx)] / 2
    S_o(x) = D[(Ux)*(Vx)*tanh(Ux/2)] / 2

Thus any finite-width single bias-free GELU FFN has a linear odd part, and any
finite-width single bias-free SwiGLU FFN has a homogeneous quadratic even part.
The statement also holds for structured linear U,V,D. At the origin J_G=DU/2
and J_S=0. The first can still be singular or zero: this is no nonvanishing-
gradient claim for a complete network. H037 already used the zero-Jacobian
observation to motivate an affine gated correction; its failed learning result
prevents treating a linear path alone as sufficient evidence.

Neither single-layer family contains the other. For one active coordinate,
phi(t) has even part whose ratio to t^2 differs at t=1 and t=2, so it cannot be
a finite bias-free SwiGLU FFN. Conversely t*s(t) has odd part t^2*tanh(t/2)/2,
whose ratio to t differs at these points, so no finite bias-free GELU FFN equals
it. Both witnesses embed in the full-size structured operators: zero every
factor, then set first.weight[0,0,0]=second.weight[0,0,0]=1 in every required
projection. Channel zero is fixed by both permutations. All other outputs vanish.
These are exact-function comparisons, not universal approximation lower bounds
between the full families, novelty priority or Transformer separation.

## A population obstruction for the multiplicative target

Under independent Uniform[-sqrt(3),sqrt(3)] coordinates, E[a^2]=1 and E[a^4]=9/5.
For T=ab+2bcd+a^2*d, the even part is ab and the odd part is 2bcd+a^2*d.
The best linear approximation to the odd part in population squared error is d.
Its residual R=2bcd+(a^2-1)d is orthogonal to every input coordinate, with

    E[R^2] = 4 + (9/5-1) = 24/5
    E[T^2] = 1 + 4 + 9/5 = 34/5.

On a centrally symmetric distribution odd and even errors are orthogonal.
Every finite single bias-free GELU FFN therefore has population error at least
24/5 for this scalar target, regardless of width, initialization or training.
Relative to zero-predictor error the lower bound is (24/5)/(34/5)=12/17,
approximately 70.5882%. Its even error may add more; attainability is not claimed.

For cyclic vector targets at d384 (also d4), the target and residual covariance
matrices are (34/5)I and (24/5)I: the distinct centered product basis terms are
orthogonal under independent symmetric coordinates. Any fixed orthogonal output
rotation and positive coordinate scales preserve the ratio of their total
population energies. This includes fixed training-derived scales when evaluating
independent new inputs. It is a population bound, not an exact bound on the
saved finite held-out split, a bound for biased/deeper GELU networks, or a claim
that SwiGLU can learn this target efficiently. H071's empirical failures stand.
A three-node product Gauss-Legendre rule exactly integrates the needed moments;
its numerical check is quadrature of polynomial identities, not a training run.

## Prior art and intended use

[GELU](https://arxiv.org/abs/1606.08415), [GLU variants](https://arxiv.org/abs/2002.05202)
and [structured FFNs](https://arxiv.org/html/2406.16450v1#S2.SS1) are established
components. The parameter reallocation and elementary symmetry consequences
are controls and analysis, not an architecture-priority claim. No cited model's
training result is used as a local measurement.

The purpose is to avoid overlooking a simpler matched conventional control and
to expose its known functional tradeoff. Passing local checks earns a separately
frozen learning comparison with positive controls and absolute-error diagnostics,
not direct language training or a breakthrough. Future synthetic assays must
report whether their positive controls actually generalize. The prior target
still requires both full-reference NLL allowances, narrow wins, real resources,
longer three-seed evidence, convergence, scale, broader data and strong prior art.
