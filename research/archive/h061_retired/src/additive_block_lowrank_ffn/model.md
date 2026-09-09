# Additive projection mathematics

For input width n, output width m, G equal blocks and global rank r:

`W = S + L R`, with `S=blockdiag(S_1,...,S_G)`, `L in R^(m*r)`, `R in R^(r*n)`.

The implementation evaluates the three small products directly. The map uses
`nm/G+r(n+m)` trainable weights. For all three SwiGLU projections at G=8,
h=8d/3,r=d/8, the total is `19d^2/8`, or19/64 of the full FFN budget.
Logical forward matrix work is twice this count per token. Additions, layouts,
nonlinear operations and backward storage require separate measurement.

The direct block path avoids a low-rank-only input bottleneck. Global interactions
enter before the SiLU gate. The two terms can still interfere, and no rank or
conditioning guarantee is imposed during training.

## Fixed initialization

Let rho=1/2 and dense-reference standard deviation sigma=.02, divided by
`sqrt(2*layers)` for down projections. Each block of S is semi-orthogonal with
`s=sigma*sqrt((1-rho)*max(n,m))`. L has orthonormal columns and R orthonormal rows,
each scaled by `sqrt(g)`, where `g=sigma*sqrt(rho*n*m/r)`.

Consequently `||S||F^2=(1-rho)*mn*sigma^2` and `||LR||F^2=rho*mn*sigma^2` in exact
arithmetic. Independent symmetric draws give zero expected cross inner product;
combined energy and row energies vary in each finite draw. All factors start
nonzero. Name-local CPU generators preserve common Transformer initialization.

## Why the optimizer has explicit multipliers

The frozen multipliers for S,L,R are

`aS=sqrt((1-rho)*G)`, `aL=sqrt(rho*n/(2*r*g))`, `aR=sqrt(rho*m/(2*r*g))`.

Consider independent centered entry perturbations with variances
`eta^2*aS^2`, `eta^2*aL^2`, `eta^2*aR^2`. The first-order effective change is
`DeltaW=DeltaS+DeltaL*R+L*DeltaR`. The expected squared Frobenius norm of its three
terms, divided by `mn*eta^2`, is respectively `1-rho`, `rho/2`, `rho/2`.
For example `E||DeltaL*R||F^2=m*eta^2*aL^2*||R||F^2`, and `||R||F^2=r*g`.
Independence eliminates cross terms, so the total is one.

This calculation explains the calibration under independent perturbations. It
does not make real Adam steps independent, equalize every coordinate/direction,
or establish optimizer invariance or faster convergence. Actual task gradients
are correlated. Parameter-level AdamW decay, betas, epsilon and clipping remain
explicit parts of the frozen policy. A local mathematical equality is not a
language-quality result.

## Constructive interaction example

The [witness](witness.py) realizes
`Q(x)=(sum(x_i^2)/2, sum((i+1)*x_i^2)/2, (sum x_i)^2/2, 0,...)`.
Local signed coordinate pairs occupy two hidden rows per input coordinate;
a global signed-sum pair occupies two unused rows. The identity
`SiLU(z)*z+SiLU(-z)*(-z)=z^2` supplies the squares. Three down-factor rows select
the desired sums. The construction fits `2*(d+1)` active features into h=8d/3.

Its first three Hessians are I,diag(1,...,d), and ones. Independent forward,
Jacobian and Hessian tests check the actual block/factor tensors. The retained
[headwise obstruction](../../../../headwise_hessian_obstruction.md) excludes
this vector from the specified square-mixer additive-headwise class. This does
not prove universal approximation, containment of all headwise functions, a new
function class, easy learning, or language-model superiority over dense/narrow.

## Prior art

Sparse-plus-low-rank pretraining is established by [SLTrain](https://arxiv.org/abs/2406.02214).
The [pinned implementation audit](../../../../additive_block_lowrank_qualification_plan.md)
discloses differences in support, initialization, scaling and execution. Existing
structured FFNs and diagonal-plus-low-rank methods are discussed in
[H058](../../../../additive_block_lowrank_proposal.md). No novelty claim follows
from the block/global combination or the calibration calculation.
