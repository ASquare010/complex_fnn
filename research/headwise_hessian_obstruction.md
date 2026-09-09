# A learned square mixer does not remove every headwise interaction limit

**A local expressivity proof, not a novelty or language-model performance claim.**
The earlier [fixed-coordinate argument](archive/retired/src/multihead_ffn/headwise.md) leaves
open whether learning the input mixer can remove a particular restriction.
Here is an explicit vector-valued quadratic that no single additive headwise
FFN with a square input mixer can represent exactly, even when its head
functions are arbitrarily flexible. A conventional full SwiGLU layer can
represent the same function with its ordinary width.

## Function class and target

Let d >= 3. Split d input coordinates into at least two nonempty heads, each
strictly smaller than d. Consider

    F(x) = B * concat_h g_h((A*x)_h),    A in R^(d by d),

where A and the linear output map B are freely learned. The g_h functions may
have arbitrary hidden width, output size and scalar activation shapes, but
see only their own head coordinates. Fixed biases would not change the argument.
The restriction is square, non-overcomplete input mixing and additive head
outputs before a linear B; it is not a restriction to a specific activation.

Define Q:R^d -> R^d by its first three coordinates

    Q_1(x) = (1/2) sum_i x_i^2,
    Q_2(x) = (1/2) sum_i i*x_i^2,       i=1,...,d,
    Q_3(x) = (1/2) (sum_i x_i)^2,

and set remaining outputs to zero. Its three input Hessians are

    H_1 = I,    H_2 = diag(1,...,d),    H_3 = 1*1^T.

These are derivatives with respect to the FFN input, not Hessians of a training
loss with respect to model parameters.

**Claim:** No F in the headwise class above equals Q on a nonempty open set.

## Proof, including learned A and B

1. A must be invertible. If A*v=0 for nonzero v, F is constant along every
   sufficiently short line x+t*v inside the open set. Q_1 has second derivative
   ||v||^2 along that line, a contradiction. This also rules out singular-mixer
   escape cases without assuming differentiability of individual g_h.

2. In q=A*x coordinates, each scalar output is a sum of functions of disjoint
   heads. Its input Hessian C_i is block diagonal in the common head partition,
   so H_i=A^T*C_i*A. Exact equality to the smooth Q makes those scalar sums
   smooth locally; restricting all other heads to fixed coordinates shows that
   the relevant head-local scalar functions have these derivatives even if
   their internal parameterization was nonsmooth. Arbitrary linear output
   mixing does not change block diagonality of C_i.

3. H_1=I implies C_1=A^(-T)*A^(-1) is positive definite and block diagonal.
   Let O=C_1^(1/2)*A. Then O^T*O=I and

       H_i = O^T * [C_1^(-1/2)*C_i*C_1^(-1/2)] * O.

   The bracketed matrices share a nontrivial block partition. Consequently
   H_2 and H_3 would have a common nontrivial orthogonal invariant subspace.
   A symmetric diagonal matrix with distinct eigenvalues has only coordinate
   spans as invariant subspaces. But 1*1^T preserves no nonempty proper
   coordinate span: it maps each coordinate basis vector to the all-ones vector.
   This is a contradiction.

The argument applies at any point in the open set because Q has constant
Hessians. It is independent of head hidden width, activation parameter count,
router complexity confined within a head, or how A/B were optimized.

An equivalent check uses commuting orthogonal projectors. Commutation with
H_2 forces a projector P to be diagonal with entries 0 or 1. If its rank is
k in {1,...,d-1}, then

    ||P*H_3 - H_3*P||_F^2 = 2*k*(d-k) > 0.

Thus no nontrivial projector can commute with both matrices. This finite
calculation is another form of the same exact obstruction.

## Constructive full-SwiGLU representation

The identity SiLU(z)-SiLU(-z)=z gives

    SiLU(z)*z + SiLU(-z)*(-z) = z^2.

Use one positive/negative hidden pair for each linear form x_1,...,x_d and
sum_i x_i: **2*(d+1) hidden units** total. Both the up and gate rows are the
same signed linear form. Give the two down columns in each pair identical
coefficients: coordinate pairs contribute 1/2 to output 1 and i/2 to output 2;
the all-ones pair contributes 1/2 to output 3. Other output weights are zero.
This implements Q exactly in real arithmetic with an ordinary bias-free SwiGLU.

For widths divisible by three and d>=3, the repository's full hidden width
8*d/3 is at least 2*(d+1); unused hidden units can be zero padded. At d=384 the
construction needs 770 of the available 1,024 hidden units. This is a constructive
representability result, not a claim that gradient descent finds these weights.

## Scope, consequences and prior art

This supplies one concrete full-SwiGLU function outside the single square-mixer
headwise family. It does not show that full SwiGLU contains arbitrary headwise
functions, establish a global ordering of the two families, or give a lower
bound on language-model NLL or ordinary value-only approximation error. Stacked
layers, nonlinear/cross-head routing, an overcomplete input projection, attention
and the complete Transformer are outside this single-module statement.

The proof applies to the proposed [structured square mixers](structured_headwise_budget.md)
as a subset of the unrestricted square-mixer class. Larger heads may still be
useful empirically, but learned scalar shapes cannot make this architecture
exactly universal. An overcomplete or cross-head mechanism can escape the
assumption and needs its own parameter/optimization controls.

Function decoupling and block decoupling are established topics; the
[2015 project publication record](https://imarkovs.github.io/decouple/publications.html)
includes block-term methods. [Second-order function decoupling](https://arxiv.org/abs/1805.08479)
already uses Hessian/tensor information, and
[multi-layer decoupling](https://arxiv.org/abs/2604.10858) studies a broader
composition setting. This note applies elementary simultaneous-congruence and
invariant-subspace reasoning to a local FFN witness. No claim that the argument,
identity, witness or general method is novel has been established. The 2015
bibliography was verified; its linked PDFs were unavailable in this audit.

The [accompanying CPU sanity check](../results/headwise_hessian_witness_v1/result.json) constructs the actual DenseFFN, checks
forward values and input Hessians, and checks the common-commutant rank at small
dimensions. Numerical checks support the implementation; the proof above,
rather than a numerical rank estimate, establishes the general claim. No GPU
training or validation target is added to H051.


## A parameter-matched way to investigate the excluded assumption

For a future overcomplete variant, map d inputs to m>d coordinates before the
head split, then return m coordinates to d outputs. With two-factor structured
input/output maps (intermediate width d, G groups), one headwise SwiGLU per
head, and private hidden width de, the count is

    P = 2*d*(d+m)/G + 3*m*de.

The arithmetic choice d=384, m=512, G=16, H=8, dh=64, de=200 uses
43,008 mixer weights plus 307,200 private weights: again **350,208 per layer**,
**2,801,664 across eight layers**. All matrix dimensions are multiples of eight.
There are 1,600 private hidden features and no router. This is a concrete
parameter-feasible alternative to merely adding activation coefficients.

The proof above uses invertibility of square A and therefore does not cover
this rectangular map. That does not prove the proposed structured maps can
represent Q, dominate the existing models or train well. Structured factors
may impose other restrictions, and overcomplete head inputs have correlated
coordinates; the previous orthogonal-square Gaussian calibration cannot be
carried over without checking it. The subsequent [H053 standalone implementation](archive/retired/src/multihead_ffn/overcomplete.md)
now supplies an explicit representation using these actual constrained factors,
and passes derivative, covariance-scale and native BF16 checks. Its isolated
forward/backward allocation exceeds full SwiGLU by 20.37%. It remains unregistered
for language training; full-model memory and a balanced quality screen are still
required. This is one represented function outside the square class, not a
complete containment, novelty or quality theorem.
