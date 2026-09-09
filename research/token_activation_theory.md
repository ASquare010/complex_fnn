# H067: a bounded conditional activation and an exact-function separation

This note concerns one real-arithmetic FFN map, not trained language performance.
The [frozen qualification plan](token_activation_plan.md) defines its numerical
checks and limits. Token-dependent activation mixing is established
[MoA prior art](https://arxiv.org/html/2605.26647v1). The particular residual
parameterization below preserves the existing baseline at initialization.

## Definition, containment and parameter budget

Let s(z)=SiLU(z)=z/(1+exp(-z)), u=U(x), v=V(x), and k=1/4. Define

    a(x) = k*tanh(w^T*x + b)
    F(x) = D[ s(u) * (v + a(x)*s(v)) ].

Products and activations are coordinatewise, while a is one scalar shared across
hidden channels for each token. U,V,D are the existing two-factor BlockShuffle
maps. Setting w=b=0 recovers the same BlockShuffle SwiGLU map for every choice
of projection weights. Setting w=0 and retaining b recovers the static control.
Thus the dynamic family contains both families at their respective shared
projection dimensions. We do not assert strict separation from the static family.

The projection count remains 3*d*(d+h)/G for h>=d. Dynamic adds d+1 weights per
layer, static adds one. At d384/h2048/G8/L8, the totals are 2,804,744 dynamic FFN
weights and 2,801,672 static FFN weights, against 2,801,664 plain and 9,437,184
full. Router computation adds O(d) work per token, plus hidden-width SiLU and
multiplications. Counts alone do not determine memory or throughput.

## A value envelope and the limits of the gradient statement

Write the modified value as

    q(v,a) = v + a*s(v) = v*(1+a*sigmoid(v)).

Because |a|<1/4 and 0<sigmoid(v)<1, its multiplier lies between 3/4 and 5/4.
It preserves the value sign and limits its magnitude change. This bound is for
the internal value branch, not for D's output after cancellation between channels.

Holding a independent of v, dq/dv=1+a*s'(v). A convenient global bound is
|s'(v)| <= 1+1/e: s'=sigmoid(v)+v*sigmoid(v)*(1-sigmoid(v)), and the absolute value
of the second term is <= |v|*exp(-|v|) <= 1/e. Therefore

    dq/dv >= 1 - (1+1/e)/4 > 0.658.

This is a positive scalar partial derivative. The full x derivative also contains
the router derivative, projection nullspaces, SiLU(u), output mixing and residual
paths. It has no positive singular-value lower bound from this argument. The
router itself saturates for large |w^T*x+b|. Vanishing/exploding full-network
gradients and easy optimization are not ruled out.

At w=b=0, the derivative of F with respect to b is
k*D[s(u)*s(v)]. The derivative with respect to w is its outer product with x.
This is generally nonzero even though the correction is initially zero; it can
vanish on particular inputs or parameter settings. The qualification measures
nonzero router signals on its fixed random examples, not on every possible datum.

## An actual structured witness

Choose the first hidden feature and first output coordinate. Set every factor
entry to zero except first-factor entry [0,0,0] and second-factor entry [0,0,0]
of U and D; for V use first-factor [0,0,1] and second-factor [0,0,0]. Set these
entries to one. Channel zero is unchanged by both intervening permutations.
Set router w=e3 and b=0. These assignments are valid at the actual
*d384/h2048/G8* shape and yield

    T(x) = s(x1)*x2 + k*tanh(x3)*s(x1)*s(x2)

in output coordinate one, with all other output coordinates zero. No additional
hidden feature or dense mixer is needed. On x1=x2=1, this becomes

    T(1,1,t) = c0 + c1*tanh(t),
    c0=s(1), c1=k*s(1)^2 != 0.

This construction proves representability, not that gradient descent finds these
sparse factor values. The implementation check verifies values and the x3
partial derivative, including the actual channel permutations.

## Exact nonrepresentation by one finite conventional FFN

**Claim.** T cannot equal any finite single SwiGLU FFN with real parameters on a
real open neighborhood of the line segment x1=x2=1, x3 near zero. This remains
true if the conventional FFN is allowed biases, unrestricted dense projections,
an output bias and an affine residual. Finite exact-GELU FFNs also cannot equal T.
This is an exact-function statement; there is no approximation-error lower bound.

**Proof for SwiGLU.** Restrict a putative scalar output to that line. Every
finite SwiGLU FFN has the form

    f(t) = p(t) + sum_j (A_j*t+B_j)*s(C_j*t+D_j),

where all coefficients are real, the sum is finite, and p is affine. Extend t
to a complex variable z. The continuation of s(z)=z/(1+exp(-z)) is meromorphic,
with simple poles at i*pi*(2m+1) and residue equal to the pole itself. This
standard pole description also appears in the
[SiLU analysis of Tran et al., Appendix C.1](https://arxiv.org/html/2606.17816v1).
The residue argument below applies it to our witness.

The target c0+c1*tanh(z) has poles z_n=i*pi*(n+1/2), each with residue c1.
Agreement on a real interval implies agreement of these meromorphic functions
by analytic continuation, so their residues at every z_n must agree.

A term with C_j=0 is entire. If a term with C_j!=0 has a pole at z_n, then
C_j*z_n+D_j must be purely imaginary. Since D_j is real and z_n is purely
imaginary, this forces D_j=0. Its residue is consequently

    (A_j*z_n+B_j) * (C_j*z_n)/C_j
      = A_j*z_n^2 + B_j*z_n.

Let I_n be the subset of terms having a pole at z_n. There are finitely many
possible subsets but infinitely many target poles, so one subset I occurs at
infinitely many distinct z_n. On that infinite subsequence, residue equality gives

    (sum_{j in I} A_j)*z_n^2 + (sum_{j in I} B_j)*z_n = c1.

A polynomial of degree at most two cannot equal the nonzero constant c1 at
infinitely many distinct points unless it is identically that constant. Its
constant coefficient is zero, which contradicts c1!=0. Hence no such f exists.
Cancellations between terms and additional poles elsewhere do not invalidate this
argument: I contains every term with a pole at the selected points, and the
residues of a finite sum add even when some cancel. The affine p has no poles.

**Proof for exact GELU.** z*Phi(z), with Phi defined through the entire error
function, is entire. Finite sums and affine compositions of it are entire.
Even multiplying by an affine gating branch preserves that property. An entire
function cannot have the target's nonzero pole residues after analytic
continuation. This statement concerns exact GELU, not its tanh approximation.

Together with the containment above, the structured dynamic family strictly
extends its plain BlockShuffle family in exact representability. Indeed, this
witness escapes even unrestricted dense single-FFN SwiGLU at any finite width.

## What the claim does not establish

Finite conventional networks can approximate smooth targets on compact sets as
width grows. Exact nonrepresentation gives no positive error floor in a usual
training norm, no rate-versus-parameter advantage and no generalization result.
It says nothing about stacked FFNs, a full Transformer, rational activations,
other gated families, or superiority to the static control. Finite-precision
programs are not meromorphic functions; their values/gradients need separate tests.

The proof uses a familiar analytic-continuation technique, and token-dependent
mixtures already have published expressive-separation results. We make no priority
claim for this residual form or proof. Its research value must ultimately be
established by matched fitting and language experiments, memory, runtime and
replication, including the stronger existing controls.
