# Output modulation: a shared latent bottleneck remains

Let x~N(0,I_d), U have rank at most h, and

\[
F(x)=b(Ux)+\operatorname{diag}(g(Ux))x,
\]

where b and g are arbitrary measurable vector functions with finite squared risk.
Affine offsets inside Ux can be absorbed into b and g. The target is Mx.
This is a relaxation of a finite shared-feature neural implementation.

## Reduction to diagonal plus low rank

Let Q be the orthogonal projector onto the row space of U and P=I-Q.
Write x=m+n, m=Qx, n=Px. Gaussianity makes n independent of Ux, with conditional
mean zero and covariance P; m is determined by Ux. Conditional on z=Ux, set
D(z)=diag(g(z)). Then

\[
\mathbb E[\|Mx-F(x)\|^2\mid z]
=\|Mm-b(z)-D(z)m\|^2+\|(M-D(z))P\|_F^2.
\]

For any gate, an unrestricted b can make the first term zero. The second term
is a fixed deterministic quadratic in the entries of D; its coefficients do not
depend on z. Varying D with z cannot improve on a constant minimizing diagonal D.
One minimizer has

\[
D_{ii}=\frac{(MP)_{ii}}{P_{ii}}\quad(P_{ii}>0).
\]

If P_ii=0, row i of P and (MP)_ii vanish; set D_ii=0 without changing the risk.
The optimal relaxed predictor for this U is

\[
D x+(M-D)Qx.
\]

The second matrix has rank at most h. Conversely every diagonal-plus-rank-h
map D+L belongs to the relaxed family: factor L=BU, take b(z)=Bz and a constant
gate diag(D). Taking infima proves

\[
\boxed{\inf_{U,b,g}\mathbb E\|Mx-F(x)\|^2
=\inf_{D\text{ diagonal},\,\operatorname{rank}(L)\le h}\|M-D-L\|_F^2.}
\]

This argument uses conditional variance, not a Jacobian-at-zero approximation.
It is valid for any finite-risk measurable activation functions of that same
latent Ux, so richer scalar activations cannot remove this particular obstruction.

## An exactly solved target

For even d, let J be a direct sum of [[0,-1],[1,0]]. J is skew-symmetric and
orthogonal; all its singular values are one. Let S(A)=(A-A^T)/2. Orthogonal
projection in Frobenius norm gives

\[
\|J-D-L\|_F^2\ge\|J-S(L)\|_F^2.
\]

Because rank(S(L))<=2 rank(L)<=2h, the rank-2h approximation bound gives
||J-S(L)||_F^2 >= max(d-2h,0). This uses the classical Eckart-Young result.

For each of k=min(h,d/2) disjoint coordinate pairs choose

\[
D_{pair}=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\qquad
L_{pair}=\begin{bmatrix}-1&-1\\1&1\end{bmatrix}
=\begin{bmatrix}-1\\1\end{bmatrix}\begin{bmatrix}1&1\end{bmatrix}.
\]

Their sum is exactly J_pair, and L_pair has rank one. Set both matrices to zero
on the other pairs. This gives rank(L)=k and squared error d-2k, attaining the
bound. Hence the relaxed optimum is exactly max(d-2h,0). The normalized error,
divided by E||Jx||^2=d, is max(1-2h/d,0).

At d=384,h=128, **at least one third of the target energy remains as population
MSE** even after optimizing U and allowing arbitrary nonlinear b and g. This is
not an NLL bound and not a finite-sample reporting-error guarantee. At h>=192 the
bound becomes zero; this removes the obstruction but proves no training success
or attainability with that many fixed GELU features.

## Why full output rank is insufficient

A constant all-one gate with b=0 gives F(x)=x, with full output covariance rank
even when U is zero. Thus full output rank alone says little about learning a
general rotation. H143's rank witness is valid but was never a learning guarantee.
The proposed shared-feature implementation

\[
z=\operatorname{GELU}(Ux+a),\quad F(x)=Bz+c+x\odot(Cz+e)
\]

is contained in the relaxed class and inherits its lower bound. It has 3dh+h+2d
trainable scalars, no repeated FFN invocation and no solver. The restriction
comes from the shared latent observation, not from insufficient flexibility in
the final scalar activation. Extra independently observed latent features or
depth change the problem and need new resource and learning checks.

## Prior art and novelty limits

[FiLM](https://arxiv.org/abs/1709.07871) establishes featurewise affine modulation.
[Multiplicative Filter Networks](https://openreview.net/pdf/e0702b90f0766df82135fc5e14a2df510e9aa9d5.pdf)
studies multiplicative function representations. Neither featurewise gating nor
multiplication is a novelty claim here.

[Chen et al., 2026](https://arxiv.org/html/2605.05659v1) studies diagonal-plus-low-rank
networks, including expressivity gained by changing width or depth. Its broad
approximation results do not imply that one fixed-width shared-latent module can
approximate every linear target. This note addresses a narrower fixed-module,
Gaussian population problem and uses standard conditional expectation and rank
approximation. Independent derivation is not evidence of novelty; a complete
priority search for this exact equivalence has not been established.
