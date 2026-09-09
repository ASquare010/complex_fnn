# Fixed-support sparse coordinates: claims and boundaries

Let Omega be a fixed set of k distinct coordinates of an m-by-n matrix. Insert
v in R^k at those coordinates and zeros elsewhere to define S_Omega(v).
The coordinate basis matrices are orthonormal in the Frobenius inner product.
Consequently <S(v),S(w)>=<v,w> and ||S(v)-S(w)||_F^2=||v-w||_2^2.
For differentiable L(W), the chain rule gives grad_v L=S^*(grad_W L), with
S^* the coordinate gather. Unlike W=AB, this parametrization has no nontrivial
linear null direction or factor-scale symmetry. Loss curvature, the nonlinear
FFN and the rest of a Transformer may still be poorly conditioned.

For rowwise2:4 support, k=mn/2. Every row has n/2 independent zero-mean values
of variance2/n. Its expected squared norm is1. Conditional on the selected
support, E[(Wx)_i^2]=(2/n) sum_{j selected} x_j^2. Each input is selected with
probability1/2 under a uniform unordered pair, giving E[(Wx)_i^2]=||x||^2/n
after averaging masks. These are initialization expectations, not learned
singular-value, activation-distribution or optimization bounds.

Let H8 be the unnormalized Sylvester Hadamard matrix, H8 H8^T=8 I8.
M=H8 tensor I48 has M M^T=8 I384. Each of its8-by-8 canonical blocks is
either I48 or -I48. A row has at most one nonzero per input quartet, and can be
stored under a2:4 mask that permits the row's residue modulo4 and its cyclic
successor. Unused allowed values are zero. This constructs one admissible mask;
it does not claim the random initialization uses this mask.

For any canonical block A_ij of rank at most6, approximation of +/-I48 has
squared Frobenius residual at least42: its48 singular values are all1 and at
most6 can be retained. Summing over64 disjoint blocks gives residual>=2688,
while ||M||_F^2=3072. Hence the squared relative error is at least7/8.
This applies to the canonical BlockShuffle projection family after undoing its
fixed output permutation. It does not apply to arbitrary changed partitions,
composed nonlinear networks, or arbitrary target tasks. The square sparse
operator has twice the learned count of the square BlockShuffle comparator.

At equal FFN count350,208 and width384, sparse hidden widths608(SwiGLU) and
912(GELU) are twice the respective narrow dense widths304/456. A particular
fixed sparse mask need not contain either narrow dense FFN's function family.
Wider features and this matrix witness motivate experiments, not dominance.

The native reference materializes W and uses dense GEMM. Its learned parameter
and optimizer moment savings are real; its FLOP savings are unproved and its
temporary matrices/indices can increase memory. A separate cached compressed
BF16 inference representation excludes training gradients and the one-time
conversion cost from repeated inference timings. Padding and sparse metadata
are storage, even when they are not learned parameters.

The [frozen plan](compact_sparse_operator_plan.md) attributes the underlying
sparse architecture and defines the numerical checks. These elementary
coordinate and rank arguments are scoped deductions, not novel theorems.
