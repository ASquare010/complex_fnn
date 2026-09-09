# H079 - BLAST comparator mathematics and limits

BLAST represents canonical blocks as U_i diag(s_ij) V_j^T, sharing row and
column bases. Its factors can be trained directly; its matrix-compression descent
theorem is a different statement from Transformer optimization. The rectangular
factorization and inclusion of block low-rank structures are established prior
art. We use these facts to implement a comparator, not to claim a new architecture.
[BLAST, sections 2-3 and Appendix A](https://arxiv.org/html/2410.21262v1)

The implementation, initialization and certificates below are our explicit local
derivation. They do not reproduce the paper's complete training recipe.

## Factor count and computation

Let m/n be output/input widths, b the number of groups and r the latent rank.
Store U as [b,m/b,r], V^T as [b,r,n/b], and S as [r,b,b]. Parameter count is
r(m+n+b^2). Encode each input group, mix groups independently for each rank,
then decode output groups. Three batched matrix products perform these steps;
no [b,b,tokens,r] broadcast product is needed. Actual allocations and runtime
remain to be measured. MAC count per token equals parameter count; FLOPs count
a multiply/add pair as two. Rearrangements and nonlinearities are excluded.

At d384/b8/r48, GELU h3200 uses two projections and SwiGLU h1984 uses three.
Both have 350,208 weights/FFN and 2,801,664 over eight layers. Adding the fixed
6,297,984 non-FFN weights gives 9,099,648 total. These are exact model-count
calculations, not a trained Transformer. Both retain the existing 70.3125% FFN
and 42.1700% total reduction. The FFN wrapper applies the same output-unshuffle
as the existing BlockShuffle projection after every canonical BLAST output.

## Explicit same-width embedding

Undo BlockShuffle's final fixed output permutation. With intermediate width
384 and b=8, its canonical block (i,j) has factors L_ij R_ij^T of rank at most
6. For each ell in 0..5 assign k=6*((i+j) mod 8)+ell. Put the corresponding
column of L_ij in U_i[:,k], the corresponding row of R_ij^T in V_j^T[k,:],
and set S[k,i,j]=1, with other entries zero.

For fixed i, the eight j values use disjoint slices of U_i. For fixed j, the
eight i values use disjoint slices of V_j. Thus there is no conflicting factor
assignment, and summing the six selected components gives each original block
exactly. This proves inclusion at the SAME dimensions and fixed partitions.
The ordinary BlockShuffle formula gives L_ij=second[i,:,j::8] and
R_ij^T=first[j,6*i:6*(i+1),:]. The certificate checks these actual tensor indices.

BLAST pays 3,072 extra coupling weights/projection at those dimensions. The
matched-budget BLAST FFNs reduce hidden width by 64. Their nonlinear families
therefore do NOT inherit this same-width inclusion statement. Learned activation
behavior, arbitrary permutations and deeper compositions are outside this proof.

## Exact strictness witness

Let H8 be the unnormalized Sylvester Hadamard matrix, H8^T H8=8I. Set every
rank's group-coupling matrix to H8. Use coordinate embeddings for U and V so
every canonical block contains a signed 48-by-48 identity and otherwise zeros.
The effective map is H8 tensor I48, padded on the larger width. It has 384
nonzero singular values sqrt(8); every one of 64 blocks has rank 48.

A rank-6 approximation of any block must discard at least 42 unit singular
values. The blocks occupy disjoint entries, so errors add: squared error is at
least 64*42, while squared target norm is 64*48. The squared relative Frobenius
error is therefore at least 7/8. Equality is possible for the unconstrained
rank-6-per-block class. This is a matrix approximation obstruction with a fixed
canonical partition, not a task loss floor or a nonlinear approximation theorem.
With all couplings one, all group blocks instead share a rank-48 global product;
the explicit witness has rank exactly 48. This guards against accidental
low-rank initialization when all group couplings are identical.

## Initial partial isometry

Assume r=min(m,n)/b. Give each U_i orthonormal columns and V_j^T orthonormal
rows. Every rank's b-by-b coupling is orthogonal. The middle operation is a
permutation-conjugated direct sum of orthogonal matrices, hence orthogonal.
For an up projection, the input block factor is square orthogonal and the
output block factor has orthonormal columns; their composition has orthonormal
columns. For down, transpose this argument: the composition has orthonormal rows.

Multiplying each of the three factors by g^(1/3) scales the map by g. Therefore
A^T A=g^2 I (up) or A A^T=g^2 I (down). Set
 g=.02*sqrt(max(m,n))*residual_scale,
with scale 1 for up/gate and 1/4 for down. The average row squared norm becomes
.02^2*n*residual_scale^2, matching the dense initialization's expected value.
Name-local seeded QR generates each factor; this is our calibration choice.

The identities hold at initialization in exact arithmetic, with numerical
verification at the stated tolerances. Down still annihilates its orthogonal
nullspace. Learned factors may become singular; activations can have small
Jacobians and SwiGLU's bias-free origin Jacobian remains zero. There is no
whole-network gradient lower bound, optimizer theorem or convergence guarantee.

[Qualification plan](blast_operator_plan.md),
[previous duration failure](ungated_duration_results.md),
[prior comparator note](ungated_comparator_note.md).
