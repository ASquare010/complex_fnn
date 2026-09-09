# H070 - One orthogonal stage across the fixed shuffle

This is a proposed structured-projection extension built from known components.
It is not a new butterfly/orthogonal-transform family or a demonstrated FFN win.

## The existing rank restriction

For W=P_out^-1 B P A with G equal blocks and intermediate width k, use canonical
output coordinates P_out W = B P A. For each output block j and input block i,
let r_ji count latent channels from A's block i routed by P into B's block j.
The corresponding matrix block is a sum of r_ji independent outer products,
so its rank is at most r_ji. These constituent rows/columns are independently
parameterized; the restriction is attainable. This is the known block-low-rank
view of [Group and Shuffle, section3](https://arxiv.org/html/2406.10019v1#S3).

With the current transpose shuffle and k divisible by G^2, r_ji=k/G^2 for every
block. At k384/G8 each canonical block has rank at most6. At k64/G8 the bound
is1. At k48/G8 there are only48 paths among64 block pairs: each count is0 or1,
with16 zero blocks. Thus H068's d48 setting preserves width ratios but not the
full-size rank/connectivity pattern. Its negative fitting result remains valid
for that setting and cannot be read as a full-size topology reproduction.
No whole-FFN mixed-derivative restriction follows just from one linear map.

## Minimal orthogonal extension and a redundant control

Apply Q after the fixed intermediate shuffle:

    W_theta = P_out^-1 B Q_theta P A.

Let h=k/2. Pair coordinate p in [0,h) with q=h+((p+shift) mod h). Each pair uses
one trainable real angle and the two-dimensional rotation

    (z_p, z_q) -> (cos(theta_p) z_p - sin(theta_p) z_q,
                   sin(theta_p) z_p + cos(theta_p) z_q).

The pairs are disjoint, so Q is orthogonal in real arithmetic and Q_0=I. Its
input Jacobian is Q and has every singular value1. This statement concerns
only Q: A/B rank, nonlinear gates and whole-model cancellations still affect
gradients. Float32/BF16 coefficient and arithmetic rounding need measurement.

Use shift1 as the candidate and shift0 as a same-cost redundant control. At
k384/G8 and k64/G8, h is divisible by G. With shift0, paired coordinates have
the same input-group origin p mod G. Hence P^T Q P is block diagonal in A's
groups and can be absorbed into unconstrained A without changing the projection
family. The control adds optimization coordinates but no projection expressivity.

With shift1, paired coordinates have different input origins AND different
B output groups. This cannot generally be absorbed into either factor. Both
forms start at the baseline and use the same number of angles and operations.

## Explicit projection separation and approximation obstruction

At n384/m2048/G8/k384, focus on output block0 and input block1 in B Q P A.
The existing six paths have post-shuffle indices1,9,17,25,33,41. Set their six
A rows to input-block basis coordinates0..5 and their B columns to output-block
basis coordinates0..5. Set B's post-shuffle column0 to output coordinate6.
Set A's row feeding post-shuffle coordinate193 (input block1, row24) to input
coordinate6. Set all other factor entries to zero. With shift1 only theta0=pi/4
is nonzero; the pair is (0,193). The canonical (0,1) block is diagonal on its
first seven rows/columns with entries (1,1,1,1,1,1,-sin(pi/4)); all other entries
of the represented matrix are zero.

This block has rank7, while every original projection's same block has rank<=6.
The new projection family strictly contains the old at these dimensions. Moreover,
by the rank6 truncated-SVD bound, the Frobenius approximation error in this block
is at least sin(pi/4)=1/sqrt(2). Since the target norm is sqrt(6.5), every original
projection has relative full-matrix Frobenius error at least1/sqrt(13).
This is a matrix approximation result for fixed canonical groups, not a lower
bound for the complete nonlinear FFN, a learning theorem or universal advantage.
A more expressive projection alone does not prove strict full-FFN separation.

## Parameter and computational scope

Each projection adds k/2 angles. At d384/h2048/G8/L8 this adds576 per FFN,
4,608 overall: FFN2,806,272 and total model9,104,256, retaining70.263671875%
FFN reduction against the full9,437,184 reference. There are no dense new
matrices. Each stage has O(k) arithmetic and a fixed pairing permutation, but
extra launches, saved intermediates and angle derivatives can cost time/memory.
No speed or resource claim follows from this count.

## Prior art

[Group and Shuffle](https://arxiv.org/abs/2406.10019) already covers products of
block matrices and permutations, including orthogonal constructions and block
rank descriptions. [Kaleidoscope](https://arxiv.org/abs/2012.14966) learns structured
linear maps and replacements for hand-designed transformations. Learnable Givens
butterfly angles also appear in [ButterflyQuant](https://arxiv.org/abs/2509.09679).
This proposal is a small, baseline-preserving use of established machinery inside
the retained FFN. The scoped witness and redundancy control explain what changes;
they do not certify novelty or practical superiority. No cited training result
has been reproduced here as evidence for this proposal.
