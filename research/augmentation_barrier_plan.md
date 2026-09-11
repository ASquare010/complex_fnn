# H155: minimum augmentation for an even target

H154 completed 120 audited GPU fits: progress, not a wait or status restatement.
Before implementing a small auxiliary state, test its representational premise.
This is mathematical analysis plus bounded CPU examples, not another learning run.

## Exact dimension statement and proof

Let T: R^d -> R^q be continuous and even, T(x)=T(-x), and suppose its image
has full affine span R^q. Embed x as E(x)=(x,0_k). Consider

    F(x) = W R(E(x)) + b,

where R is a continuous injective map on the embedded input sphere (a global
homeomorphism suffices), and W is a q by (d+k) matrix. Exact representation of T
requires **k >= q**. In particular, for the cyclic product with d=q=32,
adding 1, 4, 8 or 16 auxiliary coordinates cannot remove this exact obstruction.

Proof: full affine output span requires rank(W)=q. Therefore ker(W) has
n=d+k-q dimensions. If k<q, then n<d. Let P contain an orthonormal basis for
ker(W), so P maps the ambient state into R^n. Apply Borsuk-Ulam to the continuous
map x -> P R(E(x)) on a sphere S^(d-1). If n<d-1, pad its output with zeros.
There is an antipodal pair with equal kernel coordinates. Exact even outputs
also give W[R(E(x))-R(E(-x))]=0, so their difference lies in ker(W). Its kernel
coordinates vanish, hence the difference is zero, contradicting injectivity.
The n=0 case is immediate. Thus k>=q.

The topological input is the classical [Borsuk-Ulam theorem, Hatcher,
Corollary 2B.7, printed page 176](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf#page=185).
The architecture-specific dimension argument above is a deduction, not a claim
of new topological theory or established publication novelty.

The bound is sharp in an unrestricted continuous-core family: with k=q,
R(x,z)=(x,z+T(x)) has inverse (x,z-T(x)), and W selects the z coordinates.
This assumes access to T inside the core; it proves neither efficient neural
parameterization nor learnability. [Zhang et al., ICML 2020, Theorem 7](https://proceedings.mlr.press/v119/zhang20h/zhang20h.pdf)
already gives input-plus-output dimensional constructions with a linear readout.
Our claim is a scoped necessity argument for exact even targets, consistent
with that prior sufficiency result. Augmentation itself is established prior art.

## Conditional worst-case error bound

For k<q and full-row-rank W, assume on the radius-r input sphere

    ||R(E(x))-R(E(-x))|| >= 2*m*r > 0.

At the antipodal pair supplied by Borsuk-Ulam, the state difference is in the
row space of W. Hence its output difference is at least 2*sigma_q(W)*m*r.
For any even target, the triangle inequality then gives

    sup_(||x||=r) ||F(x)-T(x)|| >= sigma_q(W)*m*r.

The constants depend on the trained model. Shrinking the smallest readout
singular value or core separation weakens the bound. This is a uniform-error
bound on a sphere, **not a Gaussian MSE or validation-loss lower bound**.
It does not retrospectively explain H154's observed 31% error gap. A nonlinear
readout, noninjective core, extra input bypass, or different-domain problem can
fall outside the assumptions. Transformer residual branches and normalization
must be analyzed separately; this is not a theorem against Transformer use.

## CPU validation frozen before execution

Seeds 389/397/409; input dimension 32. Output/augmentation pairs:
(1,0), (8,0), (8,4), (8,7), (32,0), (32,1), (32,4), (32,16), (32,31).
For each pair generate a seeded FP64 orthogonal ambient matrix Q and take W
as the first q coordinate rows. Find an antipodal witness using the nullspace
of the last d+k-q rows of Q restricted to its first d columns. At radii
0.5/1/2 and readout scales 1/0.01, verify kernel-coordinate equality, unit
retained separation and the conditional error bound (tolerance 1e-10).
27 independent matrices, 162 scaled witness checks. These illustrate the proof;
finite samples cannot prove Borsuk-Ulam for arbitrary nonlinear networks.

Verify the d32 cyclic-product image's full affine span with exact integer
zero/basis witnesses. Demonstrate sufficiency at k=q=32 with the explicit
product shear. FP64 input/parameter finite differences on a d4/q4 shear must
pass. At d32 test inverse recovery on 257 samples at scales 0.001/1/1000,
FP32 and FP64, reporting errors for BOTH recovered blocks separately. Require
FP64 relative block error <=1e-9; FP32 is a descriptive cancellation stress
and may fail that threshold. No claim of reliable arbitrary-scale inversion.
The product oracle is hand-specified solely for a representation witness.

Save arrays and machine-readable results. Independently audit every witness
and inverse result using NumPy, without importing candidate model code. Keep
all original artifacts and source hashes. Zero optimizer updates and no CUDA
initialization. Exact state-byte accounting is illustrative, not peak VRAM:
at batch 2048/d384/FP32, d coordinates take 3 MiB and 2d take 6 MiB.

Decision: do not spend a new GPU sweep claiming that tiny augmentation repairs
exact even-vector capacity. A full auxiliary state or a different output map
needs a separate quality/VRAM experiment. Low Gaussian MSE with fewer dimensions
remains an empirical possibility and is not eliminated by this theorem.
