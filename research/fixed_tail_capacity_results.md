# H151: fixed mixing locks the asymptotic linear map

**A family-wide capacity obstruction is established.** The H147–H150 reversible
stack cannot learn an arbitrary linear tail by changing its scalar shapes and
biases. An exact finite-distribution witness forces normalized MSE>=9/4 for every
allowed setting. Six saved candidate models corroborate the decomposition and
finite-sample bounds. No GPU work or training was needed.

This is a limitation of the current fixed-mixing predictor, not a rejection of
reversible representations, compact-domain learning, or architectures with a
learnable outer map. Resource gains alone do not resolve this capacity issue.

## Derivation

Write each layer as h_j=Q_j*h_(j-1)+b_j+r_j, where z_j=Q_j*h_(j-1)+b_j and
r_j=a_j*z_j/(1+abs(z_j)) coordinatewise. Since abs(r_ji)<=abs(a_ji), unrolling gives

    F(x)=A*x+beta+R(x),  A=Q_L...Q_1,
    beta=sum_j (Q_L...Q_(j+1))*b_j,
    ||R(x)|| <= B=sum_j ||Q_L...Q_(j+1)||_op*||a_j||.

For orthogonal Q and abs(a_ji)<=c/L, B<=c*sqrt(d), regardless of biases.
Thus for every direction v and every finite learned parameter setting,

    lim_(t->infinity) F(t*v)/t = A*v.

A is fixed before training. If target M differs from A, choose v with (A-M)v!=0:
error along t*v diverges as t grows, since beta+R stays bounded. Uniformly bounded
error on all of R^d is impossible for such a linear target. Bias magnitude may
move where this behavior appears but cannot change the limiting map.

For finite-second-moment X with Cov(X)=sigma^2*I, center F(X)-MX. Reverse triangle
in the Hilbert space of square-integrable random vectors gives

    E||F(X)-MX||^2
      >= Var(F(X)-MX)
      >= max(sigma*||A-M||_F - sqrt(Var(R(X))), 0)^2
      >= max(sigma*||A-M||_F - B, 0)^2.

Var denotes total centered second moment. The translation beta disappears.
The inequality is uniform over all allowed theta and all finite biases.
No Gaussian assumption is needed. For a finite sample, use its empirical
distribution: replace sigma*||A-M||_F by RMS((X-meanX)*(A-M)^T).
These are lower bounds, not claims that the architecture attains them.

## Exact counterexample

Take d16, L8, c2, Q_j=I, M=-I. Let X be uniform on the32 vectors +/-16*e_i.
Then mean(X)=0, Cov(X)=16I, target energy is256, and RMS((A-M)X)=32.
The maximal nonlinear correction bound is B=2*sqrt(16)=8. Consequently

    MSE >= (32-8)^2 = 576,
    MSE / E||MX||^2 >= 576/256 = 9/4.

The zero predictor has ratio1, so every model in this particular constrained
family is provably worse on this distribution. Integer covariance/energy checks
and an independent rational-arithmetic audit verify the witness exactly.
The argument also applies to any exactly orthogonal fixed A with target -A;
Q_j=I makes the numerical certificate entirely integer-based. This single
counterexample does not establish a universal performance ranking.

## Saved-model checks

All six H150 learned:fused repeat0 checkpoints are included. Fresh256x384 Gaussian
samples, scales1/4/8, two targets: the original independent orthogonal target and
-A. Eighteen CPU FP64 forward passes produce36 comparisons. These are diagnostics
of saved models, not fresh fits or unbiased generalization estimates. Normalized
bounds and observed MSE divide by each sample's zero-predictor target energy.

Stored FP32 Q matrices are approximately orthogonal. Their numerical bound uses
sqrt(1+||Q^TQ-I||_F), products of these layer bounds, and1e-12 padding. The general
nonorthogonal theorem above applies; computed bounds are FP64 estimates, not
formally rounded interval certificates. The exact counterexample does not depend
on those estimates. All pointwise residual/decomposition checks pass. Independent
NumPy auditing uses covariance traces rather than the worker's matrix-output RMS.

| Source batch | Seed | Input scale | Target | Normalized lower bound | Observed normalized MSE |
|---:|---:|---:|---|---:|---:|
| 128 | 223 | 1 | independent | 0.0000 | 1.9339 |
| 128 | 223 | 1 | opposite | 0.0000 | 3.7985 |
| 128 | 223 | 4 | independent | 0.8334 | 1.9630 |
| 128 | 223 | 4 | opposite | 2.2373 | 3.9077 |
| 128 | 223 | 8 | independent | 1.3527 | 1.9796 |
| 128 | 223 | 8 | opposite | 3.0483 | 3.9469 |
| 128 | 239 | 1 | independent | 0.0000 | 1.9460 |
| 128 | 239 | 1 | opposite | 0.0000 | 3.8457 |
| 128 | 239 | 4 | independent | 0.8272 | 1.9638 |
| 128 | 239 | 4 | opposite | 2.2379 | 3.9282 |
| 128 | 239 | 8 | independent | 1.3444 | 1.9760 |
| 128 | 239 | 8 | opposite | 3.0482 | 3.9585 |
| 128 | 251 | 1 | independent | 0.0000 | 1.9024 |
| 128 | 251 | 1 | opposite | 0.0000 | 3.7360 |
| 128 | 251 | 4 | independent | 0.8345 | 1.9522 |
| 128 | 251 | 4 | opposite | 2.2375 | 3.8832 |
| 128 | 251 | 8 | independent | 1.3542 | 1.9745 |
| 128 | 251 | 8 | opposite | 3.0487 | 3.9332 |
| 2048 | 223 | 1 | independent | 0.0000 | 1.9150 |
| 2048 | 223 | 1 | opposite | 0.0000 | 3.7791 |
| 2048 | 223 | 4 | independent | 0.8370 | 1.9626 |
| 2048 | 223 | 4 | opposite | 2.2423 | 3.9062 |
| 2048 | 223 | 8 | independent | 1.3555 | 1.9806 |
| 2048 | 223 | 8 | opposite | 3.0514 | 3.9464 |
| 2048 | 239 | 1 | independent | 0.0000 | 1.9278 |
| 2048 | 239 | 1 | opposite | 0.0000 | 3.8382 |
| 2048 | 239 | 4 | independent | 0.8245 | 1.9567 |
| 2048 | 239 | 4 | opposite | 2.2408 | 3.9293 |
| 2048 | 239 | 8 | independent | 1.3397 | 1.9693 |
| 2048 | 239 | 8 | opposite | 3.0499 | 3.9593 |
| 2048 | 251 | 1 | independent | 0.0000 | 1.8850 |
| 2048 | 251 | 1 | opposite | 0.0000 | 3.7202 |
| 2048 | 251 | 4 | independent | 0.8345 | 1.9507 |
| 2048 | 251 | 4 | opposite | 2.2369 | 3.8825 |
| 2048 | 251 | 8 | independent | 1.3540 | 1.9740 |
| 2048 | 251 | 8 | opposite | 3.0475 | 3.9331 |

At scale8, independent-target estimated lower bounds range from
1.3397 to1.3555, above the zero predictor's1.
These bounds hold for every allowed shape/bias setting with these fixed matrices
in the exact-real model; the displayed numerical values use the stated FP64
estimates. Scale1 bounds can be zero and do not rule out good compact-domain
performance. This scale stress does not prove failure on normalized language data.

## Consequence for the next architecture

Do not interpret6,144 trainable scalars plus fixed Q as a general substitute for
learnable FFN weights. Stop refining this exact predictor solely for timing.
The H149/H150 resource evidence remains useful, but the fixed tail map needs a
separate capacity mechanism before making broad replacement claims.

One concrete escape is a learned final linear readout. At theta=0 the stack is
A*x+beta. For invertible A, choose readout W=M*A^(-1) and bias -W*beta; any linear
target Mx is then exactly representable in real arithmetic. At d384, this adds
147,840 trainable scalars including bias, bringing the stack total to153,984.
It preserves internal activation reconstruction but changes endpoint gradients,
optimizer memory and stability. This proves linear representability only; it
requires new gradient/resource/quality tests. H107's rejected additive full-affine
plus low-rank branch is a different architecture and remains rejected.

Learnable mixing or a learnable tail scale are other changes, but a scalar scale
alone cannot generally correct an arbitrary fixed rotation. An unrestricted
readout can evade H147's contraction bound, so that bound must not be applied
unchanged to the extended predictor.

Prior work on [neural extrapolation](https://arxiv.org/abs/2009.11848) analyzes
linear behavior along rays in ReLU networks. Here the restrictive fact is that
the tail matrix is fixed rather than learned. The decomposition and L2 argument
are elementary; independent derivation does not establish novelty.

All source/previous receipt hashes verify. There were zero optimizer updates,
zero backwards and no CUDA initialization. Maintained modules/defaults are
unchanged. The broader research goal remains open.

[Plan](fixed_tail_capacity_plan.md), [saved checks](../results/fixed_tail_capacity_v1/result.json),
[independent audit](../results/fixed_tail_capacity_v1/audit.json),
[receipt](../results/fixed_tail_capacity_v1/receipt.json).
