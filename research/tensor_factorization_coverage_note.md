# Tensor-factorization coverage check while H163 runs

This is a literature/implementation triage note, not a new candidate or GPU result.
The retained Python source search found no explicitly named Tensor-Train/BTT
implementation. That search does not establish that every archived experiment
excluded an equivalent parameterization. Existing literature.md and
interaction_report.md already identify BTT as a required stronger control.

Tensorized dense weights are established prior work, including
[Novikov et al., 2015](https://papers.nips.cc/paper/2015/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html).
Tensor trains with nonlinear input feature maps are also established:
[Stoudenmire and Schwab, 2016](https://papers.nips.cc/paper_files/paper/2016/hash/5314b9674c86e3f9d1ba25ef9bb32895-Abstract.html).
Neither attaching a scalar nonlinearity nor using a tensor product establishes novelty.

[Compute Better Spent, ICML2024](https://arxiv.org/abs/2406.06248) introduces Block
Tensor-Train, containing Monarch matrices, and emphasizes structure-dependent
initialization and learning-rate scaling. Its reported compute comparisons make
BTT a relevant prior-work control. These are the authors' results, not reproduced
results in this repository or proof of lower VRAM for our workload.

## Why the distinction matters

A TT matrix factorizes reshaped weight indices:

W[i1,...,im,j1,...,jm] = sum over internal ranks of
G1[1,i1,j1,a1] G2[a1,i2,j2,a2] ... Gm[a(m-1),im,jm,1].

The raw trainable parameter count is sum_k r(k-1)*n(k)*m(k)*r(k), with endpoint
ranks1, versus product(n)*product(m) for dense weights. This is not ordinary
matrix low rank. For example, a TT-rank1 two-core matrix can be A tensor-product B.
Choosing invertible square A and B gives an invertible, full-matrix-rank product.
Thus low TT rank alone does not impose the same linear bottleneck as W=UV with a
small inner dimension. This elementary fact is not a novelty claim and does not
make low-rank tensor families capable of representing arbitrary dense weights.

It also does not solve activation memory: keeping a wide hidden layer retains
its activation footprint. Tensor contractions, dense weight materialization,
backward temporaries and optimizer storage must all be counted. The exact memory
helper is a separate control; its benefit cannot be assumed additive.

## Decision

Do not launch a nominally new tensorized activation variant. First inspect the
paper's official implementation and establish which BTT structures are distinct
from the repository's BlockShuffle control, including parameterization and
initialization/LR scaling. Only then freeze a small capacity-and-whole-job-memory
comparison against wide dense and matched narrow dense baselines. No new GPU
experiment was started for this note; H163's fixed schedule remains unchanged.
