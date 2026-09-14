# Official BTT comparison and balanced-factor hypothesis

Read-only source inspected at upstream commit
452376450cf99667f65a07d5c5373dd559f7dd3d. Two official files and their Apache2.0
license are retained as text in references/btt_official; none was executed.
[Official builder](https://github.com/shikaiqiu/compute-better-spent/blob/452376450cf99667f65a07d5c5373dd559f7dd3d/nn/cola_nn.py),
[operator](https://github.com/shikaiqiu/compute-better-spent/blob/452376450cf99667f65a07d5c5373dd559f7dd3d/ops/operators.py).

For input m0*m1 and output n0*n1, two-core BTT stores tensors with shapes
(m1,m0,r*n0) and (n0,r*m1,n1). The intermediate has r*m1*n0 channels and the
core count is r*m1*n0*(m0+n1), excluding bias and two optional normalization
scalars. Consequently rank(W)<=min(input,output,r*m1*n0). This follows from
factoring the linear map through that intermediate, irrespective of training.

Local BlockShuffle uses two grouped matrices and fixed permutations, intermediate
min(input,output), and the same group count for both factors. BTT's per-core block
counts depend on its dimension factors and its explicit rank. A rank1 BTT can
coincide with a Monarch arrangement, but naming it BTT does not establish a new
model. Local orthogonal initialization differs from the official Gaussian core
initialization, structure-dependent learning-rate multipliers and optional RMS
normalization with learned scalar gains. A fair reproduction must resolve these.

## Exact shape accounting, batch16/context512/FP32

| Projection | Factor choice | Rank | Core parameters | Intermediate channels | One intermediate MiB |
|---|---|---:|---:|---:|---:|
| 512 to608 | official greedy | 1 | 23,552 | 256 | 8 |
| 512 to608 | official greedy | 2 | 47,104 | 512 | 16 |
| 608 to512 | official greedy | 1 | 48,640 | 1,216 | 38 |
| 608 to512 | official greedy | 2 | 97,280 | 2,432 | 76 |
| 512 to608 | closest factors | 1 | 29,184 | 608 | 19 |
| 608 to512 | closest factors | 1 | 26,112 | 512 | 16 |

The official greedy algorithm gives608=8*76. Closest factors give608=19*32;
512=16*32 in both. An initial assertion incorrectly expected16*38; the assertion
caught it, the original script is preserved, and the corrected CPU calculation
passes. No GPU experiment or research gate was changed.

Rank1 balanced up/down cores total55,296 versus622,592 dense weights for this
GELU FFN, before optional gains. The exact reduction is1-55296/622592, about91.12%. It is parameter accounting only.
Neither activation retention nor peak VRAM follows directly from these counts.
Two forward intermediates may overlap other tensors; normalization and backward
also allocate memory. Runtime and task quality remain unmeasured.

For balanced rank1, full matrix rank is attainable, not merely unexcluded:
write x[g,i]. Up: choose R[g,i,b]=1 when b=i (i<16), and L[b,g,a]=1 when a=g.
Then y[b,a]=x[a,b] for b<16, with three extra zero output groups. Down: use the
same selectors for b<16 while i ranges over19, selecting16 input coordinates per
group. These are rank512 injection/surjection maps. Full matrix rank still does
not imply arbitrary dense-map capacity or good learned nonlinear approximation.

## Candidate decision

A useful next ablation is balanced versus official greedy dimension factorization
with BTT itself treated as prior art. Hypothesis: avoiding an accidental
intermediate bottleneck improves learning while reducing the down-projection's
workspace. First verify contractions and gradients, then test wide dense, matched
narrow dense, existing BlockShuffle, and both BTT factorizations with fair tuning.
A parameter count or rank witness alone earns no performance promotion.

The paper's language-model advantage is partly attributed to classifier-head
compute, so an FFN-only replacement needs its own evidence. The paper also reports
weight normalization as necessary for stable larger-transformer experiments.
[Paper, sections5.1 and5.3](https://arxiv.org/html/2406.06248v1).
H163 continues independently; this note adds no GPU work.

## CPU algebra verification

[Contraction check](references/btt_official/contraction_check.py) compares three
small float64 rank1/rank2 contractions to explicit dense multiplication and checks
all input/right-core/left-core VJPs with directional finite differences. Maximum
dense absolute error3.56e-15; maximum normalized directional error1.36e-10.
Two rank512 selector witnesses verify the injection/surjection count. These are
mathematical implementation checks, not trained models or resource benchmarks.
[Machine-readable checks](references/btt_official/contraction_check.json).
