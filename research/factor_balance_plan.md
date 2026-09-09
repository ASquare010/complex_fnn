# H069 - Internal factor balance diagnosis

Frozen before reading checkpoint factor norms. H068 closes both tested activation
recipes. This distinct zero-update diagnosis concerns the retained plain model's
internal linear factors, not the H057 whole-FFN value/down rescaling or the H052
product-decay intervention. No active source, optimizer or model change is made.

## Exact algebra and limits

Write a structured projection as W = P_out^-1 B P A. Each hidden coordinate i
has incoming row a_i of A and, after P, outgoing column b_i of B. For positive
c_i, multiply a_i by c_i and divide its paired b_i by c_i. Diagonal scaling
commutes through the permutation with permuted entries, so W is unchanged in
real arithmetic. The represented projection's singular values and condition
number therefore cannot improve. Internal factor statistics and raw parameter
gradient norms are coordinate-dependent and can change without a better function.

Let a=norm(a_i), b=norm(b_i), both positive, and s_A,s_B be the existing factor
LR multipliers. Define the diagnostic energy E(c)=a^2*c^2/s_A+b^2/(s_B*c^2).
This is a coordinate-weighted Frobenius energy, NOT an Adam objective, loss,
curvature estimate or proof of bad optimization. Its exact minimizer is
c_star=((b^2/s_B)/(a^2/s_A))^(1/4), with minimum 2ab/sqrt(s_A*s_B).
This follows by a positive square or differentiating in log(c).

Choose c=2^round(log2(c_star)), ties to even. If q=c/c_star, then
q is in [2^-1/2,2^1/2], E(c)/E(c_star)=(q^2+q^-2)/2 <= 1.25.
The post-transform weighted norm ratio (a*c/sqrt(s_A))/(b/(c*sqrt(s_B)))
is in [1/2,2]. Nearest-integer log scaling minimizes E over all integer powers
of two, hence E(c)<=E(1). Numerical checks will allow 1e-12 relative tolerance.
Use FP64 to choose exponents. Any exponent outside [-8,8], zero/nonfinite row or
column norm, or failed bound fails qualification; do not clamp or change rules.

Power-of-two scaling is exact for representable finite floating-point values
away from overflow/underflow, but this alone is NOT a universal bitwise GEMM or
FFN theorem. Demand bitwise equality empirically in the fixed CPU/GPU cases.
Input gradients should be unchanged. Parameter gradients covary: grad_A'=grad_A/c
and grad_B'=c*grad_B; compare after mapping them back. Adam states and finite
updates are not automatically invariant. No optimizer-state transform or training
is attempted here, and no global nonvanishing-gradient claim follows.

Prior art: [Path-SGD](https://papers.neurips.cc/paper_files/paper/2015/file/eaa32c96f620053cf442ad32258076b9-Paper.pdf)
studies rescaling-invariant optimization; [Du, Hu and Lee](https://arxiv.org/abs/1806.00900)
prove balancing properties under specified homogeneous gradient-flow/GD settings.
These are not guarantees for this SiLU FFN with AdamW. [Singh 2026](https://arxiv.org/html/2608.05136v1)
separates orthogonal factor-gauge equivariance, balancedness and optimizer bias;
orthogonal gauge results do not imply diagonal scale invariance. The internal
linear gauge, energy minimizer and power-of-two rounding bound are elementary
applications, not claims of architectural or optimizer priority.

## Fixed allocation

Pin all six existing d384/L8/h2048/G8 plain BlockShuffle checkpoints: seeds
17/29/43 at 800 and 3,200 steps, peak LR .0012. Reconstruct each seed's original
initialization with the current verified source. Nine states x eight layers x
three projections = 216 projection records. Read weights only; preserve source
checkpoints and their optimizer moments byte-for-byte. Do not score corpus data.

For every projection record save the full 384 incoming/outgoing norm vectors,
weighted ratios, chosen exponents and before/after/optimal energies. The actual
multipliers are (4,4) for up/value and (4,64/3) for down. Also report unweighted
norm imbalance descriptively; do not select another balancing rule afterward.
All parameter counts remain 350,208 per FFN / 2,801,664 across eight layers.

Four small CPU harness checks: independent scalar energy/rounding bounds;
FP64 represented-matrix preservation under the actual shuffle; gradient covariance;
and name-seeded initialization matching the existing Transformer and exact counts.
Freeze source/protocol before checks. Do not run optimization in these checks.

Then 24 paired FFN checks: initialization seed17 and all three trained 3,200-step
seeds, layers 0/3/7, each CPU FP32 and CUDA BF16. CPU inputs [2,7,384]; GPU
[16,128,384]. Seed inputs and incoming gradients with 91000+100*training_seed+layer
on CPU, then transfer. Apply the SAME balancing rule to all three projections.
Use the historical native inner SwiGLU recomputation policy on both copies.
Compare output, input gradient and all six factor gradients after inverse
coordinate mapping bitwise; require every recorded value finite. Save raw paired
tensors for these cases, state/input hashes and per-tensor comparisons. No updates.

UV with existing compile/data extras, four CPU threads, TF32 disabled, one GPU
worker. A hidden coordinator records command, PID, UTC, return code and logs.
No automatic retry or source mutation after outcomes. All historical evidence,
active model folders, variants, tests and configurations remain unchanged.

## Decision

Local numerical/theorem checks must all pass. To earn a separately frozen small
optimizer-state/update qualification, EACH of the three trained 3,200-step seeds
must also show >=20% reduction in summed weighted energy across all 24 projections
AND >=10% of its 9,216 channels with initial weighted norm ratio outside [1/4,4].
These thresholds flag a substantial coordinate redundancy; they do not prove it
causes the loss gap. Initial/800-step results are contextual controls only.
Failure closes this tested balancing motivation without an optimizer or learning
allocation. Passing earns only state/update qualification, not fitting or LM
training. Full quality, resource, multi-seed longer training, convergence, scale,
broader data and strong baselines remain required for the research goal.
