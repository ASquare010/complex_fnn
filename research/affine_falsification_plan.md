# H096: can the new nonlinear lead beat an ordinary affine fit?

Previous turn classification: progress. Commit93cc0d1 records336 fresh training
runs,672 independently reproduced scores and12 verified post-training folds.
Gold remains unmet; the folded even-feature model is slower than narrow GELU.

## Why this changes the next experiment

H094's core-linear baseline has no intercept. The smooth target has a nonzero
mean, so that control cannot test whether a nonlinear feature merely supplies a
constant. More broadly, the analytic targets have substantial affine components.
A faster early fit to these components is not evidence of learning their complex
residuals. Before refining the even-feature model, try to falsify its nonlinear
advantage with closed-form linear and affine baselines.

H072 already establishes linear odd parts for single bias-free GELU FFNs. The
same restriction holds for H095: if E is even, f(x)=A*x+B*E(U*x) has odd part A*x.
For the multiplicative target ab+2bcd+a^2*d under independent uniform inputs,
the odd residual orthogonal to all linear inputs is2bcd+(a^2-1)d. Its relative
population error floor is12/17. This is an inherited scoped proof, not a new
architecture result or a bound on biased/deeper networks or Transformers.

Squared ReLU is an established control we have not used in these recent screens:
[Primer](https://arxiv.org/abs/2109.08668). Learned scale/bias on squared ReLU is
also prior art, including [StarReLU](https://arxiv.org/abs/2210.13452). They merit
consideration in a later assay with nonlinear positive controls. No such training
is allocated by this diagnostic, and their published gains are not local results.

## Frozen execution

Reuse exactly H094's saved train/selection/report splits. For each of its three
seeds, count the sampled76800 row indices from300 updates of batch256. Fit each
baseline to that multiset, including duplicates: no unsampled training row has
positive weight. Thus inputs and labels seen match each corresponding gradient
run. The solver and compute schedule differ explicitly from AdamW; this is a
strong function-class control, not an equal-optimizer benchmark.

Fit24 models: four tasks times three seeds times bias-free linear / affine.
Also record the12 weighted-mean constant predictors. Linear has147456 learned
coefficients; affine has147840, respectively87.50%/87.4674% below full FFN weights.
No ridge tuning, rate search, new neural training or optimizer update. Neither
selection nor reporting labels enter the solve.

Use FP64 weighted sufficient statistics on CUDA in4096-row chunks. Augment inputs
with one for affine fits. Require positive-definite Gram matrix, condition number
below100 and relative normal-equation residual below1e-10. Solve by Cholesky and
independently solve the saved small normal system on CPU; require coefficients
within atol/rtol1e-9. This checks the unique least-squares solution for the recorded
statistics; independent chunked reconstruction below verifies the statistics.

Evaluate both FP64 and FP32-exported weights on4096 selection/report rows.
Use FP32 reporting error for comparison to H094. Independent CPU FP64 rescoring
must agree with FP64 GPU error within1e-10; CPU FP32 within1e-6. Save coefficients,
Gram/cross-products, sample counts, residuals, condition numbers, outputs and hashes.
Record solve wall time and GPU peak memory without equating them to AdamW updates.

For all168 previously selected H094 models, evaluate the reporting inputs x and
their negatives. Construct exact analytic targets at-x using the original
permutation, output rotation and fixed training scales. Separate prediction and
target into odd/even parts. The mean of losses on x and-x equals odd-error plus
even-error energy; require absolute/relative1e-6 accounting agreement. For all12
duplicate-even checkpoints verify their odd predictions against the stored direct
linear matrix at atol/rtol1e-5, saving probe outputs. No synthetic symmetry score
is substituted for the original independently audited reporting metric.

One GPU worker, UV, four CPU threads, TF32off. Freeze this plan, source and H094/H095
anchors before numerical execution. Stop on first failure; preserve partial files
and do not automatically retry or relax tolerances. Independently regenerate
counts/statistics, check24 solves, rescore, verify saved parity pairs and recompute
aggregate ratios before publishing the result.

## Decision

The even-feature recipe must reduce geometric-mean reporting MSE by at least2%
versus the matched-sample affine fit, improve every seed's aggregate error, and
have no task mean more than5% above affine. Otherwise close the current claim of
a demonstrated material nonlinear benefit at this budget and do not allocate
kernel refinement or a longer run to it. Preserve any per-task gain as scoped
evidence. Passing this diagnostic would still not cure its speed deficit or earn
language promotion automatically.

If affine fits match or outperform the neural recipes, redesign the next cheap
assay to measure learnable nonlinear residuals and include an explicit positive
control that can represent them. Do not recycle the same early affine fitting
advantage as a new breakthrough.
