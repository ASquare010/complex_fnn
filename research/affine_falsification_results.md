# H096-H098: a stronger affine control changes the interpretation

**The even-feature lead improves aggregate reporting MSE by only 1.724%
over an ordinary affine least-squares fit, missing the frozen 2% material-benefit
threshold.** Close the general nonlinear-benefit claim for this 300-update recipe.
Its earlier 12.92% advantage over narrow GELU remains a correct measurement;
it was an incomplete comparison. No benchmark result or checkpoint is erased.

The lead uses 350,208 trained coefficients, folded to 282,624 for inference.
The affine comparator needs 147,840 coefficients, including its intercept.
The previously measured native inference speed deficit also remains unresolved.

## What survives, and what does not

| Task | Affine MSE | Even-feature MSE | Even / affine |
|---|---:|---:|---:|
| smooth | 0.429837 | 0.447696 | 1.041549 |
| oscillatory | 1.001295 | 1.013469 | 1.012158 |
| multiplicative | 0.864872 | 0.875631 | 1.012439 |
| piecewise | 0.472441 | 0.412893 | 0.873957 |

The even-feature model loses on three tasks and wins on piecewise targets
by 12.60%. Retain that as a scoped
observation, not a general architecture promotion. All three optimization seeds
have a small aggregate win, so this is not a claim that no nonlinear learning
occurred. The specified material-benefit gate fails; the other two gates pass.

Mean seed ratios: 0.982159, 0.983821, 0.982293.
These are optimization/sampling seeds on one fixed dataset, not three independent
datasets. All comparisons are post-selection diagnostics on previously used data.

## Full conventional comparison

Ratio below one favors the neural recipe; above one favors the affine fit.

| Neural recipe | Aggregate MSE / affine |
|---|---:|
| antipodal | 1.029671 |
| core_linear | 1.536272 |
| duplicate | 0.982757 |
| full_gelu | 1.078652 |
| full_swiglu | 1.302885 |
| gelu_offset | 1.083254 |
| hermite | 1.023549 |
| linear | 1.560260 |
| narrow_gelu | 1.128588 |
| narrow_relu | 1.152332 |
| narrow_swiglu | 1.427213 |
| rational | 1.038496 |
| raw_gelu | 1.043077 |
| trig | 1.047253 |

The large bias-free smooth-target error was a warning. Adding an intercept to
least squares changes its mean MSE from 2.229781 to 0.429837. A nonlinear feature
can provide a nonzero mean, so comparison only with a bias-free linear baseline
does not isolate nonlinear expressivity. This is a measured control gap, not a
proof that every neural difference is caused by its treatment of the mean.

## Structural limit and observed errors

For an even basis E, the model f(x)=A*x+B*E(U*x) has odd component A*x.
The direct matrix can fit linear trends, but no width increase in that even bank
can produce a nonlinear odd response in a single such layer. This extends the
[existing parity analysis](ungated_blockshuffle_theory.md) to the folded lead.
It says nothing about biased/deeper networks or a complete Transformer.

Antithetic evaluation pairs x with -x and splits the squared error into odd
and even components. This separate diagnostic does not replace the original
reporting split metric.

| Task | Even-feature odd error | Even-feature even error |
|---|---:|---:|
| smooth | 0.117372 | 0.330281 |
| oscillatory | 1.008220 | 0.005258 |
| multiplicative | 0.723702 | 0.151538 |
| piecewise | 0.024713 | 0.388297 |

For the cyclic multiplicative target, the known population odd-error floor
is 12/17 of zero-predictor error under the specified independent uniform
distribution. That is not an exact lower bound on this finite report set.

## Experimental scope and verification

We fit 24 linear/affine baselines and 12 constant estimates. For each seed, the
fit uses exactly the multiset of 76,800 samples seen by its 300 batch-256 updates,
including duplicate counts. No unsampled row has positive weight. Selection and
reporting labels never enter the solve. No new neural optimizer update occurs.

The least-squares solver differs from AdamW, so this tests a stronger function
class baseline rather than an equal-optimizer training comparison. FP64 weighted
statistics are computed on CUDA; the recovered small matrix solve uses CPU.
FP32-exported coefficients supply the primary error comparison. The affine
model is a diagnostic control, not the proposed nonlinear FFN replacement.

Independent audit reconstructs the sample counts and weighted statistics,
verifies all 24 solutions, checks 96 CPU scores, recomputes all 168 neural
parity evaluations and rechecks 12 saved odd-linearity tensor pairs. Mean,
median, sample variance and standard deviation are retained in the result JSON.

H096 ended with native access violation 0xC0000005 before saving a completed fit.
H097's isolated CPU/CUDA small-matrix probe passed; it did not reproduce or
diagnose the failure. H098 is a separately frozen recovery using CPU small-matrix
linalg, identical data/tolerances/decisions and phase logging. The old source,
stale RUNNING status, terminal-process receipt and empty fit directory remain
preserved. The backend change is not a proven explanation of the crash.

## Next research direction

Do not optimize kernels for this weak general lead. The next assay needs to
measure improvement on the nonlinear residual after accounting for affine
structure, with an explicit representable positive control. Include established
strong nonlinear controls such as squared ReLU
([Primer](https://arxiv.org/abs/2109.08668)) and its learned scale/bias relatives
([StarReLU](https://arxiv.org/abs/2210.13452)). These published mechanisms are
prior art, not new local results or a justified claim of novelty.

The goal still requires parameter-efficient nonlinear learning, comparable compute,
language NLL, multiple seeds, longer training, scale and broader data. None of
those missing requirements is replaced by a passed least-squares diagnostic.

[Frozen diagnostic plan](affine_falsification_plan.md),
[explicit recovery plan](affine_falsification_recovery_plan.md),
[audited result](../results/affine_falsification_recovery_v1/result.json).
