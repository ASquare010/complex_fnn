# H144: output modulation retains a shared-latent capacity limit

**Capacity obstruction verified; no training was allocated.** The proof and
its eight saved integer certificates, four conditional-risk checks and FP64
input/parameter finite-difference test all pass. CUDA was never initialized.
This is a scoped mathematical result, not a measured VRAM improvement or a
novelty claim.

For standard Gaussian input and linear target Mx, allowing arbitrary measurable
base and gate functions of one shared h-dimensional linear observation gives
exactly the same optimal population error as diagonal-plus-rank-h approximation.
For an explicit pairwise rotation J, that normalized error is **max(1-2h/d,0)**.
At d=384 and h=128, even this generous relaxation leaves **at least one third**
of the target energy as MSE. The finite GELU candidate cannot do better than
its relaxation; this does not assert it attains the relaxed optimum.

[Complete derivation and assumptions](modulated_capacity_theory.md) establish the
population statement. Numerical tests corroborate implementation, not replace
its proof. The result concerns one shared-latent module; it is not a bound for
deep stacks, arbitrary input distributions, independently conditioned gates,
nonlinear targets, finite reporting sets or language-model NLL.

## Explicit candidate and the width tradeoff

The isolated prototype computes z=GELU(Ux+a), then Bz+c+x*(Cz+e), with one feature
bank and two readouts. It needs 3dh+h+2d trainable scalars. Its input/parameter
FP64 gradcheck passes. A constant-gate witness gives identity output and full
output covariance rank with h=1, demonstrating why full output rank alone does
not establish capacity to approximate a general rotation.

| Latent h at d384 | Parameters | Reduction versus wide GELU h384 | Relaxed rotation error floor |
|---:|---:|---:|---:|
| 128 | 148,352 | 49.83% | 0.333333 |
| 192 | 222,144 | 24.87% | 0.000000 |

At h=192 the displayed obstruction disappears, while parameter savings fall
from about50% to about25%. A zero lower bound is not an actual learned solution,
convergence guarantee or proof that 192 GELU features attain the affine optimum.
Fixed-width universal approximation must not be inferred from full rank.

## Exact and independent spectral certificates

| d | h | Constructed rank | Exact squared error | Normalized error | Gaussian Monte Carlo squared error | Estimated standard error |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 0 | 6 | 1.000000 | 6.022871 | 0.038696 |
| 6 | 1 | 1 | 4 | 0.666667 | 4.035189 | 0.031406 |
| 6 | 2 | 2 | 2 | 0.333333 | 2.011596 | 0.021690 |
| 6 | 3 | 3 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 384 | 64 | 64 | 256 | 0.666667 | 256.245068 | 0.251663 |
| 384 | 128 | 128 | 128 | 0.333333 | 127.944883 | 0.175891 |
| 384 | 192 | 192 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 384 | 256 | 192 | 0 | 0.000000 | 0.000000 | 0.000000 |

The integer factors satisfy B^T B=2I and U U^T=2I, certifying their full column/
row ranks without floating tolerances. Their product has the declared rank.
Integer residual squares equal max(d-2h,0) exactly. A separate SVD check confirms
the spectrum and skew-projection lower bound. Gaussian checks use 8,192 samples
per witness and a prospectively fixed six-estimated-standard-error tolerance;
these stochastic checks neither prove nor weaken the exact bound.

## Conditional-risk identity checks

| Seed | Coordinate projection | Optimal affine risk | Nonlinear Monte Carlo risk | Mean conditional prediction | Maximum pointwise identity error |
|---:|---|---:|---:|---:|---:|
| 17 | False | 8.304759 | 8.450498 | 8.447304 | 7.105e-15 |
| 29 | False | 7.871041 | 7.946169 | 7.989385 | 7.105e-15 |
| 43 | False | 7.597136 | 7.715553 | 7.740433 | 7.105e-15 |
| 17 | True | 7.224055 | 7.356604 | 7.341397 | 3.553e-15 |

Three random rank-4 observations in dimension12 and one coordinate projection
check the Gaussian decomposition. The coordinate case includes zero conditional
coordinate variances. Two hundred diagonal perturbations per case verify the
quadratic optimality identity; each nonlinear gate simulation uses 32,768 samples.
Prediction discrepancies pass six estimated standard errors. The conditional
risk calculation is independent of the neural prototype and its autodiff test.

## Decision and provenance

Do not allocate a learning run for h128 on the premise that output modulation
alone repairs general approximation. Its full-rank output witness masks a
remaining shared-observation bottleneck. This does not eliminate modulation for
real FFN data or nonlinear targets. Any next hypothesis must address what input
information the gate/base can independently access, or explicitly accept the
larger latent width and measure its actual memory/compute tradeoff.

No historical failed recipe is reopened. No existing maintained file changed;
all prior receipt and maintained-source hashes verify. The earlier read-only
PowerShell inspection suffered a CLR failure before file creation and succeeded
on the next inspection; all mathematical checks completed in one attempt.
There were zero optimizer updates and zero GPU backwards. The broad research
goal remains open.

Prior art includes [FiLM](https://arxiv.org/abs/1709.07871) and
[diagonal-plus-low-rank expressivity](https://arxiv.org/html/2605.05659v1).
The derivation uses standard Gaussian conditioning and Eckart-Young approximation;
independent derivation does not establish priority or novelty.

[Prospective plan](modulated_capacity_plan.md),
[prototype](../results/modulated_capacity_v1/model.py),
[machine-readable checks](../results/modulated_capacity_v1/result.json),
[evidence receipt](../results/modulated_capacity_v1/receipt.json).
