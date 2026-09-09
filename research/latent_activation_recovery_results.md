# H083 - Internal learned activations: completed fitting result

Both internal curves improve synthetic fitting over plain BlockShuffle, but
neither qualifies: tanh changes aggregate held-out MSE by -5.757%
and sine by -7.069%, while the equal-count affine control changes
it by -15.034%. Both curves lose to affine and narrow GELU.
The positive-control assay passes. These fixed recipes are closed; no language
or full-model resource run is earned.

All 192 allocated runs finish, with three optimization seeds, two rates and
600 updates each. Independent verification reproduces all 192 selection scores,
96 selected held-out scores and 36 reset ablations exactly, without updates.
This study provides synthetic function-fitting evidence, not language NLL,
convergence, broad generalization or a breakthrough. The research goal is unmet.

## What changed and why

The curve sits between the two BlockShuffle factors, before their middle
permutation. The existing outer SwiGLU stays in place. Eight groups share two
controls each, independently for up/gate/down: 48 extra parameters per FFN.
Each curve starts at the identity, sharing every ordinary factor and initial
output with plain BlockShuffle. The equal-count affine control separates a
curved function from the option to learn gain and offset.

For a = 0.5 tanh(theta_a) and s = exp(log(4) tanh(theta_b)):

    phi(z) = z + a*s*q(z/s)^2, where q is tanh or sin.

The affine control is phi(z) = (1 + 0.5 tanh(theta_a))*z + 0.5 tanh(theta_b).
Tanh's real scalar input derivative is bounded by about [0.6151, 1.3849]; sine's
by [0.5, 1.5]. These are local scalar bounds. They do not remove rank limits of
the factors or guarantee whole-network gradients, convergence or GPU speed.
The sine variant is related to [Snake](https://arxiv.org/abs/2006.08195).
The [theory note](latent_activation_theory.md) gives exact assumptions and the
constrained quadratic Bezier identity; no new-family priority is established.

## Held-out quality

Aggregate ratios are geometric means of 12 paired task/seed MSE ratios. Lower
is better. Each rate is selected using a separate 4,096-example split, then the
final checkpoint is scored on 4,096 held-out examples. Rows below the table are
arithmetic means over three optimization seeds for each individual task.

| Form | Weights per FFN | MSE change vs plain | Ratio vs affine | Ratio vs narrow GELU |
|---|---|---|---|---|
| BlockShuffle plain | 350,208 | +0.000% | 1.176937 | 1.253576 |
| Internal affine | 350,256 | -15.034% | 1.000000 | 1.065118 |
| Internal tanh curve | 350,256 | -5.757% | 1.109177 | 1.181404 |
| Internal sine curve | 350,256 | -7.069% | 1.093737 | 1.164958 |
| Narrow SwiGLU | 350,208 | +3.728% | 1.220809 | 1.300305 |
| Narrow GELU | 350,208 | -20.228% | 0.938863 | 1.000000 |
| Full SwiGLU | 1,179,648 | -8.784% | 1.073560 | 1.143468 |
| Full GELU | 1,179,648 | -30.684% | 0.815800 | 0.868923 |

The curves improve over plain in every seed's four-task aggregate. However,
tanh is 10.918% worse than equal-count affine and
sine is 9.374% worse. Both beat narrow SwiGLU but fail to beat
narrow GELU. Affine also remains 6.512% worse than narrow GELU
in the aggregate; it is a diagnostic control and is not promoted by this plan.

| Form | Smooth | Oscillatory | Multiplicative | Piecewise |
|---|---|---|---|---|
| BlockShuffle plain | 0.704784 | 1.004261 | 0.982970 | 0.722216 |
| Internal affine | 0.518223 | 1.003295 | 0.918279 | 0.548499 |
| Internal tanh curve | 0.671405 | 1.004561 | 0.970565 | 0.605573 |
| Internal sine curve | 0.638929 | 1.004446 | 0.964967 | 0.605157 |
| Narrow SwiGLU | 0.827029 | 1.014603 | 1.000728 | 0.692733 |
| Narrow GELU | 0.535388 | 1.007098 | 0.899654 | 0.419473 |
| Full SwiGLU | 0.664574 | 1.040876 | 0.991823 | 0.507050 |
| Full GELU | 0.481466 | 1.019068 | 0.894863 | 0.264185 |

Full GELU satisfies the required positive-control improvement on at least two
tasks. Absolute error and the zero-predictor comparisons are retained in the
[result JSON](../results/latent_activation_recovery_v1/result.json). Normalization
divides by training population standard deviation without centering; these are
variance-scaled MSE values, not centered NMSE or language loss.

| Frozen gate | Tanh | Sine |
|---|---|---|
| complete finite grid | PASS | PASS |
| positive control assay passes | PASS | PASS |
| two percent over plain | PASS | PASS |
| every seed better than plain | PASS | PASS |
| one percent over equal count affine | FAIL | FAIL |
| beats both calibrated narrows | FAIL | FAIL |
| per task regression cap vs plain | PASS | PASS |
| per task regression cap vs zero | PASS | PASS |
| seventy percent reduction | PASS | PASS |

Both curves satisfy the count, finite, per-task regression and plain-improvement
gates. Each fails two quality gates. No threshold or training budget is changed
after looking at the result.

## Did the learned functions matter?

Yes, the final checkpoints depend on the corrections. Resetting curved amplitude
to zero, or both affine controls to zero, worsens aggregate held-out error. All
projection weights stay fixed; curved scales stay fixed. Every reset score is
independently reproduced. Values above 1 mean the reset made error worse.

| Learned function | Reset / original MSE | Smooth | Oscillatory | Multiplicative | Piecewise |
|---|---|---|---|---|---|
| Internal affine | 2.451589 | 6.925839 | 1.205165 | 1.546081 | 2.799238 |
| Internal tanh curve | 1.287028 | 1.577298 | 1.000029 | 1.040330 | 1.672070 |
| Internal sine curve | 1.371209 | 2.009124 | 1.000032 | 1.047150 | 1.680295 |

This tests dependence after coadaptation; it does not prove the correction caused
the training gain. For the oscillatory task, curve resets have almost no effect,
despite the sine function's periodic form. A useful periodic scalar does not
automatically learn a useful periodic high-dimensional mapping.

The saved shape diagnostics show no controls crossing the recorded saturation
threshold. Affine's sampled slopes are all below one, approximately 0.543 to
0.726 across selected models. This suggests gain/offset and optimization deserve
separate analysis, but does not identify which caused the benefit. Any follow-up
must isolate those mechanisms and include narrow GELU; this is not an automatic
retry, longer allocation or promotion of affine.

## Compute and memory

| Form | Mean median update ms | Max peak MiB | Mean clipped % | Upper LR / 12 |
|---|---|---|---|---|
| BlockShuffle plain | 5.274 | 240.45 | 0.000 | 3 |
| Internal affine | 7.901 | 241.22 | 0.083 | 12 |
| Internal tanh curve | 9.790 | 242.72 | 0.042 | 12 |
| Internal sine curve | 9.842 | 243.47 | 0.042 | 12 |
| Narrow SwiGLU | 2.722 | 231.55 | 0.000 | 3 |
| Narrow GELU | 2.444 | 228.93 | 0.000 | 3 |
| Full SwiGLU | 2.487 | 244.88 | 0.000 | 3 |
| Full GELU | 2.176 | 247.00 | 0.083 | 6 |

The time column averages each selected run's median update time after its first
50 updates. CUDA is synchronized around every update. Peak allocation is the
maximum across selected runs and includes the standalone GPU dataset cache.
All entries are native FP32 on the RTX 4070 Laptop GPU, TF32 off. These are
sequential laptop timings, not full-Transformer memory, compiled performance or
serving throughput. The extra curve computation roughly doubles plain update
time despite adding only 48 weights. Parameter efficiency alone is insufficient.

All 12 selections for each learned-function form choose the upper tested rate,
0.003. That is a limitation of this fixed two-rate screen. No additional rates
or steps were allocated after observing it.

## Data and budget

The four tasks use 73,728 fixed uniform input vectors of dimension 384:
65,536 training, 4,096 rate-selection and 4,096 reporting examples. A fixed input
permutation breaks the original contiguous group alignment; approximately
86.98% of neighboring target coordinates cross an input group. A fixed output
rotation mixes outputs. The formulas are smooth, oscillatory, multiplicative
and piecewise; none uses an architecture teacher or an easy linear task.

Three optimization seeds share one dataset. Both rates receive exactly 600
updates with batch 256, AdamW betas (0.9, 0.95), zero decay, clipping at one,
and the inherited factor/narrow learning-rate calibration. The whole allocation
is 115,200 updates and 29,491,200 training-example presentations. There are zero
language targets and zero full-model resource workers. No training checkpoint
recomputation or compiler is used in these cells.

The coordinator ran from 2026-09-09T18:19:35.073977+00:00 to
2026-09-09T18:32:38.945092+00:00: 13.06 minutes including preflight,
evaluation, startup and durable artifact writing. Independent verification took
31.47 seconds and presented 786,432 selection plus
540,672 held-out/reset examples, with zero optimizer updates.

## Failure preservation and verification

H082's first preflight stopped at an adapter argument collision, before any
fitting update. H083 changes only the binding mechanism and artifact destination.
All eight unchanged checks pass in the explicit recovery; there are two total
preflight attempts, one explicit preflight repetition and no fitting repetition.
The original failure remains in its [report](latent_activation_results.md).

The independent audit regenerates all inputs, target rotations/permutations,
training-only scales and sampling streams exactly. It verifies every checkpoint,
600-step optimizer history, moment shape/value finiteness, parameter membership,
learning-rate group, initialization hash, final hash and RNG state. It checks
selection order against source and persisted file timestamps. All 128 scientific
sources and the original 27 H082 files remain unchanged. Seven original
observations and six tensor payloads match recovery results; all 120 saved
identity/checkpoint gradient pairs match exactly.

Active scope remains three model folders, six registered variants and nine
recipes. The unchanged active test suite is not replaced by these eight isolated
checks. Negative fitting evidence closes these two fixed recipes without
discarding their measurements or overstating the local derivative proofs.

## Every selected rate

Both rate checkpoints and every training record remain under the study's cells
directory. The table records choices made before held-out scoring.

| Task | Form | Seed 17 | Seed 29 | Seed 43 |
|---|---|---|---|---|
| smooth | BlockShuffle plain | 0.001 | 0.001 | 0.001 |
| smooth | Internal affine | 0.003 | 0.003 | 0.003 |
| smooth | Internal tanh curve | 0.003 | 0.003 | 0.003 |
| smooth | Internal sine curve | 0.003 | 0.003 | 0.003 |
| smooth | Narrow SwiGLU | 0.001 | 0.001 | 0.001 |
| smooth | Narrow GELU | 0.001 | 0.001 | 0.001 |
| smooth | Full SwiGLU | 0.001 | 0.001 | 0.001 |
| smooth | Full GELU | 0.001 | 0.001 | 0.001 |
| oscillatory | BlockShuffle plain | 0.003 | 0.003 | 0.003 |
| oscillatory | Internal affine | 0.003 | 0.003 | 0.003 |
| oscillatory | Internal tanh curve | 0.003 | 0.003 | 0.003 |
| oscillatory | Internal sine curve | 0.003 | 0.003 | 0.003 |
| oscillatory | Narrow SwiGLU | 0.001 | 0.001 | 0.001 |
| oscillatory | Narrow GELU | 0.001 | 0.001 | 0.001 |
| oscillatory | Full SwiGLU | 0.001 | 0.001 | 0.001 |
| oscillatory | Full GELU | 0.003 | 0.003 | 0.003 |
| multiplicative | BlockShuffle plain | 0.001 | 0.001 | 0.001 |
| multiplicative | Internal affine | 0.003 | 0.003 | 0.003 |
| multiplicative | Internal tanh curve | 0.003 | 0.003 | 0.003 |
| multiplicative | Internal sine curve | 0.003 | 0.003 | 0.003 |
| multiplicative | Narrow SwiGLU | 0.001 | 0.001 | 0.001 |
| multiplicative | Narrow GELU | 0.001 | 0.001 | 0.001 |
| multiplicative | Full SwiGLU | 0.001 | 0.001 | 0.001 |
| multiplicative | Full GELU | 0.001 | 0.001 | 0.001 |
| piecewise | BlockShuffle plain | 0.001 | 0.001 | 0.001 |
| piecewise | Internal affine | 0.003 | 0.003 | 0.003 |
| piecewise | Internal tanh curve | 0.003 | 0.003 | 0.003 |
| piecewise | Internal sine curve | 0.003 | 0.003 | 0.003 |
| piecewise | Narrow SwiGLU | 0.003 | 0.003 | 0.003 |
| piecewise | Narrow GELU | 0.003 | 0.003 | 0.003 |
| piecewise | Full SwiGLU | 0.003 | 0.003 | 0.003 |
| piecewise | Full GELU | 0.003 | 0.003 | 0.003 |

[Frozen original plan](latent_activation_plan.md),
[explicit recovery plan](latent_activation_recovery_plan.md),
[independent audit](../results/verification/latent_activation_recovery_analysis_v1.json),
[all measurements](../results/latent_activation_recovery_v1/result.json).
