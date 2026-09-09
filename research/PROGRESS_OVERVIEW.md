# Project progress - 2026-09-10

We have working parameter compression, but the complete research target remains
unmet: at least 70% fewer FFN weights, within 1% relative validation loss of BOTH
full GELU and SwiGLU, while beating calibrated narrow controls. Consistent
multi-seed, longer-training, scale and broader-data evidence is still required.

## Latest completed experiment: direct-input nonlinear bases

H094 tests rational, Hermite and trigonometric feature pairs without a fixed
ReLU/GELU/SwiGLU candidate base. The336-run grid uses the same four synthetic
families with fresh input seed9844:65,536 train /4,096 selection /4,096 reporting
samples, batch256, three optimization seeds, two rates and300 updates per run.
It completes100,800 updates in5.82 minutes. All672 independent checkpoint scores
and336 initializations reproduce exactly.

All three richer recipes fail their full promotion gates. The simpler even-feature
control is the useful finding:12.92% lower aggregate MSE than narrow GELU and
8.89% lower than full GELU at this short budget. Post-training folding reduces
stored weights from350,208 to282,624 (76.04% fewer than full) with less than1e-8
MSE change across12 checkpoints. Training still used350,208 weights.

The folded model's native batch256 inference is0.261 ms versus0.129 ms for narrow
GELU. It remains an unvalidated synthetic lead with a speed deficit, not a
breakthrough or an earned language-model promotion. [Full report](input_basis_results.md).

## Previous experiment: learned neuron geometry

We tested a different mechanism: input-dependent rotations of feature pairs,
using four shared learned parameters, alongside one-parameter Bezier and
eight-parameter grouped residual bumps. All four recipes fail their frozen
promotion gates. Their aggregate held-out errors are 0.72?6.68% above narrow
GELU. The best simple control is a four-shift GELU, 3.23% below narrow GELU but
11.18% above full GELU. No new architecture is promoted.

This was synthetic regression, not a new language run: four target families,
384-dimensional inputs, 65,536 training examples plus 4,096 selection and 4,096
reporting examples per task. Each batch contains 256 examples. Fourteen forms,
three optimization seeds and two rates complete 336 runs of 600 updates:
51,609,600 total training example presentations, taking 13.29 minutes.
All 672 independent checkpoint scores reproduce exactly.

The activations did learn: curves bend and twist amplitudes change. Their
learning does not establish a useful quality/compute gain. Pair twists cost
about 2.25 times narrow GELU's native update time. [Report and learned shapes](neuron_geometry_results.md).

## Latest longer-budget language results

Six fresh WikiText-2 runs, seed 17, 3,200 updates. NLL is validation loss:
lower is better; these values are not accuracy percentages.

| Model | Final NLL | FFN weights |
|---|---:|---:|
| Full SwiGLU | 4.1054 | 9,437,184 |
| Full GELU | 4.1279 | 9,437,184 |
| Narrow SwiGLU | 4.1530 | 2,801,664 |
| Narrow GELU | 4.2058 | 2,801,664 |
| BlockShuffle SwiGLU | 4.1443 | 2,801,664 |
| BlockShuffle GELU | 4.2420 | 2,801,664 |

Plain BlockShuffle is the best compressed form in this latest one-seed cohort.
Its earlier three-seed 3,200-step study averaged 1.256% worse NLL than full
SwiGLU, missing our 1% limit. The newer GELU passes a three-seed 800-step
comparison but fails the longer test: +3.327% versus full SwiGLU, +2.765% versus
full GELU, and worse than both narrows. Its fixed recipe is closed. Schedules,
execution and optimizer differences define separate experiments.
[Latest report](ungated_duration_results.md),
[earlier plain replication](long_duration_replication_results.md).

## Data, batches and duration

- Main corpus: WikiText-2 raw v 1; train-only 4,096-token BPE vocabulary.
- Cached data: 3,083,650 training tokens and 322,802 validation tokens.
- Each training batch has 16 sequence windows of 128 input/next-token targets:
  2,048 predicted training targets per optimizer update.
- Each 3,200-update run presents 6,553,600 sampled training targets, about 2.13
  cache-token exposures. Sampling is with replacement, not ordered epochs.
- Full validation scores 322,688 targets in 158 batches; the last has nine windows.
- Earlier stages used a small TinyStories subset and synthetic function fitting.
  The cited activation fitting screen uses batch 256 and MSE, not language NLL.

Budgets progressed from 200-update screens to 800-update comparisons (including
seeds 17/29/43), then 3,200-update duration tests. The latest six-run cohort took
about 34 minutes: September 8, 10:10:41 to 10:45:06 UTC. All six models still
improved during the last 800 updates; convergence is unestablished. Reused
validation is development data, and the official test split remains unscored.

The project began September 6. At the September 10 status check, the goal
tracker recorded about 31.3 hours of accumulated agent work: reading, coding,
audits and waiting as well as experiments. This is not GPU-training time.

## Learnable activations were implemented and trained

| Tested family | Result |
|---|---|
| Shifted Bezier/quadratic | Selected BlockShuffle short-screen loss worsened 0.442%; no promotion |
| Affine correction, 128 extra weights | Only 0.101% lower mean NLL at matched LR, winning 2/3 seeds; missed material-benefit gate |
| Group-shared rational, 320 extra weights | One-seed 200-update NLL 5.8980, 1.216% below its selected base; 883.17 MiB native peak fails memory |
| Static/input-conditioned corrections | Synthetic fitting gains only 0.0585% / 0.1213% over plain; missed the frozen 2% gate |

Rational is a short-screen signal with separately selected rates, not a replicated
activation-specific advantage. Resetting its learned shape barely changes final
loss. The earlier 1.755% affine gain was rate-confounded; the corrected result is
0.101%. No tested activation has a dependable combined quality/resource win.
[Activation guide](learnable_activation_domain.md),
[rational/Bezier results](learnable_activation_results.md),
[corrected affine comparison](affine_rate_replication_results.md),
[input-conditioned fitting](token_activation_fit_results.md).

## Latest controlled offset fitting

H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by -13.149%
versus plain; offset with the fixed first-factor LR correction changes it by
-14.529%. Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.
The broad research goal remains unmet.

All 240 selection and reporting scores and 48 resets reproduce exactly. Both
selected-recipe and matched-rate conditions are evaluated against fresh full,
narrow, gain and optimizer controls. [Completed study](latent_offset_fit_results.md).

## Latest follow-up: what the affine control does

All 12 selected affine checkpoints were checked without retraining. Removing
gain alone makes aggregate error 2.97 times worse; removing offset makes it 1.84
times worse. The effects interact. Gain can instead be folded into existing
weights while preserving the function; offsets can be cached as output biases.
The measured FP32 error drift is below a relative 1.7e-8. This is a mathematical
and numerical simplification, not a measured speed or training improvement.
[Mechanism report](latent_affine_mechanism_results.md).

## Latest learned-activation result

The internal-activation study finished all 192 synthetic runs in 13.06 minutes.
Tanh and sine curves improve aggregate held-out error by 5.76% and 7.07% over
plain, but equally sized affine improves 15.03%. Both curves lose to affine and
narrow GELU, so these fixed recipes are closed. No language trial is earned.

Each learned function adds 48 weights per FFN. All eight checks pass; independent
verification exactly reproduces 192 selection scores, 96 selected held-out scores
and 36 activation resets. The current curves are useful to their trained models,
but do not provide a combined quality/runtime win. The active tree remains three
model folders. [Completed study](latent_activation_recovery_results.md).

### Latest completed language screen

H081 completes all 24 fresh 200-update WikiText-2 runs and all final checkpoint
rescores match exactly. BlockShuffle calibrated: fails fixed screen; BLAST SwiGLU: fails fixed screen; BLAST GELU: fails fixed screen.
This is one-seed short-budget evidence; longer, multi-seed, scale and broader
data requirements remain open. [Complete screen](blast_learning_screen_results.md).

| Model | Selected LR | Final NLL | Peak MiB | Mean update ms | Clip % |
|---|---:|---:|---:|---:|---:|
| Full SwiGLU | 0.0012 | 5.907694 | 384.56 | 102.25 | 10.5 |
| Full GELU | 0.0012 | 5.879278 | 393.25 | 87.15 | 12.0 |
| Narrow SwiGLU | 0.0012 | 5.912505 | 279.45 | 93.05 | 8.5 |
| Narrow GELU | 0.0006 | 5.902164 | 279.02 | 91.39 | 17.5 |
| BlockShuffle legacy | 0.0006 | 5.970604 | 279.77 | 126.89 | 16.0 |
| BlockShuffle calibrated | 0.0012 | 6.073010 | 279.77 | 139.54 | 7.5 |
| BLAST SwiGLU | 0.0006 | 6.165411 | 281.20 | 153.90 | 38.5 |
| BLAST GELU | 0.0006 | 5.951733 | 280.95 | 125.34 | 11.0 |

## September 10 status

The controlled offset study completed 240 runs in 13.87 minutes. Its best offset
recipe still has 7.14% higher error than narrow GELU and takes 6.96ms/update versus
2.24ms in this standalone FP32 experiment. No new activation is promoted.

The fixed 2:4 sparse study passes 16 native checks, then stops on its first
accelerated case because the installed CUTLASS operation is unsupported.
No training or speed result follows. A pairwise rational-rotation activation is
now the distinct mechanism under investigation, with no measured learning claim.

Rational BlockShuffle is archived after its repeated memory failures. The current
active scope is two model folders, five variants and eight recipes. Raw tensors
and datasets stay local; compact evidence and source are retained in Git.
See [current direction](research_direction_2026_09_10.md) and
[artifact inventory](ARTIFACTS.md).
