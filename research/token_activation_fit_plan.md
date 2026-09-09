# H068 - Token activation function-fitting screen

Frozen before fitting or observing target errors. H067's 21 local checks earn
this bounded synthetic comparison. The gold language-model goal remains unmet.
No active source, model folder, factory, recipe or previous evidence changes.

## Question and fixed allocation

Does the unchanged H067 residual activation learn more accurately than plain
BlockShuffle, its static ablation and a calibrated narrow SwiGLU at this budget?
This is one standalone FFN, without attention, residual stacks or normalization.
Use d48/h256/G8, preserving the d384/h2048/G8 width ratios. Counts:

| Form | Hidden | Parameters | Reduction vs either full control |
|---|---:|---:|---:|
| Plain BlockShuffle | 256 | 5,472 | 70.3125% |
| Static residual activation | 256 | 5,473 | 70.3071% |
| Token-conditioned residual activation | 256 | 5,521 | 70.0467% |
| Narrow SwiGLU | 38 | 5,472 | 70.3125% |
| Full SwiGLU | 128 | 18,432 | 0% |
| Full GELU | 192 | 18,432 | 0% |

Seven targets, six forms, seeds 17/29/43 and two base learning rates .001/.003:
252 cells, each exactly 300 updates of batch 256 (75,600 updates; 19,353,600
training examples). Each cell trains from scratch. No extension, extra rate,
candidate-only tuning or source changes after outcomes. Complete all fixed cells
unless a runtime/finite-value failure stops execution; retain failures, no retry.
One CUDA process, four CPU threads, UV with compile/data extras. No corpus access.

## Data and controls

Generate fixed CPU FP32 inputs with seed 812: 6,144 independent rows of 48
Uniform[-sqrt(3),sqrt(3)] coordinates. First 4,096 train, next 1,024 selection,
last 1,024 held-out report. Identical inputs across tasks and training seeds.
Seeds are optimization replications, not independent datasets. Save all tensors.

Three teacher targets use name-seeded (9281), variance-calibrated BlockShuffle
projections: plain, static with b=.4, and dynamic with b=.4 and w[0]=.75 (all
other router weights zero). Teachers share projection bytes; target functions
favor different members of the nested family. These include but are broader
than H067's single-path witness; no exact witness-only leaderboard.

Four additional vector-valued targets use each coordinate a=x_i and cyclic
b=x_(i+1), c=x_(i+2), d=x_(i+3), then multiply the output by one fixed orthogonal
48x48 matrix generated with seed 9282. This avoids unused scalar-output weights.

- Smooth: sin(2a) + b^2 + exp(.5c) - d.
- Oscillatory: sin(3a)cos(2b) + .5 sin(4c+d).
- Multiplicative: ab + 2bcd + a^2 d.
- Piecewise: ReLU(a+b) - .7 abs(c-d) + where(a>0,b,c).

Divide each target coordinate by its training standard deviation (population
denominator). Do not center: all compared FFNs have no output bias. This reports
variance-scaled MSE, not centered NMSE. Train statistics never use held-out rows.
Teachers and generic definitions are fixed before outcomes; no target weighting
selection. Full vector output, all learned weights counted.

For standalone inputs, initialize every projection to dense-equivalent mean row
squared norm one. Dense weights have name-derived Gaussian std 1/sqrt(fan_in).
BlockShuffle uses the existing semi-orthogonal initializer and multiplies each
factor by sqrt(1/(.02 sqrt(projection fan_in))). This rescales the product's
existing dense-equivalent .02 initialization to 1/sqrt(fan_in). Static/dynamic
share plain projection bytes and zero gates initially. Narrow down std 1/sqrt(38)
is the fan-in correction relative to full down std 1/sqrt(128); do not apply it
twice. No data-driven initialization or learned output rescaling.

Use existing parameter_groups with structured fan-in LR calibration; narrow
down uses 128/38 LR multiplier. Other dense LR multipliers are one; router LR
multiplier one. AdamW betas (.9,.95), epsilon 1e-8, decay zero, global clip norm
1; constant base rate for 300 updates. Record actual groups. Both rates receive
equal budget for every task/form/seed. Minibatches come from a saved CPU index
stream seeded 20000+training seed, shared across all tasks/forms/rates.

All arithmetic FP32 with TF32 disabled to resolve small fitting differences.
Eager native Torch for every form, no checkpointing or custom kernels. H067
CPU/GPU local qualification remains separate from this learning protocol.

## Measurement, selection and gates

Store every training loss and pre-clip norm, finite-value checks, clipping rate,
final checkpoint with optimizer moments and RNG, data/sampler/source hashes,
initial shared-state hashes, final projection activations and gradient norms,
router coefficient/saturation statistics, and reset-router errors. Time each
update with CUDA synchronization; first 50 are warmup, report median remaining
250. Record allocated/reserved peaks after warmup reset. This small FFN profile
does not qualify full-Transformer resources or imply a deployment speedup.

At final step only, score train and selection splits. Select the lower-selection
MSE rate independently for each task/form/seed (tie: smaller rate). Freeze that
selection before scoring held-out report rows for either rate. Preserve both
rates' report metrics; only selected endpoints enter decision statistics.
No best-checkpoint selection. Three independent initialization/sampling seeds
share the fixed data and teacher. Report per-task seed means, all selected rates,
and paired geometric mean error ratios (lower is better), with no significance
or convergence claim from three seeds and 300 updates.

Static earns a subsequent full-model resource qualification only if: at least
2% aggregate improvement over plain across all 21 task/seed cells; improvement
in every seed's seven-task geometric mean; lower aggregate error than calibrated
narrow; and no generic task's three-seed geometric mean is >5% worse than plain.
Dynamic must pass the same gates and additionally improve aggregate error over
static by at least 1%. Require finite metrics/checkpoints and >=70% reduction.
Report both full controls without applying an NLL threshold to synthetic MSE.

Passing earns a separately frozen actual full-model resource qualification,
not language training. Failure closes this tested fitting recipe without a
larger amplitude, richer router, extra rate or longer run automatically following.
It does not disprove all token-adaptive activation functions. The H067 exact
function proof gives no finite-budget approximation or optimization guarantee.

## Reproducibility

Keep the driver and its small harness checks beside results/token_activation_fit_v1.
Before dispatch, pin H067 result/final audit and every prior source, snapshot all
new driver bytes plus this plan, then run harness checks. Save process PID, UTC,
logs, return code and terminal result. No overwrite or automatic retries. Verify
all protected historical evidence and active source bytes at completion.
