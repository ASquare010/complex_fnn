# H073 - Ungated fitting with positive and absolute-error controls

Freeze before fitting or observing target errors. [H072](ungated_blockshuffle_results.md)
earns this full-size comparison of an existing conventional operator. H068/H071
failed activation/rotation recipes remain closed. No active source, registry,
recipe or old result is changed. One standalone FFN has no attention, residual
stack or normalization. This is not language training or a new activation family.

## Fixed allocation and forms

Use width384 and groups8 for structured forms, preserving all 64 canonical
projection blocks with six intermediate paths. Forms and counts are fixed:

| Form key | Activation | Hidden | Weights |
|---|---|---:|---:|
| plain | BlockShuffle SwiGLU | 2048 | 350208 |
| gelu_same | BlockShuffle GELU | 2048 | 233472 |
| gelu_matched | BlockShuffle GELU | 3264 | 350208 |
| narrow_swiglu | Dense SwiGLU | 304 | 350208 |
| narrow_gelu | Dense GELU | 456 | 350208 |
| full_swiglu | Dense SwiGLU | 1024 | 1179648 |
| full_gelu | Dense GELU | 1536 | 1179648 |

Matched GELU retains70.3125% FFN reduction and uses59.375% more hidden features
than plain SwiGLU; same-width GELU retains80.208333%. At L8 the corresponding
matched FFN/total counts are2,801,664/9,099,648; same-width1,867,776/8,165,760.
The two narrow forms each exactly match plain and matched GELU parameters.

Seven tasks x seven forms x seeds17/29/43 x base rates0.001/0.003 =294 cells.
Each cell trains from scratch for300 updates, batch256:88,200 updates and
22,579,200 example presentations total. Every form has equal update/rate budgets.
Complete all cells unless a runtime/finite-value failure stops execution. No
extra rate, width, activation, optimizer variant or longer run after outcomes.
One CUDA worker, four CPU threads, UV's existing compile/data extras, FP32,
TF32 disabled, native eager Torch without checkpoints/custom kernels.

## Data and fixed targets

CPU generator9812 creates73,728 independent rows of384 Uniform[-sqrt3,sqrt3]
coordinates: first65,536 training, next4,096 selection, last4,096 held-out report.
This increases the fixed training pool relative to H071's4,096 rows, to reduce
repeated fitting to a small sample. It does not guarantee generalization. These
are different task/data cohorts; cross-round changes do not identify a causal
data-size effect. The three optimization seeds share one input/teacher dataset.
Save all inputs, targets, train standard deviations and sampler tensors.

Tasks, in order:

1. linear: x times a fixed orthogonal384x384 matrix from CPU QR seed9282.
2. plain_teacher: a variance-calibrated BlockShuffle SwiGLU h2048, seed9281.
3. gelu_teacher: a variance-calibrated BlockShuffle GELU h3264, seed9281.
4. smooth: sin(2a)+b^2+exp(0.5c)-d.
5. oscillatory: sin(3a)cos(2b)+0.5sin(4c+d).
6. multiplicative: ab+2bcd+a^2d.
7. piecewise: ReLU(a+b)-0.7abs(c-d)+where(a>0,b,c).

For generic tasks use a=x_i,b=x_(i+1),c=x_(i+2),d=x_(i+3), cyclically, then
multiply by the same fixed orthogonal output matrix as linear. Teacher targets
favor different families; they are not neutral evidence of general superiority.
Their dimensions differ and their factor bytes are not asserted identical.
Divide every output coordinate by its training population standard deviation,
without centering. Report variance-scaled MSE and the zero-output predictor's
held-out MSE. Zero is an absolute diagnostic and never enters rate selection.
All output coordinates and every learned parameter count toward comparisons.

Per-seed CPU index streams use20000+seed, shape300x256, uniform integers in
[0,65536); the saved stream is shared by every task/form/rate. This presents
76,800 sampled rows per cell from the larger finite pool. Selection/report rows
never enter gradients, target scaling or initialization.

## Initialization, optimization and measurement

Each projection starts with dense-equivalent mean row squared norm one. Dense
weights use name-derived Gaussian initialization with std1/sqrt(fan_in).
Block factors use existing semi-orthogonal initialization, each multiplied by
sqrt(1/(0.02sqrt(projection fan_in))). Plain and same-width GELU share all common
up/down factor bytes at each seed; the activation differs. Matched GELU has
different shapes and no shared-function claim. Narrow down initialization uses
its own fan_in exactly once, without a second correction.

Existing factor LR multipliers are4/4 for up/gate and4/(h/96) for down. Both
narrow down multipliers are1536/456=1024/304; other dense multipliers areone.
AdamW betas(0.9,0.95), epsilon1e-8, decayzero, global norm clipone, constant base
rate. Preserve every loss/pre-clip norm, actual groups, final per-parameter
clipped gradient norms, initial/final weight hashes, projection activations,
checkpoints with optimizer moments and RNG, finite-value checks and clipping.

Synchronize CUDA around every update; first50 timings are warmup. Report median
of remaining250 and allocated/reserved peaks reset after warmup. These include
the standalone GPU input/target/index cache. They are not full-model training
or serving resources, and cannot qualify the Transformer memory allowance.

At final step300, score the entire train and selection splits. Select lower
selection MSE independently for each task/form/seed, tie smaller rate. Save the
selection before scoring either rate on held-out rows. Preserve both rates'
checkpoints and all held-out metrics. No intermediate checkpoint selection.
Paired geometric means over equal-weight tasks/seeds define gate ratios;
report all rates, task/seed metrics, zero-relative errors and training trends.
No statistical-significance or convergence claim from three optimizer seeds.

## Assay validity and fixed promotion gates

The entire assay qualifies only if the selected full GELU control:

- Has held-out linear MSE <=50% of zero-predictor MSE in each seed.
- Has three-seed geometric mean held-out/zero ratio <=0.98 on at least two of
  the four generic tasks.

These are fixed positive-control requirements, not guaranteed outcomes. If they
fail, label the screen INCONCLUSIVE_ASSAY_FAILURE and earn no model allocation,
regardless of relative candidate gains. Still finish and preserve all fixed cells;
do not adapt training to the observed positive-control results.

Each ungated candidate earns a separately frozen full-model resource comparison
only if all of the following hold:

1. The positive-control assay passes.
2. Paired geometric mean error is at least2% lower than plain across all21
   task/seed endpoints, with improvement in every seed's seven-task mean.
3. Aggregate error beats both calibrated narrow controls.
4. Each generic task's three-seed mean ratio is <=1.05 versus plain AND zero.
5. Its linear error is <=50% of zero in every seed.
6. All final states are finite and FFN reduction is >=70%.

Matched-width GELU additionally needs >=1% improvement over same-width GELU to
justify the extra weights. Report both full controls without applying a language
NLL allowance to synthetic MSE. If the assay passes but candidates fail, reject
these fixed recipes at this budget. No automatic data/step/rate expansion,
activation repair or language allocation follows from either negative outcome.

H072's bias-free parity limits remain relevant: GELU's odd part is linear and
its population multiplicative-target relative floor is12/17; this is not an
exact bound on the finite held-out set or on deeper/biased networks. Passing
fitting still establishes no novelty, general superiority or gold-goal completion.

## Execution and audit

Before dispatch pin H072 final/result and all94 preceding sources, plus three
new source files and this plan. Five isolated harness checks cover counts/shared
factors, optimizer coverage/calibration, train-only scaling, stream reproducibility
and rejection on positive-control/absolute-error failures. They are separate
from the prior108-test active suite. Save source archive, provenance, PID/UTC,
logs and return code under a hidden coordinator; no overwrite/automatic retry.
Verify live OS process identity before recovering an interrupted observation.
Independent analysis regenerates data, audits every checkpoint and selection,
rescores selected held-out predictions on CPU and rederives all decision gates.
Preserve all previous plans, active sources, data, results and archives.
