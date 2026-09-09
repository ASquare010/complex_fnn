# H071 - Full-size rotated-shuffle fitting screen

Freeze before fitting or observing target errors. The [H070 qualification](rotated_shuffle_results.md)
earns this comparison. The active shortlist and prior rejections remain unchanged.
This is one standalone FFN, without attention, residual stacks or normalization.
No corpus training or full-model resource qualification is included.

## Question, dimensions and fixed budget

Does the cross-origin rotation learn more accurately than plain BlockShuffle,
its equally sized absorbable rotation control and calibrated narrow SwiGLU?
Use the actual d384/h2048/G8 compressed projection dimensions. Every intermediate
width is k384, so all 64 canonical blocks have six original latent paths. This
preserves the full-size projection topology, addressing the H068 scale caveat.

| Form | Hidden width | Parameters | Reduction vs either full control |
|---|---:|---:|---:|
| Plain BlockShuffle | 2048 | 350,208 | 70.3125% |
| Absorbable rotation, shift0 | 2048 | 350,784 | 70.263671875% |
| Cross-origin rotation, shift1 | 2048 | 350,784 | 70.263671875% |
| Calibrated narrow SwiGLU | 304 | 350,208 | 70.3125% |
| Full SwiGLU | 1024 | 1,179,648 | 0% |
| Full GELU | 1536 | 1,179,648 | 0% |

The narrow control matches plain exactly and has 576 fewer parameters than each
rotation (0.1642% of the rotated count). At eight layers the rotated FFN count
is 2,806,272 and the corresponding full Transformer count would be 9,104,256;
no Transformer is trained in this screen.

Seven targets, six forms, seeds 17/29/43 and constant base rates 0.001/0.003:
252 cells, each 300 updates with batch 256. Total allocation is 75,600 updates
and 19,353,600 training-example presentations. All cells train from scratch.
Equal rate and update budgets for every form; no extension, new rate, pairing,
angle initialization, angle LR multiplier or source change after outcomes.
Complete all cells unless a runtime or finite-value failure stops the process.
One GPU worker at a time, four CPU threads, existing UV compile/data extras.
Use FP32, TF32 disabled, eager native Torch and no checkpointing/custom kernel.
This larger fitting screen is bounded by the fixed cell/update allocation.

## Fixed data and target functions

CPU generator seed 812 creates 6,144 rows of 384 independent Uniform[-sqrt(3),sqrt(3)]
coordinates: 4,096 training, 1,024 selection, 1,024 held-out reporting. Save every
input/target tensor. The three optimization seeds share this one dataset.

Three teachers have the same name-seeded (9281) variance-calibrated block factors:
plain, absorbable and cross-origin. Both rotated teachers set every projection's
192 angles to linspace(-0.3,0.3,192); only the pairing differs. These are family-
aligned learning probes, not neutral proof of broad superiority. Plain and the
absorbable teacher belong to the same projection family; their fixed functions
differ. The cross teacher uses H070's qualified component at nonzero angles.
The students' rotation angles always start at zero. Do not use the sparse rank
witness as a fitting target or change the teacher amplitude after results.

Four generic vector targets use a=x_i, b=x_(i+1), c=x_(i+2), d=x_(i+3), cyclically,
then one fixed 384x384 orthogonal output rotation from CPU QR generator seed 9282:

- Smooth: sin(2a) + b^2 + exp(0.5c) - d.
- Oscillatory: sin(3a) cos(2b) + 0.5 sin(4c+d).
- Multiplicative: ab + 2bcd + a^2 d.
- Piecewise: ReLU(a+b) - 0.7 abs(c-d) + where(a>0,b,c).

Divide each output coordinate by its training population standard deviation;
do not center. The metric is variance-scaled MSE, not centered NMSE or NLL.
No held-out data enters scaling, initialization, gradients or rate selection.
Each seed's saved CPU index stream uses generator 20000+seed to sample 300x256
indices in [0,4096), shared across every task, form and rate.

## Initialization and optimizer

Every projection has dense-equivalent mean row squared norm one at initialization.
Dense weights use the existing name-derived Gaussian seed and std 1/sqrt(fan_in).
Structured factors use the retained semi-orthogonal initializer, each multiplied
by sqrt(1/(0.02 sqrt(projection fan_in))). All three compressed forms have exactly
the same initial block-factor bytes; rotations start at zero. Narrow down's
1/sqrt(304) initialization already applies its fan-in correction exactly once.
There is no data-driven calibration or learned output rescaling.

Use existing structured fan-in LR calibration: factor multipliers 4/4 for up
and gate, 4/(64/3) for down. Narrow down multiplier is 1024/304. Dense remaining
multipliers and all angle multipliers are one. AdamW betas (0.9,0.95), epsilon
1e-8, decay zero, global gradient clip norm one, no schedule. Record actual
groups, every loss/pre-clip norm and final per-parameter gradient norms.

## Measurement and selection

Save final checkpoints including all weights, optimizer moments and RNG; exact
source/plan/data/sampler hashes; initial/final state hashes; all 300 update times;
clipping fraction; projection activation/weight diagnostics; rotation angle RMS,
maximum absolute radians and sine RMS. Check finite loss/norms and final states.
Time updates with CUDA synchronization; first 50 are warmup, report median of
remaining 250. Reset peak counters after warmup and report allocated/reserved
peaks. These include the standalone input/target/index cache, not full-model
resources, autoregressive serving or deployment measurements.

At step 300 score train and selection splits. Choose lower selection MSE for
each task/form/seed, tie smaller rate; save selection before scoring either rate
on held-out reporting rows. Keep both rates' checkpoints and held-out metrics.
No intermediate checkpoint selection. For each rotated checkpoint also reset
all angles to zero, score held-out rows, then restore and verify state hashes.
This post-training ablation measures final dependence, not causal optimization
benefit: the factors have coadapted and the absorbable control can change basis.

## Frozen decision

Each rotated form earns a separately frozen full-model resource qualification
only if its selected endpoints satisfy every condition:

1. At least 2% aggregate reduction versus plain in paired geometric mean MSE
   ratio over all 21 task/seed cells.
2. Better than plain in every seed's seven-task geometric mean.
3. Better aggregate error than calibrated narrow.
4. No generic task's three-seed geometric mean is over 5% worse than plain.
5. At least 70% parameter reduction and finite completed final states.

Cross-origin additionally needs at least 1% aggregate improvement over the
absorbable control. The control can earn resource qualification as an optimization
parameterization, with no claim of a new projection family. Report both full
controls without transplanting the language NLL allowance onto synthetic MSE.
Report all selected rates, task/seed metrics and loss trends; 300 updates and
three optimization seeds do not establish convergence or statistical significance.

Failure closes this tested full-size fitting recipe, without automatically
adding angles, stages, learning rates, teacher amplitude or training budget.
Passing earns full-model resources only, not language training. H070's matrix
approximation obstruction is not an FFN learning theorem. The overall goal
still needs language quality, actual resources, longer runs, convergence,
multiple scales, broader data and a strong prior-art comparison.

## Execution and preservation

Keep all new code beside results/rotated_shuffle_fit_v1. Preserve H070's final
and result hashes and every prior source. Freeze this plan and all three new
source files before four isolated harness checks: counts/shared initial functions,
parameter-group coverage/calibration, train-only target scaling, reproducible
sampling and shape-sensitive hashes. Their passing is separate from the active
108-test suite. A hidden coordinator records PID, UTC, stdout/stderr, terminal
return code and immutable source archive before dispatching one fitting worker.
No automatic retry or overwrite. Verify process identity before recovering any
lost observation. Independently inspect checkpoints, rate selection and held-out
scores before interpreting results. Preserve old results, plans and archives.
