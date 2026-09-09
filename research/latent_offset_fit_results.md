# H085 - Controlled training of internal offset and gain

H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by -13.149%
versus plain; offset with the fixed first-factor LR correction changes it by
-14.529%. Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.
The broad research goal remains unmet.

These are synthetic, finite-budget measurements. The experiment has three
optimization seeds on one fresh dataset, not three independently drawn datasets
or a language-model result. Its selected-recipe gates and matched-rate gates are
reported separately below. No curve recipe from H083 is reopened.

## What was isolated

H084 showed affine gain can be folded into existing first factors and an offset
can remove the isolated SwiGLU's zero derivative at zero. Checkpoint interventions
did not explain training gains. H085 therefore trains gain-only, offset-only and
full-affine controls from matching initial functions, alongside a fixed optimizer
control and both conventional narrow baselines.

The partial forms preserve the original affine arithmetic. The unused group
control becomes a fixed zero buffer, while the other remains learned. They have
24 learned controls and 24 fixed zero buffer elements per FFN; full affine has
48 learned controls. No optimized offset kernel or changed backward formula is
introduced. Plain and narrow forms have 24 fewer learned weights than the partial
forms, a 0.00685% gap. Gain and offset have exactly equal learned counts.

The first-LR recipes multiply only first-factor learning rates by 5/8. They leave
initial tensors, the represented function, second-factor rates, shape rates,
epsilon, clipping and decay unchanged. The value5/8 was chosen before this new
dataset as a simple point in H083's observed slope range. It is history-informed,
not a tuned optimum or an exact reconstruction of learned-gain Adam dynamics.
There are two offset candidate recipes; all other forms are controls.

## Selected-recipe quality

Lower error is better. Aggregate ratios are geometric means over 12 paired
task/seed endpoints. Individual task errors are arithmetic means over three
optimization seeds. Rates are selected on a separate split before final
reporting; all rates and every checkpoint remain available.

| Form | Learned weights | MSE change vs plain | Ratio vs gain | Ratio vs narrow GELU |
|---|---|---|---|---|
| Plain BlockShuffle | 350,208 | +0.000% | 1.005094 | 1.253580 |
| Plain, first LR x5/8 | 350,208 | +0.889% | 1.014032 | 1.264728 |
| Learned gain | 350,232 | -0.507% | 1.000000 | 1.247227 |
| Learned offset | 350,232 | -13.149% | 0.872938 | 1.088752 |
| Offset, first LR x5/8 | 350,232 | -14.529% | 0.859061 | 1.071444 |
| Full affine | 350,256 | -14.990% | 0.854434 | 1.065673 |
| Narrow SwiGLU | 350,208 | +3.699% | 1.042275 | 1.299953 |
| Narrow GELU | 350,208 | -20.228% | 0.801779 | 1.000000 |
| Full SwiGLU | 1,179,648 | -8.462% | 0.920042 | 1.147501 |
| Full GELU | 1,179,648 | -30.786% | 0.695664 | 0.867651 |

| Form | Smooth | Oscillatory | Multiplicative | Piecewise |
|---|---|---|---|---|
| Plain BlockShuffle | 0.705830 | 1.006692 | 0.984869 | 0.723044 |
| Plain, first LR x5/8 | 0.725340 | 1.005774 | 0.986520 | 0.728401 |
| Learned gain | 0.701678 | 1.005583 | 0.981316 | 0.716060 |
| Learned offset | 0.539558 | 1.006686 | 0.923727 | 0.573819 |
| Offset, first LR x5/8 | 0.524942 | 1.005814 | 0.921216 | 0.555163 |
| Full affine | 0.521151 | 1.005601 | 0.917646 | 0.549498 |
| Narrow SwiGLU | 0.829293 | 1.016906 | 1.002543 | 0.692085 |
| Narrow GELU | 0.536858 | 1.009109 | 0.901112 | 0.419723 |
| Full SwiGLU | 0.665374 | 1.043419 | 0.994101 | 0.514751 |
| Full GELU | 0.482472 | 1.020878 | 0.896615 | 0.262953 |

| Frozen gate | Learned offset | Offset, first LR x5/8 |
|---|---|---|
| complete finite grid | PASS | PASS |
| positive control assay passes | PASS | PASS |
| two percent over plain | PASS | PASS |
| every seed better than plain | PASS | PASS |
| one percent over equal count gain | PASS | PASS |
| one percent over fixed lr plain | PASS | PASS |
| beats both calibrated narrows | FAIL | FAIL |
| within one percent full affine | FAIL | PASS |
| per task regression cap vs plain | PASS | PASS |
| per task regression cap vs zero | PASS | PASS |
| seventy percent reduction | PASS | PASS |
| matched rate offset benefit | PASS | PASS |

The narrow controls must both be beaten. The selected offset recipe must also
beat gain-only and the fixed-LR plain control, remain within 1% of full affine
and pass the separately required matched-rate robustness condition. These are
MSE-specific gates; the language NLL allowance is not applied to synthetic MSE.
Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.

## Effect at each learning rate

Each row compares the same base learning rate, dataset, sampling stream and
training budget. The fixed-LR offset form is paired with the matching fixed-LR
plain form. For either offset candidate, both rates must show at least1% lower
aggregate error and improvements in all three seeds to support a robust offset
benefit and earn resource qualification.

| Form | Reference | Base LR | Aggregate MSE change | Seed17 ratio | Seed29 ratio | Seed43 ratio |
|---|---|---|---|---|---|---|
| Learned gain | Plain BlockShuffle | 0.001 | -0.480% | 0.995219 | 0.995241 | 0.995154 |
| Learned gain | Plain BlockShuffle | 0.003 | -2.094% | 0.979322 | 0.979925 | 0.977938 |
| Learned offset | Plain BlockShuffle | 0.001 | -12.055% | 0.876228 | 0.882907 | 0.879236 |
| Learned offset | Plain BlockShuffle | 0.003 | -15.529% | 0.846838 | 0.844489 | 0.842795 |
| Offset, first LR x5/8 | Plain, first LR x5/8 | 0.001 | -11.564% | 0.881751 | 0.890003 | 0.881365 |
| Offset, first LR x5/8 | Plain, first LR x5/8 | 0.003 | -15.656% | 0.844501 | 0.842662 | 0.843147 |
| Full affine | Plain BlockShuffle | 0.001 | -12.011% | 0.876384 | 0.883421 | 0.879884 |
| Full affine | Plain BlockShuffle | 0.003 | -17.559% | 0.826279 | 0.824087 | 0.822872 |

The [result JSON](../results/latent_offset_fit_v1/result.json) also contains the
full fixed-rate comparison matrices and all240 reporting rows. Independent
verification rescored both rates for every form, so the robustness gate is
checked against the actual saved checkpoints, including unselected rates.
This is stronger evidence than a post-training reset, but still limited to the
fixed dataset, architecture, optimizer family and training budget.

## Dependence on learned controls after training

All trainable shape controls are reset to zero in each selected learned-control
checkpoint; factors and fixed zero buffers stay unchanged. Values above one
mean reset worsens error. These 48 interventions are independently reproduced.

| Form | Reset / original MSE | Smooth | Oscillatory | Multiplicative | Piecewise |
|---|---|---|---|---|---|
| Learned gain | 1.126200 | 1.190852 | 1.173551 | 1.036127 | 1.110938 |
| Learned offset | 1.790748 | 5.439966 | 1.001021 | 1.077916 | 1.751914 |
| Offset, first LR x5/8 | 1.819891 | 5.431354 | 1.001064 | 1.094935 | 1.842565 |
| Full affine | 2.339984 | 6.665150 | 1.205871 | 1.336628 | 2.790811 |

Resets measure dependence after coadaptation. They do not identify the causal
effect of including those controls throughout training; the matched-rate
trained comparisons above are the relevant controlled evidence.

## GPU resources

| Form | Mean median update ms | Max peak MiB | Mean clipped % | Upper LR / 12 |
|---|---|---|---|---|
| Plain BlockShuffle | 4.558 | 240.45 | 0.000 | 3 |
| Plain, first LR x5/8 | 4.770 | 240.45 | 0.000 | 9 |
| Learned gain | 6.425 | 241.21 | 0.014 | 3 |
| Learned offset | 6.327 | 243.14 | 0.042 | 8 |
| Offset, first LR x5/8 | 6.959 | 240.46 | 0.000 | 12 |
| Full affine | 6.825 | 243.89 | 0.083 | 11 |
| Narrow SwiGLU | 2.396 | 231.55 | 0.000 | 3 |
| Narrow GELU | 2.235 | 228.93 | 0.000 | 3 |
| Full SwiGLU | 2.171 | 244.88 | 0.000 | 3 |
| Full GELU | 1.909 | 247.00 | 0.083 | 6 |

Update times average each selected run's median after 50 warmup updates.
Allocated peaks are maxima across selected runs and include the standalone GPU
dataset cache. Timing uses synchronization around every update, FP32 native
CUDA, TF32 off, four CPU threads and one worker on the RTX4070 Laptop GPU.
These are sequential laptop measurements, not full-model memory, serving
throughput or a compiled-kernel benchmark. Partial controls intentionally retain
the original affine arithmetic for this comparison.

## Data, budget and verification

Fresh CPU input seed9813 replaces H083's9812. The fixed input permutation9283,
output rotation9282 and four analytic target formulas stay the same. The dataset
contains73,728 width384 examples:65,536 training,4,096 rate-selection and4,096
reporting. Output scaling uses training population standard deviation without
centering. Absolute MSE is not directly interchangeable with H083's different
sample draw; fresh full and narrow controls are used throughout.

Ten forms x four tasks x three seeds x two rates give240 fresh cells. Every run
uses600 updates and batch256:144,000 optimizer updates and36,864,000 training
example presentations. AdamW betas(0.9,0.95), epsilon1e-8, zero decay, clip1,
constant base rates and established fan-in/narrow calibration are retained.
There are no language targets, Transformer resource workers or training retries.
The coordinator took 13.87 minutes including qualification, fitting,
evaluation, startup and durable writing.

All eight preflight checks pass with zero updates, including80 exact retained/
checkpoint gradient tensor pairs, all30 form/seed initialization and optimizer
checks, finite differences, fresh-data contracts and rejection behavior.
The independent audit reconstructs all30 initial states,240 RNG/optimizer
histories, fresh targets/training scales/streams, fixed zero buffers and actual
counts. It exactly reproduces240 selection scores, all240 reporting scores and
48 resets. That is983,040 selection and1,179,648 reporting/reset example
presentations, zero optimizer updates. The extra120 unselected reporting passes
verify the mandatory two-rate gate. Audit duration:
32.40 seconds.

A status-inspection PowerShell process exited abnormally while listing CIM
processes after printing the passing preflight result. A native Get-Process
recheck confirmed the original coordinator and fitting handles were live and
logs advanced. Training was not restarted. The cause is undiagnosed and the
[observer record](../results/latent_offset_fit_v1/observation_shell_failure.json)
is separate from the experiment's terminal records.

All135 scientific sources,74 plans and prior H083/H084 anchors remain intact.
The active repository still has three model folders, six variants and nine
recipes. No universal gradient, convergence, scale or broad-corpus claim follows.
Any earned resource qualification requires its own frozen full-model study;
failure does not authorize another rate, step count or architecture variation.

## Every selected rate

| Task | Form | Seed17 LR | Seed29 LR | Seed43 LR |
|---|---|---|---|---|
| smooth | Plain BlockShuffle | 0.001 | 0.001 | 0.001 |
| smooth | Plain, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| smooth | Learned gain | 0.001 | 0.001 | 0.001 |
| smooth | Learned offset | 0.003 | 0.003 | 0.003 |
| smooth | Offset, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| smooth | Full affine | 0.003 | 0.003 | 0.003 |
| smooth | Narrow SwiGLU | 0.001 | 0.001 | 0.001 |
| smooth | Narrow GELU | 0.001 | 0.001 | 0.001 |
| smooth | Full SwiGLU | 0.001 | 0.001 | 0.001 |
| smooth | Full GELU | 0.001 | 0.001 | 0.001 |
| oscillatory | Plain BlockShuffle | 0.003 | 0.003 | 0.003 |
| oscillatory | Plain, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| oscillatory | Learned gain | 0.003 | 0.003 | 0.003 |
| oscillatory | Learned offset | 0.003 | 0.003 | 0.003 |
| oscillatory | Offset, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| oscillatory | Full affine | 0.003 | 0.003 | 0.003 |
| oscillatory | Narrow SwiGLU | 0.001 | 0.001 | 0.001 |
| oscillatory | Narrow GELU | 0.001 | 0.001 | 0.001 |
| oscillatory | Full SwiGLU | 0.001 | 0.001 | 0.001 |
| oscillatory | Full GELU | 0.003 | 0.003 | 0.003 |
| multiplicative | Plain BlockShuffle | 0.001 | 0.001 | 0.001 |
| multiplicative | Plain, first LR x5/8 | 0.001 | 0.001 | 0.001 |
| multiplicative | Learned gain | 0.001 | 0.001 | 0.001 |
| multiplicative | Learned offset | 0.001 | 0.001 | 0.001 |
| multiplicative | Offset, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| multiplicative | Full affine | 0.001 | 0.003 | 0.003 |
| multiplicative | Narrow SwiGLU | 0.001 | 0.001 | 0.001 |
| multiplicative | Narrow GELU | 0.001 | 0.001 | 0.001 |
| multiplicative | Full SwiGLU | 0.001 | 0.001 | 0.001 |
| multiplicative | Full GELU | 0.001 | 0.001 | 0.001 |
| piecewise | Plain BlockShuffle | 0.001 | 0.001 | 0.001 |
| piecewise | Plain, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| piecewise | Learned gain | 0.001 | 0.001 | 0.001 |
| piecewise | Learned offset | 0.003 | 0.003 | 0.001 |
| piecewise | Offset, first LR x5/8 | 0.003 | 0.003 | 0.003 |
| piecewise | Full affine | 0.003 | 0.003 | 0.003 |
| piecewise | Narrow SwiGLU | 0.003 | 0.003 | 0.003 |
| piecewise | Narrow GELU | 0.003 | 0.003 | 0.003 |
| piecewise | Full SwiGLU | 0.003 | 0.003 | 0.003 |
| piecewise | Full GELU | 0.003 | 0.003 | 0.003 |

[Frozen plan](latent_offset_fit_plan.md),
[independent audit](../results/verification/latent_offset_fit_analysis_v1.json),
[all measurements](../results/latent_offset_fit_v1/result.json),
[preceding mechanism study](latent_affine_mechanism_results.md).
