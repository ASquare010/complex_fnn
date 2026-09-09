# H094: direct-input nonlinear basis results

**Later evidence:** the matched-sample affine diagnostic limits the apparent
general advantage to 1.724%, below its 2% threshold. The general lead is closed;
the 12.60% piecewise improvement remains a scoped result. See the
[updated interpretation](affine_falsification_results.md). The measurements below
are preserved as the historical H094-H095 result.

The original gold objective remains unmet. This is a fixed300-update synthetic
screen of learned projections/readouts with fixed nonlinear bases. None of the
three candidates uses a GELU/ReLU/SwiGLU base. Their basis functions have zero
learned shape parameters; the input and output weights learn their combinations.

[Frozen equations and prior art](input_basis_lift_plan.md),
[fitting plan and decision thresholds](input_basis_fit_plan.md),
[audited machine-readable result](../results/input_basis_fit_v1/result.json.gz).

## Main comparison

Ratios are geometric means of12 paired reporting MSE ratios (four tasks, three
seeds), after rate selection on a separate split. Lower is better.

| Form | MSE vs narrow GELU | Mean median update ms | Weights |
|---|---:|---:|---:|
| rational | 0.920172 | 2.968 | 350,208 |
| hermite | 0.906929 | 2.891 | 350,208 |
| trig | 0.927932 | 2.649 | 350,208 |
| duplicate | 0.870785 | 2.707 | 350,208 |
| linear | 1.382488 | 2.217 | 350,208 |
| antipodal | 0.912353 | 2.506 | 350,208 |
| raw_gelu | 0.924232 | 2.115 | 350,208 |
| core_linear | 1.361233 | 1.476 | 147,456 |
| full_gelu | 0.955753 | 2.074 | 1,179,648 |
| full_swiglu | 1.154438 | 2.660 | 1,179,648 |
| narrow_gelu | 1.000000 | 2.142 | 350,208 |
| narrow_relu | 1.021038 | 2.098 | 350,208 |
| narrow_swiglu | 1.264600 | 2.562 | 350,208 |
| gelu_offset | 0.959831 | 2.707 | 350,212 |

All direct-input nonlinear designs use70.3125% fewer weights than either full
reference. This does not imply70% lower whole-Transformer size or measured FLOPs.

## Frozen decisions

- **rational: REJECTED_AT_THIS_BUDGET**. Failed gates: two percent over each control, every seed beats each control, per task regression cap, update time cap.
- **hermite: REJECTED_AT_THIS_BUDGET**. Failed gates: two percent over each control, every seed beats each control, per task regression cap, update time cap.
- **trig: REJECTED_AT_THIS_BUDGET**. Failed gates: two percent over each control, every seed beats each control, per task regression cap.

A rejected recipe gets no automatic refinement or longer run. This verdict
is limited to the fixed initialization, optimizer and300-update budget; it is
not an impossibility theorem about rational, polynomial or periodic functions.

## Per-task reporting MSE

Mean +/- sample standard deviation over three seeds.

| Form | Smooth | Oscillatory | Multiplicative | Piecewise |
|---|---:|---:|---:|---:|
| rational | 0.45601 +/- 0.00038 | 1.02404 +/- 0.00046 | 0.88712 +/- 0.00014 | 0.49375 +/- 0.00021 |
| hermite | 0.44353 +/- 0.00014 | 1.01910 +/- 0.00037 | 0.88074 +/- 0.00013 | 0.48486 +/- 0.00017 |
| trig | 0.46429 +/- 0.00049 | 1.03069 +/- 0.00066 | 0.89445 +/- 0.00005 | 0.49419 +/- 0.00067 |
| duplicate | 0.44770 +/- 0.00049 | 1.01347 +/- 0.00043 | 0.87563 +/- 0.00018 | 0.41289 +/- 0.00138 |
| linear | 2.25551 +/- 0.01231 | 1.02372 +/- 0.00007 | 0.88601 +/- 0.00036 | 0.50943 +/- 0.00024 |
| antipodal | 0.45005 +/- 0.00036 | 1.02103 +/- 0.00043 | 0.88323 +/- 0.00012 | 0.48706 +/- 0.00014 |
| raw_gelu | 0.45644 +/- 0.00018 | 1.02804 +/- 0.00032 | 0.89150 +/- 0.00014 | 0.49764 +/- 0.00032 |
| core_linear | 2.24167 +/- 0.00716 | 1.00821 +/- 0.00025 | 0.87138 +/- 0.00010 | 0.49741 +/- 0.00021 |
| full_gelu | 0.49291 +/- 0.00041 | 1.02590 +/- 0.00022 | 0.90558 +/- 0.00024 | 0.51987 +/- 0.00113 |
| full_swiglu | 0.71175 +/- 0.00038 | 1.02462 +/- 0.00052 | 0.99469 +/- 0.00051 | 0.69858 +/- 0.00255 |
| narrow_gelu | 0.55774 +/- 0.00047 | 1.01309 +/- 0.00015 | 0.91432 +/- 0.00024 | 0.55224 +/- 0.00016 |
| narrow_relu | 0.56985 +/- 0.00035 | 1.01676 +/- 0.00028 | 0.92462 +/- 0.00038 | 0.57881 +/- 0.00021 |
| narrow_swiglu | 0.88072 +/- 0.00084 | 1.01188 +/- 0.00019 | 1.00575 +/- 0.00041 | 0.81407 +/- 0.00042 |
| gelu_offset | 0.50034 +/- 0.00045 | 1.00828 +/- 0.00018 | 0.91222 +/- 0.00024 | 0.52618 +/- 0.00042 |

Median, sample variance and every seed value are in the public result JSON.

Zero-predictor reporting MSE: smooth=2.780481, oscillatory=0.999906, multiplicative=1.002498, piecewise=1.019462.

## Data, training and verification

Fresh synthetic input seed9844; four previously studied analytic families.
Each task has65,536 training,4,096 rate-selection and4,096 reporting samples,
with384 input/output dimensions. Batch256;300 updates per run (76,800 sampled
presentations, approximately1.17 training-set equivalents, not ordered epochs).
Fourteen forms, three seeds and two rates complete336 runs:100,800 updates and
25,804,800 presentations. No language dataset, real-data test or convergence
claim is part of this experiment.

Fitting wall time: 5.82 minutes, including data,
evaluation and checkpoint work. One RTX4070 Laptop GPU, FP32 native CUDA,
TF32off, four CPU threads. Exact optimizer and initialization differences are
declared in the frozen plan. No hyperparameter extension followed the outcomes.

Independent audit regenerates data exactly, verifies all336 initializations,
all168 selected rates, histories/counts and all672 checkpoint scores exactly.
Aggregate statistics, paired ratios and every decision are independently checked.

The H092 launcher had a Windows DLL initialization failure before numerical
tests. H093's explicit single-interpreter attempt passes the unchanged25 checks
and157 saved tensor comparisons, including56 exact checkpoint pairs. The
failure and its logs remain preserved; its root cause is undetermined.

## Interpretation limits

The raw-input feature map preserves distances before the final readout. The
readout can still annihilate directions; Transformer residual connections
already supply a direct route. No whole-network gradient guarantee follows.
Duplicate/linear/antipodal controls have algebraic redundancies; optimizer
coordinates and early learning speed can still differ. Better fixed-budget MSE
does not establish a larger function class, novelty or eventual convergence.

Timing reports synchronized native training, including optimizer update, after
50 warmup iterations. It does not measure inference latency. Peak allocated
GPU bytes and forward/backward timings are in the public result JSON.


## H094-H095: broader bases eliminated; simpler even-feature lead retained

The three richer rational/Hermite/trigonometric pair recipes are closed at the
fixed budget. They reduce aggregate reporting MSE by7.21-9.31% versus narrow
GELU, but a simpler duplicated-even control is stronger. The additional odd
basis does not earn its parameter allocation in this experiment.

The even control lowers aggregate MSE by12.92% versus narrow GELU and
8.89% versus full GELU at300 updates. This is a post-selection
synthetic lead, not a validated language architecture. A direct-GELU control
already improves7.58% versus narrow GELU, so gains cannot be assigned entirely
to a new nonlinearity. Core-linear is slightly better on the multiplicative
task, and the oscillatory task remains near the zero-predictor error.

The duplicate control folds exactly in real arithmetic:
A*x + V*E(U*x) + W*E(U*x) = A*x + (V+W)*E(U*x),
where E(z)=z^2/(1+abs(z)). Twelve selected checkpoints pass FP32 output/input-gradient
checks and reporting rescoring after folding; maximum absolute MSE change is
below1e-8. One FP64 function/Jacobian check and26 saved tensor-pair rechecks pass.

**Training used350,208 weights (70.31% fewer than full). Folding reduces stored
and inference weights to282,624 (76.04% fewer).** It is not evidence that training
from scratch with282,624 weights reproduces these results or AdamW trajectories.

Native FP32 batch256 inference is0.261 ms for folded even versus
0.297 ms for duplicate,0.129 ms for narrow GELU,
0.177 ms for full GELU and0.226 ms for full SwiGLU.
Folding lowers its own inference time by12.11%,
but it remains2.02 times narrow GELU. The compute
target is therefore unmet. This is a native inference measurement, separate
from the H094 training-time gate.

Keep the folded even-feature source as an unvalidated research lead. Do not
register a new model or automatically allocate refinement/longer training.
The three richer fixed recipes remain rejected. No test of a language corpus,
convergence, larger scale or broader real data occurred in these rounds.

[Detailed fitting results](input_basis_results.md),
[folding plan](input_basis_fold_plan.md),
[folded summary](../results/input_basis_fold_v1/summary.json),
[compact release receipt](evidence/input_basis_release.json).

The [complete336-rate table](../results/input_basis_fit_v1/metrics.csv.gz) and
[fold timing blocks](../results/input_basis_fold_v1/result.json.gz) are preserved
losslessly. [Artifact inventory](evidence/input_basis_artifacts.csv.gz) covers
1,297 local files and1,306,379,951 bytes. The inventory is not a backup of tensors.
