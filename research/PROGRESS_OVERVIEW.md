# Project progress - 2026-09-10

## Latest: checkpoint-input offload preserves gradients but misses the combined gate

[H121](checkpoint_input_offload_results.md) temporarily places eight block
checkpoint inputs in pinned CPU memory using an existing PyTorch API. With
native FP32 classifier loss, complete diagnostic-case peak falls by exactly
48 MiB (11.81–11.99%) on WikiText and TinyStories, with 1.21–2.34% event-time
overhead. With chunked loss it falls by only 14.47–15.40 MiB (4.98–5.19%).
**The all-four-fixture 10% memory gate fails.** No training extension is
earned under this protocol; no maintained default changes.

All traces confirm eight pinned [8,512,384] FP32 inputs, exact copy hashes,
one unpack each and zero logical payload after cleanup. Payload is 48 MiB;
rounded pinned-host allocation is about 64 MiB and its cache persists. FP64
finite differences and 24 NumPy gradient comparisons pass; maximum global
gradient relative L2 is 9.344e-8 and checked losses match exactly. GPU allocator
boundaries are zero and model/moment states remain unchanged.

The first audit process crashed during SymPy import before PyTorch/replay.
Its failure is preserved; a prospective recovery runs only the eight remaining
replays through the unchanged audit function. Total work: 262 backwards,
zero optimizer/training updates, 1,048,576 diagnostic target evaluations,
24 gradient artifacts and 824 full-model memory intervals. Repeated fixed-state
targets are not training exposure. The native import issue remains undiagnosed.

All 162 frozen sources and 61 maintained files stay intact. The previous
116-test pass is historical, not a new test execution. No parameter reduction,
endpoint quality improvement or novelty is claimed. H117's quality and H119's
active-clipping failures remain. A separately budgeted allocation timeline
could identify the remaining chunked backward peak before another change;
phase-boundary inventories alone do not uniquely explain it.
[Plan](checkpoint_input_offload_plan.md),
[recovery](checkpoint_input_offload_audit_recovery.md),
[receipt](../results/verification/checkpoint_input_offload_final_v1.json).

## Previous: optimizer temporary savings do not reduce the complete-job peak

[H120](optimizer_memory_results.md) compares default, per-tensor and fused
native AdamW on the same update800 states for WikiText/TinyStories, using both
native and chunked FP32 classifier losses. Twelve cases × 30 updates complete.
Backward sets all 12 job peaks. Fused saves 35–36 MiB within the optimizer phase,
but complete-job allocation rises 25 KiB; per-tensor mode leaves it identical.
**Both are ELIMINATED for the fixed 10% whole-job memory gate.**

Fused event-sum timing is 2.7–3.6% lower in the instrumented screen; this is not
an uninstrumented throughput claim. Per-tensor mode fails several timing gates.
All short NLL, stability, numerical, batch and scoring gates pass. These are
correlated continuations of one seed per corpus, not a broad quality result.
First-step clipping is inactive (norm<1), so its zero error does not resolve
H119's active CPU clipping failure.

The audit independently verifies first-step AdamW with NumPy FP64, every
first/final model/moment state, all 360 batches/samplers and 12 native final
scores. Total work is 360 updates/backwards / 1,474,560 targets, plus 24 study
and 12 audit scores. All 1,884 memory intervals and 26 zero CUDA boundaries
are retained. No scientific stage or case was repeated. A later read-only
PowerShell display failure is recorded separately.

All 152 frozen sources and 61 maintained files stay intact. No optimizer,
layer, parameter count or default is promoted; the historical 116-test result
is not represented as a new test run. H117's old quality gate remains failed.
Move the memory hypothesis to backward storage: inspect which checkpoint
inputs remain resident and test selective host storage only under a separate
budget with gradient/lifetime/transfer-cost checks. The eight input tensors
contain at most 48 MiB of payload; that is an estimate, not a measured saving.
The broad VRAM/quality and architectural goal remains open.
[Plan](optimizer_memory_plan.md),
[receipt](../results/verification/optimizer_memory_final_v1.json).

## Previous: first-step sensitivity supports a mechanism, not a training cure

[H119](adam_update_sensitivity_results.md) uses six frozen initial probe gradients
from the selected WikiText seed and 12 disposable native AdamW steps. Across all
nine cross-policy pairs, ideal first-update direction differences are amplified
**61.21–63.51x**; at least **99.888%** of squared difference comes from gradients
within 10 epsilon of zero. The direction difference is still only about 0.0017%.
This does not prove the cause of the later 800-step quality gap.

The independent NumPy calculation verifies all 45 direction pairs, 12 distortions,
15 native pairs and six device comparisons. All six CUDA cases pass the fixed
numerical checks. **The overall numerical audit fails**: CPU clipping differs
from FP64 by up to 6.872e-6, above the fixed 1e-6 bound. All parameter/moment
checks pass. Larger epsilons reduce discrepancy but change directions by
10.79%/32.76%, so both fixed low-distortion remedies are rejected.

The original process completed six CPU steps and then failed its CUDA-init
guard. Its outputs and exit are preserved; a frozen continuation runs only the
remaining six GPU cases through the unchanged function. Total is 12 disposable
optimizer steps, zero language-training updates/forwards/backwards/scores,
24 tensor artifacts and seven zero CUDA allocator boundaries. No tolerance or
case is repeated. All 145 original frozen sources and 61 maintained files remain
intact; recovery sources have their own manifest. The prior 116-test pass is
historical, with no new maintained-suite run.

H117's failed two-corpus gate stays failed, H118 stays inconclusive, and no
optimizer or model default is promoted. Return the memory investigation to
measuring optimizer temporaries versus persistent moments at the complete-update
peak under a separate plan; no new long training sweep is allocated. The broad
VRAM/quality and architectural goals remain open.
[Plan](adam_update_sensitivity_plan.md),
[recovery](adam_update_sensitivity_recovery_plan.md),
[evidence receipt](../results/verification/adam_update_sensitivity_final_v1.json).

## Previous: repeated execution exposes material trajectory variation

[H118](training_variability_results.md) adds four 800-update executions of the
failed WikiText seed101, using the same initial state, batches and unchanged
H117 runner. New chunk/native NLL gaps are **+1.143% and +0.940%**, versus the
original +1.352%. All three chunked endpoints are worse than all three native
endpoints, but the smallest cross-comparison gap is only 0.630%, below the fixed
1% rule. The preset diagnostic verdict is **INCONCLUSIVE**. These are repeats
of one deliberately selected seed, not three independent training seeds.

The native NLL range is 0.035284, smaller than the original 0.066509 pair gap.
Observed ranges do not overlap, so repeat variability does not span the entire
original gap. It still affects interpretation, and no unique cause is proved.
Within-policy initial gradients differ by at most 5.904e-8 symmetric relative L2;
median model distances reach 36–37% by step 800. First recorded scalar-loss
differences appear at steps 14–18. This does not establish catastrophic gradients.

All 12 new trained checkpoints, 13 native scores, four FP32 backward replays and
all 3,200 new sampled batches verify. The new runs preserve 26.816% lower allocated
memory versus native FP32, costing 12.65–17.65% more warmed update time. New
budget is 3,200 updates / 13,107,200 targets / 3,208 backwards, with no failed stage
or scientific retry. All 137 frozen sources and 61 maintained hashes are retained.

H117's failed WikiText/two-corpus gate stays failed; its scoped TinyStories
component remains separate. No new default, layer or parameter reduction is
claimed. Do not allocate more identical long repeats automatically. A bounded
first-Adam-update comparison on the six saved initial gradients is the next
specific diagnostic question and needs a separate plan. The broad goal remains
open; the prior 116-test result is historical, with no maintained-suite rerun.
[Plan](training_variability_plan.md),
[receipt](../results/verification/training_variability_final_v1.json).

## Previous: fresh replication retains memory savings but requires every seed's quality

[H117](fp32_training_replication_results.md) completes 18 fresh 800-update runs
on seeds 101/113/127 for both WikiText-2 and TinyStories. Full-job tensor allocation
falls 26.98–27.75% versus BF16 native, with practical warmed update time.
wikitext2 fails the fixed gates; tinystories qualifies in this scope. Failed gate names: final_quality_bf16, final_quality_native.
The two-corpus recipe does not qualify.

The candidate and both references have identical 9,099,648 parameters. Initial
FP32 gradient fidelity does not guarantee the endpoint after 800 updates.
Independent verification covers six exact initial-state regenerations, 54
trained model/Adam/sampler checkpoints, 60 complete native validation scores,
12 FP32 initial-gradient replays and all 14,400 sampled batches. Three seeds,
reused development validation and no complete-trajectory repeats limit the claim.

The original startup failed after 24 CPU qualification backwards and before
any training update because environment metadata initialized CUDA too early.
Its frozen recovery reorders CPU initialization before metadata; no training,
scientific recipe or tolerance is repeated or changed. Total backwards across
attempts: 14,478; recorded memory intervals: 28,998. Original failure is retained.
The initial figure process later hit a Matplotlib import access violation;
one frozen-input CPU recovery ran the unchanged plot successfully. No training
or audit was repeated and no runtime cure is claimed.

No default or maintained model is promoted. All 61 maintained files remain
unchanged from the earlier 116-test pass; that suite is not rerun. Keep the
scoped memory evidence and close failed fixed quality claims. Paired versus
within-policy trajectory variability is the next bounded diagnostic question,
before any further broad training sweep. The broad goal remains unmet.
[Plan](fp32_training_replication_plan.md),
[recovery](fp32_training_replication_recovery_plan.md),
[receipt](../results/verification/fp32_training_replication_final_v1.json).

## Previous result: approximately 27% lower allocation at comparable update time

[H116](fp32_decoder_resource_results.md) turns the prior gradient diagnosis into
a passing resource screen for narrow context-512 models. Default FP32 chunks
save 26.97% / 27.23% job allocation on WikiText/TinyStories, with -0.12% / +0.33%
median update-time changes versus BF16 native. Same-precision native controls
also pass the memory/time comparison. Every numerical and short-NLL gate passes.

The 30 cases contain 1,500 updates. Independent verification covers 24 FP32
backward replays, 36 native scores, all batches and endpoint model/Adam states.
Math attention and full GELU/SwiGLU fail the memory target and are eliminated
from this screen. Four narrow fixtures are correlated seed61 pairs, so the
next earned work is broader replication and duration, not a general quality
claim. This establishes a promising execution component with unchanged
parameter count; the full research objective remains unmet.

## Previous result: decoder precision passes diagnosis; resource test is next

[H115](decoder_gradient_transport_results.md) separates two numerical effects.
On fixed incoming gradients, default BF16 decoder repeats sometimes vary.
Forced math attention removes the observed repeat variability but leaves
roughly 0.003 paired gradient error from roughly 1e-6 classifier differences.
Removing checkpointing does not solve the problem. Both FP32 decoder modes
pass unchanged transport/replay thresholds across all 30 diagnostic conditions.

An elementary rounding witness produces exactly 32,768-fold perturbation
amplification in CPU/CUDA BF16 linear backward. This explains a possible
mechanism, without claiming a new theorem or isolating every H114 kernel cause.
The study uses 320 backwards including qualification, **zero optimizer updates**,
and earns only a separate uninstrumented whole-job memory/runtime screen.
All 30 native hidden-state recaptures and stored first-gradient arithmetic verify.
Actual lower VRAM with preserved quality remains the primary unmet objective.


## Previous result: 32-33% lower memory, but full-gradient qualification fails

[H114](fp32_classifier_profile_results.md) completes 56 matched 50-update
continuations in 502 seconds. A BF16 decoder with FP32 classifier chunks keeps
about one-third lower whole-job allocation on narrow T512 models and costs
roughly 13% more update time. All short native validation changes are tiny.
Full T128 GELU/SwiGLU achieve only 0.53-1.08% job saving, below the 15% gate.

The stronger test rejects the fixed precision recipe: every scope fails the
original full-model gradient threshold. Exact and numerical gradient replays
also fail; both are preserved and the cause remains unresolved. The successful
completion audit covers native scores, saved states/gradients and batches, not
complete backward reproducibility. No fresh training or new architecture is
earned. The next useful question is why the matched BF16 decoder gradients
change, while preserving the demonstrated memory component and every failed gate.

## Previous classifier diagnosis

**Measured VRAM remains primary; the broader goal is unmet.** H113 identifies
one numerical cost of the memory-saving classifier: a shared BF16 weight cast
accumulates chunk gradients before converting back to FP32. All 24 saved-state
comparisons reproduce the effect, with unchanged loss and hidden gradients.
The independent audit checks 144 gradient reruns and 24 explicit derivative
references. No optimizer update is performed.

Disabling caching alone fails the preset local improvement/runtime gates.
FP32 classifier chunks pass them, with about 72% lower **local classifier**
allocation and roughly 4e-7 weight-gradient error. This is not a whole-job
memory or language result. Individual timing ratios range from 0.335 to
15.862, making current speed evidence weak despite the passing median rule.
The next qualified step is full-model gradient and resource measurement,
with sustained warmup and separate wall/GPU-event timing.
[Results, proof and all-fixture figure](classifier_precision_results.md).

## Previous whole-job memory result

**Measured VRAM remains primary; the broader goal is unmet.** H112 completes
12 fresh 800-update trials on WikiText-2 and TinyStories with three seeds each.
Classifier/loss chunking reduces actual measured job tensor allocation by
32.43% / 33.06%, with 13.65-14.52% slower updates. Both arms use the same
memory-aware evaluator and unchanged narrow GELU model.

TinyStories passes every quality/memory/time gate. Its NLL changes range from
-0.514% to +0.316%. WikiText misses the 1% quality allowance in two seeds,
losing 1.465% and 1.954%. The fixed two-corpus recipe is rejected. The useful
result is a scoped TinyStories execution component, with no new architecture,
parameter saving or general superiority claim.

All 72 native scores, 48 saved checkpoints, twelve initializations and 9,600
sampled batches pass independent audit. Training takes 15.37 minutes and is
not repeated. Native startup failures and explicit recoveries are retained;
their root cause remains unknown. Tensor allocations exclude driver/context
memory. Trials use fresh models in one process with 25 verified empty tensor
storage boundaries; library workspaces are fully charged during measurement.
[Complete report and every seed](whole_job_memory_results.md).

H111 first verified the evaluator on 24 existing states with no learning.
Classifier streaming saves 16-31% evaluation allocation at roughly 1% median
time overhead. Sequence microbatching saves slightly more but costs 3.7-7.1x,
so it fails. This resolves H110's evaluation bottleneck without erasing its
earlier fidelity failure. [Evaluation report](streamed_evaluation_results.md),
[earlier duration result](token_memory_duration_results.md).

The maintained code, two model folders, five variants and eight recipes remain
unchanged from the prior 116-test pass. A numerical-gradient diagnosis is the
next justified question, before further language training is allocated.

## Previous derivative-initializer learning result

**Measured VRAM remains primary; the overall goal is unmet.** H109 tests the
promising derivative-aware initializer under actual learning on 18 fresh input
sets. It completes 144 fits, 86,400 updates and 72 selections in 6.05 minutes.
Its 6-10% initial derivative advantage shrinks to 0.4-0.7%, failing the fixed gate.

The candidates' output error improves about 35% from initialization, while their
directional-derivative error worsens 7-8% for GELU and 51-52% for SwiGLU. Local
fitting uses 41-43% less tensor allocation than full FFNs, but ordinary narrow
controls share that saving and absolute quality remains inadequate. Capture
raises the pipeline peak to 117.908 MiB. Neither learning recipe earns language
insertion; H108's zero-SGD finding remains a separate promising component.

All 648 endpoint metrics, 216 states and 18 exact recaptures pass independent
audit. The preflight path-type failure is retained; actual training runs once.
The maintained source and tests are unchanged from the prior 116-test pass.
[Results, variation and diagnostics](sobolev_learning_results.md).

## Previous derivative-aware calibration result

**Measured VRAM remains primary; the overall goal is unmet.** H108 finds a useful
compression component: fit a narrowed FFN's output projection to preserve both
values and directional derivatives. In 432 zero-SGD comparisons, derivative
error improves 6.83-16.15% versus value-only fitting across both teacher families,
all three widths and every sampling-seed aggregate. Output error also improves.
Readout fitting alone explains about 88-90% of the combined derivative gain.

The approximately 71%-smaller FFNs save 33-35% of local inference allocation,
but their quality fails the fixed absolute gates. Larger widths also fail at
least one quality/memory gate. The complete capture/calibration pipeline peaks
at 117.908 MiB; no training-memory or language result is claimed. Keep the
derivative-aware readout as a promising component, with no model promotion.

All 864 scores and 432 exports pass independent auditing. The study uses three
sampling seeds and three depths from each of two existing teachers, whose
training seed remains fixed. It takes 231.87 seconds; there are zero SGD updates.
The original batch-partition audit failure is preserved alongside the successful
recovery. Active code/tests remain unchanged from the prior 116-test pass.
[Report, equations and ablations](sobolev_selection_results.md).

## Previous affine-residual result

**Measured VRAM remains primary; the overall goal is unmet.** H106/H107 test a
full-rank affine path plus a small nonlinear correction. The capacity screen
qualifies two widths, but the subsequent 288 fresh fits reject both against six
conventional controls. They use about 70.8% fewer FFN parameters and 42-45% less
local training allocation than full references, while ordinary narrow controls
offer better reconstruction at almost the same memory. Candidate inference is
1.64-2.32 times narrow GELU's cost. Full teacher capture raises the pipeline peak
to 117.91 MiB; the local fitting saving is not an end-to-end memory result.

The 600-update, batch-256 comparison uses three disjoint sampling seeds, three
depths, two existing teachers and two rates. It completes in 12.98 minutes. All
864 scores, 144 selections/exports and 18 recaptured datasets pass independent
verification; 116 tests pass. The original native postprocessing failures remain
recorded separately from successful training. No active variant is added.
[Complete report](affine_residual_fit_results.md),
[finite-data proof](affine_residual_capacity_results.md).

## Previous real-input projection result

H105 tests whether synthetic feature-discovery gains transfer to real FFN inputs.
All energy projection recipes fail: 378 comparisons across two existing trained
teachers, three layers, three calibration seeds and three ranks finish in 82.18 s
without SGD. Independent audit recaptures all 18 datasets and checks all scores.

A post-screen bound finds that any rank-64 global output bottleneck has mean
normalized function-error floors of 10.98% / 21.25% for GELU / SwiGLU on these
data, above the 5% local threshold. All 54 rank bounds agree with direct SVD.
The next architecture must avoid this restrictive global output bottleneck or
save memory another way. This is a local capacity result, not an NLL theorem.
116 maintained tests pass. [Real-data report](real_subspace_results.md).

## Previous synthetic feature-discovery result

**Measured VRAM remains primary; the overall target is unmet.** H103/H104 tests
supervised feature discovery and a 66,048-parameter factorized FFN. The cubic
variant has 94.41% fewer parameters and 15.97% lower aggregate error than an
equally initialized full SwiGLU on four synthetic Gaussian tasks. It still fails
its fixed promotion criteria: a tiny cubic control is better on cubic, and
inference costs 2.12 times narrow GELU. No language-model claim follows.

Including initialization, peak tensor allocation saves 25.02% / 22.68% versus
full GELU / SwiGLU; equally initialized small controls have the same pipeline
peak. All 408 fits, 816 checkpoint scores and 204 selections are audited. Fits
use 300 updates of batch 256, three independent dataset seeds and two rates:
122,400 updates and 31,334,400 training example presentations in 10.38 minutes,
plus the separately recorded diagnostic screen/audit. [Report](spectral_fitting_results.md).

## Previous scoped memory result

**Actual VRAM reduction is now primary.** H101/H102 measure token-local execution
without changing model parameters. Narrow GELU qualifies in a short three-seed
comparison: 21.01% / 32.23% lower peak training allocation at contexts 128 / 512
versus whole-block checkpointing, with 7.45-22.51% longer updates and no more
than +0.189% final NLL change. This is 12-update evidence, not converged quality.

The initial grid stops after 22 cases because full GELU has a reproducible
fidelity failure. A focused 36-case replication completes; all 38 saved endpoint
scores, including two diagnostic replays, reproduce independently. Full SwiGLU
fails the all-seed time gate. No model/default is replaced and no novelty is
claimed. [Full memory report](token_memory_results.md).

## Architectural objective

We have working parameter compression, but the complete research target remains
unmet: at least 70% fewer FFN weights, within 1% relative validation loss of BOTH
full GELU and SwiGLU, while beating calibrated narrow controls. Consistent
multi-seed, longer-training, scale and broader-data evidence is still required.

## Latest result: three-way interactions fail despite representational capacity

H100 tests raw and bounded products of three learned projections on fresh
Gaussian tasks with hidden rotated directions. Both have 351,276 parameters,
70.27% fewer than full GELU. Both are rejected: aggregate MSE is 2.079/4.543 times
narrow GELU and native update time is 1.562/1.881 times its cost.

The 312-run screen uses four tasks, three seeds, two rates, batch 256 and 300 updates,
taking 6.97 minutes. All 30 qualification checks and 177 saved pairs pass. Independent
audit reproduces 624 checkpoint scores and 156 rate selections exactly.

A privileged construction proves raw products can represent the cubic target;
randomly initialized training does not find it adequately. A plain cubic ridge
partially learns cubic (18.02% less error than zero prediction) but misses the
50% reduction gate. Low final cubic feature alignment motivates a distinct
feature-discovery investigation, not further refinement of rejected products.

Post-audit report imports fail separately; preserved traces and a checksum-checker
correction lead to successful NumPy diagnostics under Python 3.12.9. Original
training and the independent Torch audit are unchanged. No active model is added.
The language, convergence, compute, scale and broader-data goal remains unmet.

[Full results and failure record](triadic_interaction_results.md),
[frozen plan](triadic_interaction_plan.md),
[audited result](../results/triadic_interaction_v1/result.json).

## Previous result: nonlinear residual learning, but no activation promotion

H099 completes 264 fresh runs: eleven forms, four Gaussian nonlinear-residual
tasks, three seeds and two rates, each with 300 updates and batch 256. The study
takes 5.76 minutes. All 25 qualification checks and 163 saved tensor comparisons
pass; independent audit reproduces 528 checkpoint scores and 132 selections.

A four-parameter learned rational mixture uses 351,052 parameters, 70.29% fewer
than full GELU. Aggregate MSE improves 12.39% versus narrow GELU and 13.79%
versus narrow SwiGLU, but only 0.060% versus established StarReLU. It loses one
seed to StarReLU, regresses on individual tasks and costs 40.49% more native
update time than narrow GELU. Its fixed recipe is rejected at this budget.

Nonlinear learning beyond affine structure is demonstrated on three tasks.
Cubic remains near zero-predictor error for every ordinary model, although the
privileged known-feature readout learns it. This exposes a feature-discovery or
optimization question, not evidence that a new activation solved cubic learning.
All full and narrow models have biases and start with zero output; this is a
changed, explicitly documented assay, not a direct continuation of H094 scores.

Learned curves and gradients are recorded. No active model is added and no
kernel or longer run is allocated to the rejected mixture. The full parameter,
language NLL, compute, convergence, scale and broader-data goal remains unmet.

[Complete findings](nonlinear_residual_results.md),
[frozen plan](nonlinear_residual_plan.md),
[audited result](../results/nonlinear_residual_v1/result.json).

## Previous result: the affine control closes the older general lead

H096-H098 fits ordinary linear and affine controls to exactly the training
samples seen by each H094 run. The even-feature recipe is only **1.724% better
than affine least squares** in aggregate, missing the frozen 2% material-benefit
gate. It wins on piecewise targets by 12.60% but loses on smooth, oscillatory and
multiplicative targets. Its three small aggregate seed wins remain recorded.

The affine control needs 147,840 coefficients, versus 350,208 used in the neural
training and 282, 624 after folding. The previously measured native speed deficit
also stands. Close the general nonlinear-benefit claim for this fixed recipe;
retain the piecewise result as scoped evidence, without kernel refinement or
automatic longer training. The original 12.92% advantage over narrow GELU is
numerically correct but does not survive as a sufficiently strong general claim.

All 24 least-squares fits are independently verified, including reconstructed
sample counts/statistics, 96 CPU score checks, 168 recomputed parity evaluations
and 12 saved odd-linearity pairs. H096's native access violation is preserved;
H097 did not reproduce it, and H098's explicit CPU-solver recovery passed the
unchanged mathematical/statistical checks. No new neural optimizer update ran.

The next assay must measure nonlinear residual learning beyond affine structure,
include a representable positive control and stronger conventional activations,
and separate representational limits from short-budget optimization. The full
parameter, NLL, compute, duration, scale and broader-data goal remains unmet.

[Complete findings](affine_falsification_results.md),
[frozen diagnostic](affine_falsification_plan.md),
[recovery](affine_falsification_recovery_plan.md),
[audited result](../results/affine_falsification_recovery_v1/result.json).

## Previous experiment: direct-input nonlinear bases

H094 tests rational, Hermite and trigonometric feature pairs without a fixed
ReLU/GELU/SwiGLU candidate base. The336-run grid uses the same four synthetic
families with fresh input seed9844:65,536 train /4,096 selection /4,096 reporting
samples, batch 256, three optimization seeds, two rates and 300 updates per run.
It completes100,800 updates in5.82 minutes. All672 independent checkpoint scores
and336 initializations reproduce exactly.

All three richer recipes fail their full promotion gates. The simpler even-feature
control is the useful finding:12.92% lower aggregate MSE than narrow GELU and
8.89% lower than full GELU at this short budget. Post-training folding reduces
stored weights from350,208 to282, 624 (76.04% fewer than full) with less than1e-8
MSE change across12 checkpoints. Training still used350,208 weights.

The folded model's native batch 256 inference is0.261 ms versus0.129 ms for narrow
GELU. H098's stronger affine control subsequently closes its general-benefit claim;
the piecewise gain remains a scoped observation. [Full report](input_basis_results.md).

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
