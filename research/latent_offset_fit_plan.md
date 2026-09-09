# H085 - Controlled training of internal offset and gain

Freeze before implementation or new fitting. The previous goal turn made
PROGRESS: H084 established algebraic gain redundancy and measured separate
gain/offset checkpoint dependence. It did not promote a model or establish
training causality. H083's two curved recipes and earlier outer-affine failures
remain closed. This is a distinct controlled training hypothesis, not a longer
retry of either curve.

## Hypothesis and design

An internal offset can add a nonzero first-order response to an isolated
bias-free SwiGLU, while internal gain alone does not enlarge its function family.
H084 shows both components affect trained checkpoints. Hypothesis: training only
the offset, possibly with a simpler fixed factor-step correction, retains the
useful expressivity with half the learned controls and improves on conventional
parameter-efficient baselines. Failure against narrow GELU rejects promotion.
The [H084 derivation and limits](latent_affine_mechanism_plan.md) remain binding;
this is not a new activation-family or global-gradient claim.

All forms use one width-384 FFN. The structured forms use G=8, hidden2048,
middle384, existing factor initialization and SwiGLU. Order and counts:

| Form | Learned weights | Purpose |
|---|---:|---|
| plain | 350208 | Original structured control |
| plain_first_lr | 350208 | Same function and initialization, fixed first-factor LR control |
| latent_gain | 350232 | Learn only 24 group gains; offset fixed at zero |
| latent_offset | 350232 | Learn only 24 group offsets; gain fixed at one |
| latent_offset_first_lr | 350232 | Offset plus the same fixed first-factor LR correction |
| latent_affine | 350256 | Original full affine control, both sets learned |
| narrow_swiglu | 350208 | Hidden304 conventional parameter control |
| narrow_gelu | 350208 | Hidden456 conventional parameter control |
| full_swiglu | 1179648 | Hidden1024 full control |
| full_gelu | 1179648 | Hidden1536 full control |

Preserve the H082 affine primitive phi(z)=z+0.5*tanh(theta_a)*z+
0.5*tanh(theta_b), its native arithmetic and placement before the middle shuffle.
For gain-only convert theta_b from Parameter to an immutable zero buffer; for
offset-only convert theta_a instead. This keeps the same forward arithmetic and
retained gradient calculations for controlled training. It leaves 24 fixed zero
buffer elements per FFN; these are not learned weights. No dedicated optimized
offset kernel is introduced. All controls start at zero, so every structured
form shares its common initial weights and initial function with plain.

The two first_lr forms multiply the established learning rate of each first
factor, and only those factors, by 5/8. This gives first-factor multiplier2.5
instead of4. Second-factor multipliers remain4 for up/gate and2048/96 for down;
shape multiplier remains1. The choice5/8 is a simple value inside H083's observed
affine slope range0.543..0.726, selected as a hypothesis before this new dataset
or its results. It is history-informed, not a tuned optimum or exact replica of
adaptive gains. Parameters/initialization, epsilon, clipping and decay are not
changed. Fixed gains and Adam coordinate changes need not reproduce this
optimizer exactly; only the explicit per-parameter LR change is claimed.

The candidate forms are latent_offset and latent_offset_first_lr. Gain-only,
full affine and the plain LR variant are diagnostic controls, not additional
promoted architectures. Offset/gain have exactly equal learned counts. Each
exceeds plain/narrow by24 weights (about0.00685%); report this small mismatch.
Full affine has24 more than offset. No padding or falsely exact budget claim.

## Fresh data and equal allocation

Generate a fresh fixed dataset with CPU seed9813 rather than H083's9812:
73,728x384 Uniform[-sqrt(3),sqrt(3)] inputs. Keep permutationseed9283 and output
QR rotationseed9282; keep H083's four analytic tasks in order: smooth,
oscillatory, multiplicative, piecewise. Use the first65,536 training examples,
next4,096 rate-selection examples and last4,096 reporting examples. Divide each
output by its training population standard deviation without centering.
No new teacher, target expression, permutation search or linear task is added.
Fresh inputs avoid reusing H083's reporting examples. These are still synthetic
tasks from one distribution draw, with three optimization seeds, not new corpora.

For every form and task, use seeds17/29/43 and both base rates0.001/0.003,
600 updates, batch256. CPU samplerseed20000+seed yields600x256 indices shared
across all forms/tasks/rates. There are240 fresh cells,144,000 updates and
36,864,000 example presentations. No language or full-model resource workers.
Order task, seed, form, rate. Retain all runs and all final checkpoints.

Use H073's numerical training loop through an isolated binding adapter: FP32
CUDA, TF32 off, four CPU threads, AdamW betas(0.9,0.95), epsilon1e-8, zero decay,
clip1, constant base rate, existing fan-in and narrow-width calibration. No
compiler, training checkpoint recomputation or schedule change. Timings exclude
the first50 updates, with CUDA synchronization and peak reset as before.
Use UV and one GPU worker at a time.

Select lower final selection MSE per task/form/seed, tie to lower rate; persist
the choice before scoring either rate on reporting rows. There are120 choices.
There are no intermediate checkpoints selected by validation. Independent
analysis must regenerate data/streams, verify all240 checkpoints/optimizers/
histories and reproduce all240 selection scores and120 selected reporting
scores exactly. Do not compare absolute MSE with H083 as if the datasets match.

For48 selected learned-control cases (gain, offset, offset_first_lr, affine),
perform one reporting reset: zero all trainable shape controls, retaining all
factor weights and fixed buffers. This measures final dependence, not causal
training advantage. Persist original/zeroed MSE and verify all48 independently.
Save learned gain/offset values, buffer/parameter classification, sampled slopes,
activation and gradient statistics, clipping, finite values and actual resource
costs. Full checkpoint and optimizer state, all600-step histories, RNG, initial/
final hashes and source/data hashes remain mandatory.

## Qualification before fitting

Eight pytest cases, zero optimizer updates:

1. Count, state/initial-function agreement and exact optimizer parameter coverage
   across all10 forms and3 seeds. Check every LR/decay rule, fixed-buffer zero
   values, and learned-control membership. Initial shared parameter bytes match.
2-5. Gain-only and offset-only, each CPU FP64 and CUDA BF16: compare to the full
   affine control with the corresponding unused control fixed at zero. Use
   theta values linspace(-0.4,0.3,8) independently in all three projections.
   Require exact outputs, input gradients and every retained parameter gradient.
   For the same four cases, checkpointed/eager output and all gradients must
   match exactly. Save all four tensor payloads; these are standalone checks,
   not full-Transformer resource qualification. CPU probe shape2x7x384;
   GPU probe shape16x128x384; inputgenerator9384, modelseed17.
6. FP64 finite-difference check of both active scalar control families and input:
   shape2x16,4groups, epsilon1e-6, atol1e-5, rtol1e-3, fast mode. Record actual
   function-call counts; no optimizer update.
7. Fresh data seed differs from H083, deterministic permutation/streams and
   train-only normalization. Use a small input probe and an out-of-training
   sentinel for this check; the actual dataset is generated only after passing.
8. Selection and candidate gates reject incomplete/nonfinite grids, failed
   positive controls and improvements that do not beat gain/fixed-LR/narrow
   controls. Keep primary and rate-controlled conclusions separate.

## Frozen quality decisions

Positive-control assay: selected full GELU's three-seed geometric reporting
MSE ratio to the zero-output predictor <=0.98 on at least two of four tasks.
If this fails, label INCONCLUSIVE_ASSAY_FAILURE and promote nothing. Otherwise,
for EACH offset candidate require all of:

1. Complete finite grid and all eight qualification checks.
2. Geometric reporting MSE over12 paired task/seed endpoints <=0.98 of plain,
   with each seed's four-task aggregate strictly better than plain.
3. Aggregate <=0.99 of equal-count latent_gain, and <=0.99 of plain_first_lr.
4. Aggregate strictly beats BOTH calibrated narrow controls.
5. Aggregate <=1.01 of full latent_affine, despite using24 fewer parameters.
6. Each task's three-seed ratio <=1.05 versus plain AND the zero predictor.
7. At least70% fewer learned FFN weights than the full references.

These gates use separately selected rates with equal tuning effort and support
only a candidate-recipe comparison. Additionally report BOTH individual-rate
paired ratios, per seed and task, and the matched-rate effect of offset versus
plain (or offset_first_lr versus plain_first_lr). A claim of robust offset
benefit requires >=1% aggregate improvement over its matching no-offset form
at EACH rate, with all three seeds improving at EACH rate. Report gain versus
plain at both rates too. Failure of this additional robustness condition blocks
an activation-specific claim and blocks resource promotion, even if the selected
recipe gates pass. No NLL1% threshold is applied to synthetic MSE.

A qualifying offset recipe only earns a separately frozen full-model resource
study, followed by a language screen if resources qualify. Full controls, gain,
affine and plain_lr remain reported regardless of outcome. No old recipe is
reopened, no extra rate/step/kernel is allocated automatically, and no causal
generality or convergence follows from a600-step screen.

## Preservation and closeout

Isolate four new files(model.py,study.py,test_harness.py,launch.py) under
results/latent_offset_fit_v1/source. Pin H084 final/result/analysis/source/support
and H083 data/results/source anchors. Preserve prior131 scientific sources and
73 plans; add four sources for135 total and this plan for74 total. Archive
navigation before updates. Durable writes use fsync, ZIP validation and semantic
readback, and coordinator records actual PID/UTC/log/return state. Stop on any
runtime, finite-value or fidelity failure; preserve partial output. No automatic
retry or tolerance relaxation. Inspect actual handles before any recovery.

After independent analysis, report all rate choices, candidate gates, fixed-rate
effects, resets, GPU resources and limitations; update current state, overview,
activation guide and ledger. Retain three active model folders, six variants,
nine recipes. The broad research objective remains unchanged and unmet.
