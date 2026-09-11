# H109: does derivative-aware initialization survive actual learning?

Previous goal turn: progress. H108 audited 432 compressed FFNs and established a
useful local calibration component, but no complete quality/memory gate passed.
The full VRAM/quality objective and original >=70%-FFN parameter target remain
unchanged. This is a distinct learning test of the qualified component, not an
automatic language insertion or a relaxed version of H108.

## Hypothesis and scope

An output projection initialized to match teacher values and derivatives may
retain a useful advantage after all narrow-FFN weights learn from fresh value
targets. Alternatively, ordinary optimization may erase that advantage; then
the derivative-aware initializer does not justify extra preparation for this
training recipe, even if the zero-SGD result survives.

[Sobolev training](https://arxiv.org/abs/1706.04859) and
[GRAIL](https://proceedings.mlr.press/v328/tang26a.html) supply the derivative
distillation and calibrated reconstruction precedents. No new activation,
selection theorem, pruning principle or universal gradient guarantee is claimed.
The H108 Schur identity proves optimal readout gains on a fixed feature set;
it says nothing about the subsequent nonconvex training trajectory.

## Frozen experimental design

Two existing seed-17 teachers: full GELU and full SwiGLU. Three depths 0/3/7,
sampling seeds 71/83/97, and primary widths 448/298 only. These give 344,448 /
343,680 parameters including the output bias, versus 1,179,648 teacher FFN
parameters: reductions 70.8008% / 70.8659%. This tests the actual architectural
count target. There is no width or derivative-weight search.

Four initializations are loaded byte-for-byte from audited H108 exports:

1. Value-greedy teacher neurons, value-only ridge readout: primary control.
2. The identical value-greedy neurons, Sobolev readout: simpler candidate.
3. Sobolev-greedy neurons and Sobolev readout: combined candidate/ablation.
4. Random teacher-neuron subset and value readout: static control.

For each teacher/depth/seed, all shapes and parameter counts are identical.
No initialization receives new calibration or a reporting-based choice. The
old H108 calibration costs and full-teacher capture are included in preparation
accounting, without rerunning those scientific trials. All four are ordinary
dense GELU/SwiGLU FFNs; this does not compare every scalar activation or qualify
a new architecture against SOTA. Earlier H107 broader controls remain relevant.

Select 960 distinct 128-token training-cache windows, excluding every H105 and
H107 collection window. Use a single CPU permutation seed 18001, divided into
three nonoverlapping blocks of 320 windows for seeds 71/83/97. Within each block:
first 32,768 token rows train, next 4,096 select rate/checkpoint, last 4,096 report.
Both teachers use identical window IDs. Store explicit IDs and verify exclusions.
Fresh windows are new local reconstruction data, not an independent language
holdout: the teachers already trained on this corpus, and their training seed
is fixed. H108 initialization also carries prior development decisions.

Capture raw FFN inputs through the existing decoder in BF16, then recompute
smooth FP32 teacher outputs, TF32 off, batch512. Old BF16 labels are discarded.
Use H108 calibration output RMS sy to normalize the loss: ||f(x)-teacher(x)||²/sy².
Raw inputs and exported parameters are retained unchanged. Division of both
predictions and targets by the same positive sy is algebraically this objective;
no extra trainable coefficient is introduced.

Train all parameters with the existing `fit_regression` loop: 600 updates,
batch256, AdamW betas (0.9,0.95), epsilon1e-8, zero weight decay, global norm clip1,
constant LR0.001 or0.003. No derivative term is used in learning. The same saved
CPU index stream (seed31000+sampling_seed) is reused for all four methods and
both rates. Every rate starts from the original initialization, with fresh Adam
state. Training data stays on CPU between batches. Rotate method execution order
by seed to reduce systematic timing order effects; this is not randomized hardware
replication. Use FP32, four CPU threads, one GPU job, recorded malloc allocator.

Each method chooses among initialization and its two final endpoints solely by
selection value MSE, ties favoring the earlier endpoint. Report all endpoints,
including unselected rates and any zero-update winner. There are 144 fresh fits,
86,400 optimizer updates, 72 selected models and 216 distinct endpoint states.
Full FFNs receive separate identical 30-update LR0 profiles (540 total profile
updates) to allocate Adam/gradient storage without changing teacher weights.
No full teacher is retrained to improve quality.

## Measurements and numerical qualification

Record every training loss, preclip gradient norm, update/forward/backward timing,
finite weights/optimizer state, clipping frequency and activation/per-parameter
gradient summaries at steps0/50/150/600. Keep endpoint values, selection MSE,
reporting MSE/sy², and two-direction reporting JVP error normalized by teacher
JVP energy. Probes use seeds32000+seed and33000+seed with H108 input SD scaling;
they are independent of calibration probes. No report derivative is used in
training, endpoint selection or rate tuning.

The JVP metric estimates average standardized sensitivity, not a worst-case or
deep-network gradient bound. Record per-sample maximum errors as diagnostics.
All models and targets must remain finite. Qualify loss-scaling algebra,
parameter-gradient scaling, one-step optimizer identity, raw deployment identity,
counts and JVPs on small fixtures before loading fresh real data.

Profile selected raw models and original full FFNs identically at batch256:
ten warmups, five synchronized blocks of twenty forward calls. Record allocated
and reserved tensor peaks, parameter/optimizer bytes, exact state/data/source
hashes, and all preparation costs. Report the complete pipeline peak as the max
of prior capture/calibration, fresh capture/target generation, fitting and
inference. This does not include original teacher pretraining or whole-process
driver VRAM. CPU working sets are distinct from CUDA allocation.

## Predeclared decisions

A derivative-aware initializer qualifies for further learning investigation only
if BOTH teachers satisfy, for selected models: paired geometric-mean reporting
JVP error <=95% of value-greedy, value error <=101% of value-greedy, every sampling
seed meets both comparisons, and neither static random nor value-greedy strictly
Pareto-dominates it in the two errors. Require all finite and mean update cost
<=110% of value-greedy. Also report equal-rate comparisons to expose selection
artifacts. Failing closes the fixed initializer-plus-value-learning recipe;
it does not erase H108's zero-SGD component result or prove derivative training
can never help. No automatic added derivatives, more steps, rates or beta search.

A method earns a separately frozen language-insertion diagnostic only if BOTH
teachers and each seed aggregate have mean value error<=0.05 and derivative
error<=0.10; training and inference allocation each<=90% of full; raw inference
latency<=110% of full; and >=70% actual FFN parameter reduction. Derivative
candidates must first pass the comparative component rule. Value-greedy is
eligible as an established control under identical absolute rules. Passing
would allocate a diagnostic, never prove preserved language NLL, broad capability,
scale, novelty or completion of the overall goal. Preparation peaks/costs remain
reported separately even when local training memory passes.

Independent audit must recapture all fresh inputs, recompute smooth targets,
check exact initializations/shared streams, rescore every selection/report value
and reporting JVP using autograd, and verify all endpoint selections. Preserve
all failures, sources and partial artifacts without silent training reruns.
