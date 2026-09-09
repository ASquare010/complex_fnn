# H077 - Longer three-seed replication of matched-budget BlockShuffle GELU

Freeze before new implementation or language scoring. H076 completes and
independently verifies all 21 short-screen trials. Matched-budget GELU alone
passes; its 0.1167% selected narrow-GELU advantage is small. The previous turn
made PROGRESS. Preserve the failed same-width recipe and the H051 longer plain
failure. No active model folder, registered variant or recipe is added.

## Fixed cohort and allocation

Run six forms at seeds 17, 29 and 43 for 800 updates each: full SwiGLU h1024,
full GELU h1536, narrow SwiGLU h304, narrow GELU h456, plain BlockShuffle SwiGLU
h2048 and matched-budget BlockShuffle GELU h3264. All 18 runs start fresh;
none continues a short-screen checkpoint. Each sees 1,638,400 training targets:
29,491,200 targets and 14,400 optimizer updates in total.

Peak rates are frozen from H076's minimum final validation NLL within each form,
with equal three-rate search effort: 0.0012 for full SwiGLU, full GELU, narrow
SwiGLU and matched GELU; 0.0006 for narrow GELU and plain. No new rate selection
uses these longer endpoints. The seed-17 short screen already influenced this
choice; seeds 29 and 43 are additional training-seed replications, not a new
validation set. Report selected-rate comparisons as fixed recipes, not proof
of optimal rates or a pure activation-only ablation.

Seed-17 order: full SwiGLU, narrow SwiGLU, plain, matched GELU, narrow GELU,
full GELU. Seed 29 reverses this order. Seed 43 rotates the seed-17 order by
three positions. Run one GPU worker at a time through a hidden durable
coordinator. The allocation is bounded to 18 runs, with no automatic retry,
extension, additional rate, width, activation or optimizer repair.

## Unchanged computation and explicit schedule changes

Reuse H076's existing model constructor, scalar observers, two-projection
operation counter and unchanged src/core trainer. Preserve batch 16, context
128, d384/L8/heads6/vocab4096, structured groups 8, native BF16 with FP32
parameters, TF32 off, four CPU threads, AdamW (0.9, 0.95), eps 1e-8, base decay
0.1 and global clip 1. Name-local Transformer initialization and down residual
scale 1/4 are unchanged. For seed 17, initial weights must match H076 exactly.
For every seed, all non-FFN tensors must agree across the six forms.

Structured projections retain fan-in factor LR correction and parameter decay.
Both narrow controls retain exactly-once fan-in down initialization/LR correction
and product decay, so down weight_decay * lr_scale equals 0.1. This is the H076
language recipe; no return to H075's synthetic parameter-decay narrow setting.
Execution remains H075/H076's fixed block checkpointing for all except narrow
GELU's block-plus-inner mode. No compiler or custom activation is introduced.

Only steps, log interval and seed differ from each selected H076 recipe:
800 updates, log_every 200 and the specified seed. The shared 10% warmup grows
from 20 to 80 updates; cosine decay stretches to 800 and ends at 0.1 peak.
These are fresh longer schedules with different early update rates, not direct
continuations. Initial validation and first pre-update training loss for seed
17 must exactly match its H076 selected run. Do not compare its later trajectory
as if the first 200 updates were identical.

FFN/total parameters are 9,437,184/15,735,168 for each full control and
2,801,664/9,099,648 for each compressed form: 70.3125% FFN and 42.1700% total
reduction. Matrix forward FFN work across eight layers is twice FFN weights per
token. The shared approximate FLOP account excludes nonlinearities, shuffles,
norms, loss, optimizer and other non-matrix costs.

## Data, scoring and resources

Keep the hash-pinned data/wikitext2_v1 cache and train-only 4096-token BPE:
Salesforce/wikitext revision f776294184f13b8ff2337b3841cf9269a6216d1e,
3,083,650 train and 322,802 validation tokens. CUDA sampler seed is 10000 plus
model seed. Before training, a read-only qualification constructs all 800
actual sampled batches for each seed, saves first-batch and final RNG hashes,
and verifies the seed-17 state after 200 against H076. No forward, backward or
optimizer update occurs in this qualification. The 4,915,200 constructed target
elements are unscored sampler checks, not training presentations.

Verify exact full validation order: 2,521 windows of 128 targets, 322,688 targets
in 158 batches, last batch nine windows; 113 suffix tokens are outside complete
windows. Shared loss evaluations occur at initialization and steps 1/200/400/600/
800 plus repeated final validation: seven passes per run, 40,658,688 validation
target exposures across all 18 runs. Select no intermediate checkpoint. Official
test data is neither fetched nor scored. This heavily reused development split
cannot supply an unbiased holdout or broader-corpus claim.

Preserve every training loss, preclip norm and rate using H076's unchanged scalar
observers; verify agreement with shared logged history. Record per-layer gradient/
activation diagnostics, finite weights and moments, clipping, parameter/state
bytes, allocated peak, throughput and full-sequence inference. Mean timed update
uses all 790 post-warmup-for-timing updates; it includes sampling, optimizer and
observer overhead, but excludes validation and checkpoint writes. Timing warmup
of ten updates differs from optimizer warmup of 80 updates.

Peak allocation includes GPU token cache, training and intervening/final
validation, as in H076. Compare fresh same-harness controls. Inference is B16/
T128 full sequence without KV cache, ten warmups and three repeats of 30 forwards;
it is not autoregressive serving. Order changes reduce one fixed cohort ordering
but do not justify controlled hardware-speed or gradient guarantees.

## Frozen decisions and interpretation

Require all 18 runs and five isolated integration checks to complete, with finite
histories, diagnostics, final weights and optimizer moments, preserved sources/
data and exact initial/sampler qualification. For EACH seed require matched GELU:

1. Has at least 70% fewer FFN weights than both full controls.
2. Final validation NLL is at most 1.01 times EACH same-seed full-control NLL.
3. Final validation NLL is strictly below EACH same-seed calibrated narrow NLL.
4. Allocated peak is at most 1.10 times EACH same-seed full-control peak.

Primary replication passes only if every seed passes every requirement. A passing
mean cannot rescue a failed seed. Report plain comparisons and the separate
0.2% narrow margin as diagnostics, not new gates. Report all seeds, means/sample
SD, paired candidate-minus-control NLL differences and exploratory 95% Student-t
intervals (n=3, df=2). Use the closed-form df=2 critical value
sqrt(2 * 0.95^2 / (1 - 0.95^2)). These small-sample intervals depend on the usual
independent approximately normal paired-difference assumptions and are descriptive;
they do not establish statistical significance or universal superiority.

Keep an operational late-plateau diagnostic for every form/seed: absolute final
200-step NLL change <=0.2%, and final NLL <=1.002 times the best recorded endpoint.
Report every outcome, with all-form plateau qualification requiring all 18.
This diagnostic does not prove convergence and does not replace the primary
quality gates. Longer duration, convergence, scaling, broader data, strong
published controls and novelty remain required even if this replication passes.
A primary failure closes this fixed replicated recipe without automatic tuning.
A pass earns only a separately frozen duration/convergence study.

## Verification and preservation

Pin the H076 final/result/source archive, its 106 sources, this plan and three
new isolated source files. Preserve the prior 65 frozen plans and all historical
evidence. Five harness checks cover exact recipe differences/frozen rates,
full-size initial states/counts/optimizer calibration at all seeds, read-only
sampler/validation qualification and schedule, every-seed gate boundaries with
no mean rescue, and statistical/plateau diagnostics. They perform zero optimizer
updates. The previously passing 108 active tests and H076's observer-fidelity
checks need no rerun when their source is unchanged.

The coordinator records process identity, UTC, output, return code and source
hashes, stopping on runtime or nonfinite failure. Retain partial artifacts and
confirm OS process state before diagnosing any interruption; never restart from
an observation timeout or overwrite a completed trial.

Independent analysis reloads every final checkpoint, checks moments at step 800,
all step logs/rates, raw parameter counts, initialization/calibration, data/source
archives, sampler states and all selections/gates. Recompute every final BF16
validation NLL once with the forward evaluator: 5,808,384 additional validation
targets and zero updates. This verifies stored development scores, not new data.

Update the readable report, current state and candidate ledger. Preserve all
171 original language/profile runs, 21 H076 language trials, 798 prior fitting
checkpoints, H074/H075 resources and failures. No active model addition,
publication or completed-research claim follows from this round alone.

References: [H076 results](ungated_lm_screen_results.md),
[H076 plan](ungated_lm_screen_plan.md),
[earlier three-seed decision rules](long_duration_replication_plan.md).
