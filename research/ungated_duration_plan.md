# H078 - Matched GELU at 3,200 steps with fixed selected recipes

Freeze before implementation or scoring. H077 passes the full/narrow quality,
compression and memory gates on all three seeds at 800 steps. All 18 late-plateau
diagnostics fail. Matched GELU's mean gain over plain is only 0.0959%, with wins
in 1/3 seeds, although mean updates are 14.47% faster. These findings justify a
bounded duration test, not convergence or dependable plain-quality superiority.
The previous goal turn made PROGRESS. Preserve all existing evidence and the
three-folder/six-variant/nine-recipe active tree.

## Allocation

Run exactly six fresh seed-17 models for 3,200 updates: full SwiGLU h1024,
narrow SwiGLU h304, plain BlockShuffle SwiGLU h2048, matched BlockShuffle GELU
h3264, narrow GELU h456 and full GELU h1536, in that order. Every model consumes
6,553,600 sampled training targets; total 39,321,600 targets and 19,200 updates.
These are about 2.125 training-cache token exposures per model, not ordered
epochs, because windows are sampled with replacement.

Keep H077's fixed peak rates, originally selected by H076's equal three-rate
screen: 0.0012 for full SwiGLU/GELU, narrow SwiGLU and matched GELU; 0.0006 for
narrow GELU and plain. There is no duration-specific rate search or selected
intermediate checkpoint. Retain the earlier boundary-rate caveat. Use one fresh
GPU worker at a time. This one-seed stage limits compute before separately
allocating any additional long-budget seeds. No automatic extra rate, duration,
architecture, activation, optimizer change or retry follows from outcomes.

## Computation and explicit duration changes

Only steps and log_every change from each seed-17 H077 configuration:
steps=3200 and log_every=800. The unchanged shared 10% warmup grows from 80 to
320 updates, followed by cosine decay to 0.1 peak. Every trial starts from the
original initialization, not an 800-step checkpoint. Its first 800 updates do
not equal the separately trained short schedule. Initial NLL and the first
pre-update training loss must match H077 exactly; no later-trajectory equality
is asserted.

Keep d384/L8/heads6/context128/vocab4096, batch16, structured groups8, native
BF16 with FP32 parameters, TF32 disabled and four CPU threads. Reuse H076's
existing model constructor, observers and correct two-projection operation
counter, with unchanged src/core training, data, loss, attention, normalization,
evaluation and diagnostics. No compiler, new activation or registered variant.

Name-local Transformer initialization and down residual scale 1/4 stay fixed.
AdamW betas (0.9,0.95), eps 1e-8, base decay 0.1 and global clip 1 are unchanged.
Structured factors keep fan-in LR scaling and parameter decay. Both narrows
keep exactly-once fan-in down initialization/LR calibration and product decay,
so scaled down LR does not impose extra effective decay. Execution stays whole-
block checkpointing except narrow GELU's block-plus-inner mode.

Full controls have 9,437,184 FFN / 15,735,168 total weights; compressed forms
have 2,801,664 / 9,099,648: 70.3125% FFN and 42.1700% total reduction. FFN
matrix forward FLOPs equal twice FFN weights across eight layers per token;
nonlinearities, shuffles, norms, loss, optimizer and non-matrix work are excluded.

## Data and measurements

Preserve the frozen data/wikitext2_v1 cache, train-only 4096-token BPE and all
hashes: Salesforce/wikitext revision f776294184f13b8ff2337b3841cf9269a6216d1e,
3,083,650 train and 322,802 validation tokens. CUDA sampler seed is 10017.
A read-only preflight constructs 3,200 actual batches, with zero model forwards
or optimizer updates, and verifies the state after 800 against every H077
seed-17 trial. Save first-batch and final-state hashes. Its 6,553,600 constructed
target elements are unscored sampler checks, not training presentations.

Full validation is 2,521 windows of 128 targets: 322,688 targets in 158 batches,
last nine windows, with 113 suffix tokens outside complete windows. Check exact
order. Shared loss evaluations occur at initialization and steps 1/800/1600/2400/
3200 plus repeated final evaluation: seven full passes per trial, 13,552,896
validation-target exposures across six trials. Official test is neither fetched
nor scored. This heavily reused development split does not provide an unbiased
holdout, broader corpus or proof of generalization.

Save every training loss/preclip norm/rate through unchanged scalar observers,
all logged gradients and layer statistics, finite checks, final weights/moments,
RNG states, counts, bytes, source snapshots, configuration and environment.
Allocated peak includes GPU token cache, training and intervening/final validation.
Mean timed update uses 3,190 updates after the ten-update timing warmup, excludes
validation/checkpoint writes and includes sampling, optimizer and observer costs.
Optimizer warmup is 320, distinct from timing warmup. Inference is full sequence
B16/T128 without KV cache, ten warmups and three repeats of 30 forwards. It is not
autoregressive serving. Report same-harness measurements with session/order limits.

## Fixed decisions

Require all six trials and five isolated integration checks to complete with
finite histories, diagnostics, weights and optimizer moments, unchanged source/
data and exact initialization/sampler qualification. At the final 3,200-step
endpoint require matched GELU:

1. At least 70% fewer FFN weights than each full control.
2. NLL at most 1.01 times EACH full-control NLL.
3. NLL strictly below BOTH calibrated narrow controls.
4. Actual allocated peak at most 1.10 times EACH full-control peak.

A pass earns only a separately frozen long-budget replication at additional
seeds. A failure closes this fixed duration recipe without automatic tuning or
repair. No confidence interval, significance or replicated long-budget claim
comes from this single seed. Report plain comparisons and the >=0.2% narrow
margin descriptively; neither replaces the original gates.

For every recipe, report final-800 NLL change and whether its absolute value
is <=0.2%, together with final NLL <=1.002 times the best recorded endpoint.
All-form operational plateau qualification requires all six to pass. This is
a diagnostic, not optimal convergence; its outcome changes no primary gate.
Even a primary and plateau pass leaves independent long-budget seeds, rate
sensitivity, scaling, broader data, strong published controls and novelty open.
H051's old longer plain failure remains preserved, with no old score substituted
for any new gate. H077's shared-source evidence remains the initialization anchor.

## Qualification and preservation

Pin H077 final/result/source archive and 109 prior scientific sources, this plan
and three new isolated source files. Five preflight checks cover unchanged
recipes/counts/calibration/initialization; actual sampler and full validation
order; the 3,200-step schedule; primary boundary/incomplete/tie behavior; and
separate plateau/narrow-margin diagnostics. They perform zero optimizer updates.
The prior 108-test active suite and H076 observer-fidelity qualification remain
unchanged and need no unrelated rerun.

A hidden durable coordinator records PID, UTC, logs, return codes and source
hashes for each phase, runs one GPU worker at a time and stops on any runtime
or nonfinite failure. Preserve all partial work. An observation timeout is not
process termination; inspect the actual handle before recovery. No automatic
retry or overwrite is permitted.

Independent analysis audits all six checkpoints and moments at step3200, all
3,200 per-step records/schedules, exact counts and calibration, initial weights/
losses, data/source archives and regenerated CUDA sampling state. Rescore every
saved final BF16 NLL once with the forward evaluator: 1,936,128 additional
validation targets and zero updates. Recompute every primary and descriptive
decision independently. These checks validate saved development scores, not new
holdout data.

Update the report, current state and ledger. Preserve all 171 original language/
profile runs, 21 H076 and 18 H077 language trials, 798 fitting cells, prior
resource pairs/failures and 66 earlier frozen plans. No active model addition or
publication, novelty claim or completed-research claim follows from this stage.

References: [H077 result](ungated_lm_replication_results.md),
[H077 plan](ungated_lm_replication_plan.md),
[earlier duration rules](long_duration_plan.md),
[remaining comparators](ungated_comparator_note.md).
