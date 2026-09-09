# H076 - Equal-budget ungated GELU WikiText language screen

Freeze before language scoring or implementation. H075 completes and independently
verifies the resource comparison earned by H073 fitting. Both conventional GELU
forms qualify for this screen; neither has language-quality evidence. The previous
turn made PROGRESS. Keep the active three-folder/six-variant/nine-recipe tree and
all prior results unchanged. New runs remain in the isolated experiment root.

## Allocation and controlled recipes

Run seven forms, each at peak rates0.0003,0.0006,0.0012 and seed17 for200 updates.
Order forms: full SwiGLU h1024, full GELU h1536, narrow SwiGLU h304, narrow GELU
h456, plain BlockShuffle SwiGLU h2048, same-width BlockShuffle GELU h2048,
matched BlockShuffle GELU h3264; rates ascend within each form. All21 are fresh
runs. Each sees409,600 training targets; total8,601,600 targets and4,200 updates.
No reuse of older language scores for the selection gates and no extra rate,
width, initialization, activation or budget following the outcomes.

All use batch16/context128/d384/L8/heads6/vocab4096, groups8 for structured
projections, BF16 with FP32 parameters, TF32 disabled, four CPU threads, native
eager Torch and one bounded GPU worker at a time. Use the unchanged shared
trainer, data, attention, normalization, loss, evaluation and cosine LR schedule:
20-update warmup, ending at0.1 times peak. AdamW betas(0.9,0.95), eps1e-8,
base weight decay0.1 and global preclip limit1.

Use H075's fixed memory-selected execution per form: block for every form except
narrow GELU, which uses block_inner. Inner wraps the existing native activation/
product only. No compiler or custom activation. Initialization matches H074/H075
name-local Transformer weights and down residual scale1/4. Non-FFN tensors match
across forms; plain and same-width GELU share common up/down factor bytes.

Structured forms use existing fan-in factor LR correction and parameter decay.
Dense controls use uniform base multipliers, with narrow down initialization and
LR calibrated by reference_hidden/actual_hidden exactly once. For BOTH narrow
forms use the established product-decay correction so their enlarged down LR
receives weight_decay0.1/LR_scale. This matches the calibrated narrow language
recipe; H074/H075 used parameter decay for their synthetic resource probes.
The decay treatment is an explicit recipe difference, not a pure activation-only
ablation. Do not disadvantage narrow by multiplying its effective down decay.

Actual FFN/total counts: full9,437,184/15,735,168; narrow/plain/matched
2,801,664/9,099,648; same-width1,867,776/8,165,760. The isolated ungated config
and factory use the existing two-projection operator. Correct the operation
counter in the isolated adapter: ungated matrix work is2 times actual FFN weights
per token, not the old three-projection assumption. Existing controls retain the
shared operation accounting. No active registry or shared source edit.

## Data, validation and measurement

Use the existing hash-pinned data/wikitext2_v1 cache: Salesforce/wikitext revision
f776294184f13b8ff2337b3841cf9269a6216d1e, train-only4096-token BPE,3,083,650 train
tokens and322,802 validation tokens. The shared CUDA sampler uses seed10017.
Preserve exact batch policy and independently reproduce final200-step sampler
state. Verify full validation order:2,521 windows x128 =322,688 targets,
158 batches, last9 windows;113 suffix tokens remain outside complete windows.
Official test data is neither fetched nor scored.

Use shared evaluations at initialization and steps1/50/100/150/200, including
its repeated final evaluation: seven full-validation passes per run,47,435,136
validation-target exposures across21 runs. This is adaptive development on the
previously used validation split, not an unbiased test or new-corpus result.
Select each form's lowest final validation NLL across its three completed rates,
exact ties lower rate. No best-intermediate selection. All21 cells must finish.

The shared loop source stays unchanged. Isolated observer hooks record every
training loss/preclip norm and rate to an additional sidecar; compare sidecar
logs to shared history and clipping. They call the original loss, initialization
and clipping functions and add no parameter/gradient transformation. Record their
small logging overhead as part of measured training time for every form.
Save initial weight signatures, all final weights/moments, sampler/Torch RNG,
configuration, provenance/source archive, diagnostics, gradients, finite checks,
clipping, training throughput, allocated peak and full-sequence inference timing.

Timing excludes the first10 updates, validation and checkpoint writes; it includes
sampling, optimizer, observer overhead and occasional gradient diagnostics.
Actual allocated peak includes GPU corpus cache, training and intervening/final
validation, matching the shared trainer. It is not directly interchangeable with
H075's synthetic resource peak. Compare same-harness full controls. Inference
metrics are full sequence without KV cache, not autoregressive generation.

## Fixed promotion gates

For each GELU form require all of the following:

1. All21 cells and the integration checks complete with finite histories,
   diagnostics, weights and optimizer moments; the original data/sampling policy
   and source hashes remain unchanged.
2. At least70% fewer FFN parameters than each full control.
3. Selected final validation NLL <=1.01 times BOTH selected full-control NLLs.
4. Selected final NLL strictly below BOTH selected calibrated narrow forms.
5. Actual allocated peak <=1.10 times BOTH selected full-control peaks.

Report same-rate and selected comparisons to plain, both full and both narrow
controls. Report the separate0.2% narrow margin as a diagnostic, not a new gate.
Do not select a candidate only because a gate passes on one rate while a different
rate supplies its quality headline. Report the parameter/quality/resource tradeoff
between GELU widths, all rates, trajectories and boundary winners. One seed and
200 steps establish neither significance nor convergence. A pass earns separately
frozen longer training, then independent seeds, scale and broader-data evidence.
A failure rejects this fixed recipe at this budget without automatic repair.
H051's longer plain-SwiGLU failure remains part of the evidence.

## Verification and execution

Pin H075 final/result and103 prior sources, this plan and three new source files.
Freeze actual git/source state, software/hardware and all tokenizer/cache hashes.
Five isolated harness tests verify configuration/counts/operation accounting,
shared and exactly-once initialization, optimizer coverage/effective narrow decay,
observer and checkpoint fidelity for the changed narrow recipes, and selection/
gate failure behavior. The fidelity check uses two updates for each of two narrow
forms in native and selected modes:8 isolated CPU qualification updates on a
fixed2x16 training-prefix batch,256 target presentations, no validation selection.
These temporary states never initialize a language trial. The existing108-test
active suite is not rerun solely for isolated adapters.

A hidden durable coordinator records PID/UTC, logs, return code and source
preservation per harness/worker/analysis phase. Stop on any runtime/nonfinite
failure and retain all artifacts; no automatic retries. Inspect live processes
before recovery. Independent analysis audits all21 checkpoints, actual parameter
counts and optimizer moments, sidecar/log consistency, source and data archives,
and all choices/gates. Reconstruct and rescore all21 final checkpoints once with
the same BF16 forward evaluator:6,776,448 additional validation targets, no updates.
This verifies stored development scores, not an independent holdout.

Update the report, candidate ledger and current state, preserving171 prior LM/
profile runs,798 prior fitting checkpoints, all H074/H075 resources and failures.
Retain the repository's readable structure; no new active model or publication.
The gold research goal remains unchanged and unachieved by this screen alone.

References: [earned resource qualification](ungated_resource_recovery_results.md),
[prior language-screen rules](additive_block_lowrank_screen_plan.md),
[synthetic fitting and limits](ungated_fit_results.md).
