# H061: additive block/low-rank WikiText screen

Frozen before candidate language-quality scoring. H060 qualifies the fixed
mechanism, numerical behavior and synthetic full-model allocation. Its synthetic
losses are not evidence of language quality.

## Allocation and fixed recipe

Run exactly three new seed17 cells, peak LR0.0003,0.0006,0.0012 in that order.
Each uses200 updates, batch16, context128:409,600 training targets,1,228,800 total.
Model d384,L8,attention heads6,V4096,h1024,G8,r48. FFN2,801,664 and total9,099,648
weights. All three use H060's nonzero orthogonal50/50 branch initialization and
perturbation LR multipliers, parameter-level decay0.1, AdamW betas(0.9,0.95),
epsilon1e-8, preclip norm limit1. Native BF16, four CPU threads, one GPU trial
at a time. Native gate recomputation; no activation modification or compilation.
No branch-gain, rank, support, initialization, optimizer or width tuning.

Use the shared trainer's20-update warmup/cosine schedule ending at0.1*peak.
Use the pinned data/wikitext2_v1 train-only4096-token BPE cache and CUDA sampler
seed10017. Evaluate the same322,688 validation targets at initialization and
steps1/50/100/150/200; retain the shared trainer's repeated final evaluation.
The158th validation batch has9 windows. The official test split is not fetched
or scored. This is a validation-selected development screen, not unbiased test
performance. Record all allocated trials, trajectories and failures.

## Retained controls and compatibility

Reuse the twelve full SwiGLU, full GELU, calibrated narrow and plain BlockShuffle
200-step controls at these same three rates, hash-pinned in H056's preflight.
H059 archived discarded active implementations; it did not remove these results.
Keep each control's recorded optimizer recipe, including narrow width calibration
and product-decay correction and BlockShuffle's fan-in/native recomputation.
This compares explicitly calibrated recipes, not an isolated ablation.

Before dispatch, verify every control's metrics/checkpoint/source archive hashes,
source contents, configuration, tokenizer/data hashes, environment, target counts,
finite checkpoint weights/moments and final sampler state. Compare the retained
training-update AST, shared attention/norm/forward/loss and evaluation functions
to the current code. Data and dense equations must remain unchanged; the reusable
grouped operator has moved without changing its class/function behavior.

Execute an isolated reference probe from each distinct retained source snapshot,
using the current UV environment and identical full model dimensions. For each
control compare all initial weights, optimizer groups, FP32 predictions/loss,
every gradient, one clipped AdamW update and layer diagnostics on a fixed2x16
training-prefix batch. This is a compatibility probe, not validation ranking.
Compare common non-FFN initial tensors to the additive candidate too. H060's
six-variant CPU/BF16 signatures and synthetic qualification remain required.

Verify exact validation target order and the live200-step CUDA sampler state.
Freeze current source/configurations and use the minimal frozen_train_worker.
One durable parent records process start, output, exit and unchanged source per
phase. Never overwrite a trial or automatically retry. A process failure or
incomplete allocation prevents a positive decision until explicitly resolved.

## Selection and gates

For each recipe, select the completed finite trial with minimum final validation
NLL; exact ties choose the lower rate. All three new cells must complete with
finite history, diagnostics, checkpoint weights and optimizer moments.
The selected additive candidate must satisfy every original gate:

- at least70% fewer FFN weights;
- at most1% relative NLL cost versus BOTH selected full controls;
- strictly lower NLL than selected calibrated narrow;
- actual allocated training peak at most110% of BOTH selected full controls.

Report the separate0.2% narrow margin and same-rate/selected comparisons to all
four controls as diagnostics. Do not change gates based on which rate wins.
Report all trajectories, gradients/clipping, resource use and matrix work.
Cross-session timings remain descriptive; the H060 short synthetic timing
already shows no dense training-speed win. Memory savings alone cannot promote
quality. Boundary winners do not establish an optimal learning rate.

A pass earns a separately frozen longer comparison, followed by independent seeds,
scaling and broader-data evidence. A failure rejects this recipe at this budget;
no automatic extra-rate, activation, normalization or longer-training repair.
Keep H051's longer BlockShuffle failure and every previous negative result.
No scoped construction, short screen or known-family combination establishes a
breakthrough, general learnability or architectural novelty.