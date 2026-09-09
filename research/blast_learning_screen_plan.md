# H081 - BLAST learning screen with explicit factor calibration

Freeze before implementation or new numerical execution. The previous goal turn
made PROGRESS: H080 qualified the operator and independently verified its saved
evidence. Language learning remains untested. This stage uses that qualification
to run a bounded, equal-tuning WikiText-2 comparison. The gold goal remains unmet.

## Hypothesis and optimizer calculation

Test whether BLAST's shared block bases and rank-wise coupling give a useful
quality/resource tradeoff at the existing 2,801,664 FFN-weight budget. BLAST is
published prior art; the orthogonal initializer, outer permutation and calibration
below are local recipes, not a reproduction of the authors' full training setup.

For a projection with K multilinear factors and initial factor RMS rho_j, set
the fixed learning-rate multiplier s_j = rho_j / (K * sigma), where sigma is
0.02 for up/gate and 0.02/sqrt(2L) for down. Compute rho_j once from the actual
initial CPU FP64-converted tensor. For a unit-RMS normalized Adam direction,
the factor-relative update RMS is eta*s_j/rho_j = eta/(K*sigma). This is only a
parameter-relative scale calibration: actual directions, epsilon, clipping,
factor Jacobians and correlations need not satisfy that assumption. It neither
matches dense functional updates nor proves convergence or an optimal learning rate.

Set factor decay lambda_j = lambda/(K*s_j). In real arithmetic, with zero
gradient and zero Adam moments, every factor shrinks by (1-eta*lambda/K), so
the represented projection shrinks by (1-eta*lambda/K)^K. The leading-order
decay equals dense (1-eta*lambda). For K=2 the difference is a^2/4; for K=3
it is a^2/3-a^3/27, a=eta*lambda. Fixed permutations do not change this proof.
Actual trained trajectories are not scalar shrinkage. This first-order product
decay is explicit rather than an unreported threefold decay of the represented map.

Apply this RMS/product recipe to both BLAST (K=3) and a separate BlockShuffle
control (K=2). Retain the existing fan-in/parameter-decay BlockShuffle as another
control. Their comparison measures a complete optimizer-recipe change. BLAST
versus calibrated BlockShuffle still changes structure and hidden width; do not
claim a pure activation or a pure optimizer causal effect from that comparison.

## Fixed allocation

Eight forms, in this order, each with peak rates 0.0003, 0.0006, 0.0012 in ascending
order; seed 17; 200 updates per cell. All 24 language runs are fresh:

| Form | Hidden | FFN weights | Total weights | Factor treatment |
|---|---:|---:|---:|---|
| Full SwiGLU | 1024 | 9,437,184 | 15,735,168 | Existing dense |
| Full GELU | 1536 | 9,437,184 | 15,735,168 | Existing dense |
| Narrow SwiGLU | 304 | 2,801,664 | 9,099,648 | Existing calibrated narrow |
| Narrow GELU | 456 | 2,801,664 | 9,099,648 | Existing calibrated narrow |
| BlockShuffle SwiGLU, legacy | 2048 | 2,801,664 | 9,099,648 | Fan-in LR / parameter decay |
| BlockShuffle SwiGLU, calibrated | 2048 | 2,801,664 | 9,099,648 | RMS LR / product decay |
| BLAST SwiGLU | 1984 | 2,801,664 | 9,099,648 | RMS LR / product decay |
| BLAST GELU | 3200 | 2,801,664 | 9,099,648 | RMS LR / product decay |

Use d384/L8/heads6/context128/vocab4096/batch16, groups8/rank48 for BLAST.
Native eager CUDA BF16, FP32 parameters, TF32 off, four CPU threads. One GPU
worker at a time, through UV. Whole-block nonreentrant recomputation for every
form except narrow GELU, whose previously qualified block+inner mode is retained.
No compile, new activation, extra widths, dynamic rank or post-outcome rate repair.

The shared trainer, data, attention, normalization, evaluation, clipping and
schedule stay unchanged: AdamW betas(0.9,0.95), eps1e-8, base decay0.1, clip1,
20-update warmup then cosine to0.1 times peak. Full/narrow/legacy recipes match
H076. Both narrows retain width-adjusted initialization/LR and corrected decay.
Construct BLAST FFNs after the shared dense constructor so its generic normal
initializer cannot overwrite BLAST's qualified orthogonal factors. Use full
layer-local projection names for their seeds. Non-FFN initial weights must match
the controls exactly. The calibrated BlockShuffle must start byte-identical to
legacy BlockShuffle. No earlier fitted model initializes a new language trial.

The isolated config and operation counter must use actual BLAST factor counts:
per projection r*(m+n+b^2), two projections for GELU, three for SwiGLU. Report
2*FFN weights as matrix forward FLOPs/token; nonlinearities, rearrangement,
normalization, softmax, loss and optimizer are excluded. No speed claim from counts.

## Data and measurements

Pin the existing data/wikitext2_v1 cache and H078 tokenizer/data hashes. It has
3,083,650 train tokens, 322,802 validation tokens; train-only4096 BPE; published
Salesforce/wikitext raw-v1 revision f776294184f13b8ff2337b3841cf9269a6216d1e.
Use shared CUDA sampling seed10017:16x128 targets/update,409,600 per cell,
9,830,400 training targets and4,800 optimizer updates across24 cells.
Validation scores322,688 targets in158 batches (last9 windows). The shared
initial/1/50/100/150/200/repeated-final evaluations make54,211,584 development
validation target presentations. Official test stays unscored; this reused
development split is not an independent generalization test.

Record every training loss, preclip norm and rate; shared per-layer activation
and gradient diagnostics; sampled SiLU/GELU slopes at initialization/final;
finite weights/moments; clipping; source/config/environment; sampler/RNG; final
checkpoint; actual allocated peak; synchronized training and full-sequence
inference time. Timing excludes first10 updates, validation/checkpoint writes,
and includes observer logging and occasional shared gradient diagnostics.
The allocated peak includes cache/training/intervening validation, as in H076.
Sampling, schedule and validation policy must match across all cells.

## Preflight, stop rules and gates

Before language work, verify all eight model counts/configurations, common
initial tensors, layer-specific BLAST initialization, optimizer coverage, fixed
RMS multipliers/decay identities, unchanged legacy controls, and failure/selection
logic. Verify full-model eager versus chosen checkpoint execution for the three
new recipes (calibrated BlockShuffle, BLAST SwiGLU, BLAST GELU): two updates per
model/mode, CPU FP32 on2x16 fixed train-prefix targets and CUDA BF16 on16x128
fixed train-prefix targets. Require exact losses, gradients, optimizer/model
states and observer records for paired modes. These are24 qualification updates,
384 CPU and24,576 CUDA target presentations, separate from language allocation.
They are temporary, not candidate training or validation selection.

Stop on any runtime/nonfinite/fidelity failure; preserve logs/partial evidence.
No automatic rerun or numerical tolerance relaxation. Inspect actual process
handles before recovery. Freeze three new scientific sources and their inherited
dependencies, git state, environment, plan and prior H080 final/source/result.
Use hidden durable coordinator, PID/UTC/return records, fsynced logs and completed
artifact flush/readback. Prior damaged H079 artifacts remain unchanged.

Select the lowest final NLL per form among its three rates; exact ties lower LR.
All24 runs and preflight checks must complete and be finite. For each BLAST form
and calibrated BlockShuffle, require >=70% fewer FFN weights, <=1.01 times BOTH
selected full-control NLLs, strictly below BOTH selected narrow NLLs, and peak
<=1.10 times BOTH selected full-control peaks. Report every rate, comparison to
legacy/calibrated BlockShuffle, clipping, time, memory and boundary winners.
Passing earns separately frozen longer/multi-seed work, not an accepted model.
A failure closes this fixed recipe/budget. Do not change gates after outcomes.

Independent postprocessing must verify checkpoint/config/metric hashes, final
sampling state, actual counts/moments, histories, source/data preservation,
selection/gates, and rescore all24 checkpoints once in native BF16 (7,744,512
extra validation targets, zero updates). Preserve failures. Update the readable
report, overview/current state and candidate ledger. Active source remains three
model folders, six variants and nine recipes until stronger evidence earns a
separate integration decision. Do not rerun unrelated completed studies.

References: [operator qualification](blast_operator_recovery_results.md),
[previous screen](ungated_lm_screen_plan.md),
[BLAST paper](https://arxiv.org/html/2410.21262v1),
[AdamW paper](https://arxiv.org/abs/1711.05101).
