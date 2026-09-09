# Current research state

**The gold target remains unmet. H078 fails its single-seed duration gates.**
The latest completed longer-budget language experiment is matched-budget BlockShuffle GELU h3264, with
2,801,664 FFN / 9,099,648 total weights: 70.3125% FFN and 42.1700% total reduction.
This fixed duration recipe is closed. Further training needs a distinct, justified hypothesis.

## Current direction and retirement (September10)

H087 archives native rational BlockShuffle after repeated memory failures.
Dense full/narrow controls and plain BlockShuffle remain active comparison tools.
[Retirement plan](release_cleanup_plan.md) and [artifact policy](ARTIFACTS.md)
preserve the research history without committing large local tensors.

H086 passes16 native sparse-operator checks, but the first accelerated inference
case stops with CUTLASS unsupported. It has no earned training allocation.
[Complete partial-result report](compact_sparse_operator_results.md).

The next mechanism under investigation couples feature pairs with a bounded,
input-dependent rational rotation. It is a hypothesis, not a trained winner.
See [direction, equations and prior work](research_direction_2026_09_10.md).
The historical source-preservation and108-test statements below describe their
original rounds; H087 intentionally changes the active tree and records a source
snapshot for reproducing those older rounds.

## Latest controlled offset fitting

H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by -13.149%
versus plain; offset with the fixed first-factor LR correction changes it by
-14.529%. Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.
The broad research goal remains unmet.

All240 selection and reporting scores and48 resets reproduce exactly. Both
selected-recipe and matched-rate conditions are evaluated against fresh full,
narrow, gain and optimizer controls. [Completed study](latent_offset_fit_results.md).

## Latest mechanism result

H084 separates learned internal gain and offset without new training. Removing
gain multiplies aggregate checkpoint MSE by 2.972; removing offset multiplies it
by 1.843. Gain alone does not add represented functions, because it can be
absorbed into the first factor. An offset can give an isolated SwiGLU a nonzero
linear response at zero. These are local algebra and checkpoint results.

All 24 independent new reset scores match exactly. Folding the affine operations
preserves FP32 held-out error within a relative 1.7e-8, with explicit bias-buffer
storage. No BF16, speed, training-causality or full-model benefit is established.
No candidate is promoted. [Completed analysis](latent_affine_mechanism_results.md).

## Latest completed nonlinear fitting study

Both internal curves improve synthetic fitting over plain BlockShuffle, but
neither qualifies: tanh changes aggregate held-out MSE by -5.757%
and sine by -7.069%, while the equal-count affine control changes
it by -15.034%. Both curves lose to affine and narrow GELU.
The positive-control assay passes. These fixed recipes are closed; no language
or full-model resource run is earned.

H083 completes all 192 cells: four permuted nonlinear tasks, two rates, three
seeds, 600 updates and batch 256. The curves add 48 weights per FFN. All eight
recovery checks pass; independent analysis exactly reproduces 192 selection
scores, 96 selected held-out scores and 36 resets. Resetting learned curves
worsens error, but does not explain the training advantage over plain.
No active model is added. [Completed result](latent_activation_recovery_results.md),
[original adapter failure](latent_activation_results.md),
[scalar proofs and limits](latent_activation_theory.md).

## Latest completed short screen

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

## Latest operator qualification

[H080](blast_operator_recovery_results.md) qualifies a conventional BLAST comparator:
GELU h3200 and SwiGLU h1984 each have 2,801,664 FFN / 9,099,648 total weights by
count. All 34 checks pass; independent analysis verifies 32 tensor files,
114 exact checkpoint pairs, analytic derivatives, four exact same-width embeddings
and four matrix witnesses. H080 itself contains no learning or full-model resource
measurements; H081 is the first language allocation.

[H079](blast_operator_results.md) remains incomplete because of damaged artifacts.
One explicit recovery preserves numerical tests and adds durable readback; all
24 readable original tensor payloads match. A separate scalar-norm audit mismatch
is corrected on the original devices without changing thresholds. There are zero
optimizer updates, corpus targets or Transformer runs. Only a separately frozen
learning comparison is earned; no active model is added.

For data, batches, duration and learned-activation results, see the
[plain-language progress overview](PROGRESS_OVERVIEW.md).

## Latest complete longer-budget language comparison

The [H078 report](ungated_duration_results.md) contains all six fresh seed-17
trials at 3,200 steps. Models retain H077's fixed recipes, with rates originally
selected under H076's equal three-rate short search. Only steps and logging change;
warmup grows to 320. These are fresh schedules, not checkpoint continuations.

| Form | Fixed peak LR | Final NLL | Peak MiB | Mean update ms |
|---|---:|---:|---:|---:|
| Full SwiGLU | 0.0012 | 4.105417 | 384.558 | 92.600 |
| Full GELU | 0.0012 | 4.127881 | 393.245 | 85.931 |
| Narrow SwiGLU | 0.0012 | 4.153048 | 279.454 | 90.440 |
| Narrow GELU | 0.0006 | 4.205838 | 279.017 | 87.550 |
| BlockShuffle SwiGLU | 0.0006 | 4.144306 | 279.767 | 121.954 |
| BlockShuffle GELU h3264 | 0.0012 | 4.242014 | 280.017 | 104.317 |

Matched's relative NLL differences are +2.7649% versus full GELU,
+3.3272% versus full SwiGLU,
+0.8601% versus narrow GELU and
+2.1422% versus narrow SwiGLU. Negative favors matched.
It exceeds both full-control NLL allowances and fails to beat either narrow control. Each full-control allowance is 1%; BOTH narrows must be beaten strictly.
The separate 0.2% narrow margins remain descriptive.

Versus plain, candidate NLL changes by +2.3576% and mean update time
by -14.46%. These are one-seed, sequential laptop-GPU measurements.
No confidence interval or replicated long-duration conclusion follows.
Late-plateau diagnostics pass 0/6 models; the report gives each final-800
loss change. Plateau screening does not establish optimal convergence.

Peak allocation includes the CUDA corpus cache and validation. Time averages
3,190 updates after ten timing-warmup updates; optimizer warmup is separately
320. Full-sequence inference is not autoregressive serving. Heavily reused
WikiText-2 development validation is not an independent holdout.

The [H077 three-seed comparison](ungated_lm_replication_results.md) remains
shorter-budget evidence: all 18 fresh 800-step trials pass each seed's primary
gates for matched GELU, mean NLL 4.834511 +/- 0.013382 (sample SD), but it beats
plain only in 1/3 seeds with a 0.0959% mean gain. None of its 18 plateau screens pass.

## Retained repository scope

After H087 retirement, the active tree has two model folders, five registered variants and eight recipes.
Experimental GELU uses the existing ungated operator with an isolated adapter;
it is not a duplicated model folder or new registered variant.

| Component | Role and limitation |
|---|---|
| [dense_ffn](../src/dense_ffn/README.md) | Four full/narrow GELU/SwiGLU controls for fair quality and resource comparisons |
| [blockshuffle_ffn](../src/blockshuffle_ffn/README.md) | Structured operator and plain SwiGLU reference; supports the isolated GELU experiment |

All 171 original language/profile runs remain, with 21 H076, 18 H077 and six H078 trials
isolated beside their protocols. Discarded architectures and obsolete drivers
are in the [archive](archive/README.md). Historical current-state pages are saved
in the per-round before-documents archives; detailed decisions stay in the
[complete ledger](idea_bank.md) and linked reports.

## Findings that still constrain the next step

| Evidence | Decision that remains in force |
|---|---|
| [H051 longer plain replication](long_duration_replication_results.md) | 3,200-step mean NLL 4.147443 is 1.256% above full SwiGLU; longer-training goal fails |
| [Affine matched-rate correction](affine_rate_replication_results.md) | Only 0.101% mean benefit, wins in 2/3 seeds; earlier larger claim was rate-confounded |
| [Native rational activation](learnable_activation_results.md) | 200-step NLL 5.898003, but 883.17 MiB peak; learned-shape reset barely changes quality |
| [H063 native](native_recompute_results.md) / [H066 staged](staged_resource_results.md) recomputation | Numerical fidelity passes, rational memory gates fail; no automatic partition or language repeat |
| [H061 additive low-rank screen](additive_block_lowrank_screen_results.md) | Quality gates fail; implementation retired in [H062](additive_retirement_results.md) |
| [H068 conditioned activation fitting](token_activation_fit_results.md) | Static/dynamic gains miss the frozen 2% gate; richer routing receives no automatic allocation |
| [H069 factor balance](factor_balance_results.md) | Exact local preservation but insufficient measured imbalance; no optimizer experiment earned |
| [H071 rotated-shuffle fitting](rotated_shuffle_fit_results.md) | Both tested rotations fail learning gates despite a valid local rank witness |
| [H072 local GELU analysis](ungated_blockshuffle_results.md) | Linear odd-component restriction and scoped population error floor; no whole-network or learning guarantee |
| [H073 fitting](ungated_fit_results.md), [H075 resources](ungated_resource_recovery_results.md) | Qualified the language screen; synthetic gains and timings are not corpus or convergence proof |
| [H076 language screen](ungated_lm_screen_results.md) | Same-width GELU fails; matched passes with only 0.1167% gain over selected narrow GELU at 200 steps |

Bezier/quadratic, single-factor grouped, coupled/shared, paired-feature,
parallel/headwise/overcomplete, shifted/affine and failed rational compiler branches
remain archived. Local proofs survive within their stated assumptions; failures
do not justify automatic tuning or relabeling known components as novel.

## What remains before the research goal can be accepted

1. This fixed duration recipe is closed. Further training needs a distinct, justified hypothesis. Keep both full and both calibrated narrow controls, equal tuning
   effort, data policy and actual memory/runtime accounting. Duration-specific
   rate sensitivity and convergence remain unestablished.
2. Establish scale and broader-data results with multiple seeds. Reused development
   validation and fixed short-selected rates cannot prove those requirements.
3. Reproduce strong published structured comparators under the same task scope.
   The [primary-source comparator note](ungated_comparator_note.md) identifies
   BLAST and BTT-MoE distinctions and the original count-compatible BLAST budget.
4. Further learnable activation work needs a distinct justified mechanism, correct
   derivatives/initialization, practical memory and measured gain over the strongest
   relevant control. See the [activation guide](learnable_activation_domain.md).
5. Local representation proofs, measured learning, training speed and full-sequence
   inference remain separate claims. No arbitrary-data or global-gradient guarantee,
   novelty priority or automatic publication is established.

## Verification

Five H078 integration checks pass with zero optimizer updates/model forwards.
All six initial states reconstruct exactly, initial NLL/first training loss match
H077, sampler states and full validation order agree, and every final checkpoint's
BF16 NLL rescores exactly. Independent checks cover finite weights/moments, all
3,200 step records per run, counts, calibration and primary/descriptive decisions.

The [final audit](../results/verification/ungated_duration_final_v1.json) preserves
112 scientific sources, 67 frozen plans, 798 earlier fitting cells, 21 H074/H075
resource checkpoint pairs and all historical language evidence. H078 uses
19,200 updates / 39,321,600 training targets; independent rescoring adds
1,936,128 validation targets and zero updates. All scientific trials finish on
the first attempt. The previously passing 108-test active suite and its source
are unchanged; isolated integration checks do not inflate that count.

The [H080 final audit](../results/verification/blast_operator_recovery_final_v1.json)
checks 118 scientific sources, 69 frozen plans and the unchanged original H079
damaged/intact files alongside earlier language, fitting and resource evidence.
Its 34 isolated checks do not replace the unchanged 108-test active suite.
