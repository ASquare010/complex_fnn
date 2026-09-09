# H052: One longer-duration factor-decay control

Frozen after H051 fails the three-seed 3,200-step quality rule. Plain BlockShuffle
mean NLL is 1.256% above full SwiGLU, and seed 43 also loses to narrow. H050's
seed-17 parameter-decay run is retained at NLL 4.145726. No new architecture or
activation has earned promotion from these results.

## Question and prior negative evidence

Does the existing product-decay option materially improve the same 3,200-step
BlockShuffle recipe when only its factor decay coefficients change? This is
one mechanistic optimizer ablation, not a balanced search for optimal global
rates or a claim that the previous loss gap was caused by decay.

H013 already tested product decay at d=192, four layers, hidden 832, 800 steps
on TinyStories: NLL 3.118254 versus parameter-decay 3.115322, with no improvement.
That negative result remains. The larger WikiText duration is a different
setting, not a rerun of that failed cell. The [CPU duration diagnosis](duration_optimizer_geometry.md)
finds a large accumulated decay difference but no collapse of the inspected
trained Frobenius norms. Neither fact predicts the direction of this ablation.

## Single fixed change and resource budget

Train one fresh model at seed 17 with the exact H050 BlockShuffle dimensions,
initialization, peak LR 0.0012, 320-step warmup/cosine schedule, 3,200 steps,
logging every 800, batch 16, context 128, native BF16, native gate recomputation,
AdamW/clip-1 and frozen WikiText data. Only TrainConfig.ffn_decay_mode changes
from parameter to product. No checkpoint continuation or new tuning rate.

For a factor with LR multiplier s_j in a K=2 projection, the existing option
uses lambda_j=0.1/(K*s_j). Up/gate factors retain LR multipliers (4,4) and get
decay (0.0125,0.0125). Down factors retain (4,64/3) and get
(0.0125,0.00234375). Non-FFN parameter treatments do not change.

With zero gradients and Adam moments, the represented-map decay per step is
(1-eta*0.1/2)^2, equal to dense decay through first order; the scalar difference
is exactly (eta*0.1)^2/4. This is not a match of complete gradient updates or
actual trained norms. [AdamW](https://arxiv.org/abs/1711.05101) and the elementary
[H013 accounting](product_decay_plan.md) are established; no optimizer novelty claim.

The new trial uses **6,553,600 sampled tokens**, about five minutes on this GPU
based on retained timings, with a 2400-second ceiling. Evaluate all 322,688
validation targets at initialization and steps 1/800/1600/2400/3200. Official
test remains unscored. Keep all four seed-17 H050 references without retraining.
No inference implementation, parameter count, activation or data change occurs.

## Preflight and integrity

Require the H051 final audit and 233-test computation snapshot. Verify its failed
primary decision and the four retained seed-17 metric/checkpoint/source archives.
The previously recorded report palette exception is analysis-only. Verify all
eight data hashes and the exact validation stream including the nine-window
last batch. Reuse the already verified seed-17 3,200-call CUDA sampler state;
there is no reason to repeat the identical sampler reconstruction.

Construct actual optimizer groups before GPU training: prove every LR scale
and every non-FFN decay stays equal, every changed coefficient has the formula
above, and all 2,801,664 FFN weights are the changed treatment. Verify initial
weights are unchanged. Applying product mode to the full/narrow references must
leave their actual parameter treatments unchanged; these equivalent control
configurations need no duplicate training. This is an isolated policy ablation,
not equal new optimizer-search effort across architecture families.

Use the qualified minimal training worker, which directly calls the unchanged
trainer without importing report/plot drivers. Preserve the source/plan,
configurations, actual groups, logs and every outcome. New initial validation
NLL and first pre-update training loss must match the original seed-17 run within
1e-7. All later updates may differ. Verify history, schedule, counts/FLOPs,
finite weights/gradients/layers, complete source archive and final CUDA sampler.

A trainer-recorded nonfinite clip-norm failure is a failed numerical cell.
Unexplained native/import/process failures stop for diagnosis and an explicitly
qualified continuation; never overwrite, restart a completed cell, reuse a
partial checkpoint or silently retry. No failed cell gets a fabricated NLL.

## Decision before any extra work

The original local gates remain: >=70% fewer FFN weights, <=1% relative final
NLL cost versus BOTH full controls, strictly beating calibrated narrow, finite
verified diagnostics and allocated training peak <=1.1 times EACH full control.
Additionally require **>=0.2% lower final NLL than the retained parameter-decay
BlockShuffle** to earn a separately frozen independent-seed replication.
Both conditions must pass; tiny favorable noise does not earn more training.

Retain the H050 late-plateau diagnostic and clipping/resource/trajectory records.
It cannot prove convergence. If the new recipe fails its material gain or local
gates, do not extend product-decay tuning based on this cell; retain the negative
result and move to another separately justified direction. If it passes, freeze
seeds 29/43 with the same settings before any robust improvement claim. No
automatic rates, longer runs, new activations, SOTA claim or breakthrough follows.
