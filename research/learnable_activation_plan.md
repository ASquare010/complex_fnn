# Frozen learnable-activation screen: H034 and H035

Written before any new LM training. H034 tests shifted quadratic Bezier
coordinates (the earlier H004 idea); H035 tests a stable rational residual.
The two recipes and proofs are fixed in
[model notes](../src/learnable_activation_ffn/model.md). Amplitude 0.25, four
Bezier coordinates, eight activation groups, bounded centers/slopes/denominator,
zero initial corrections and global-rate/no-decay shape optimization are fixed.
No shape hyperparameter is selected using these LM losses.

## Cohort and controls

Use the pinned WikiText-2 raw train/validation cache from the
[completed transfer screen](wikitext_screen_results.md). Official test stays
unfetched and unscored. d384/L8/H6/context128/vocabulary4096; narrow hidden304,
BlockShuffle hidden2048/groups8. Evaluate all 322,688 validation targets,
including the final nine-window batch. B16, BF16, seed17, 200 steps = 409,600
sampled training tokens per trial. This is a cheap selection screen, not convergence.

Four new recipes: calibrated narrow + shifted; calibrated narrow + rational;
BlockShuffle + shifted; BlockShuffle + rational. Each receives the same global
peak-rate grid (0.0003,0.0006,0.0012), 12 new LM trials in total (4,915,200
training tokens). Ascending rates, rotate recipe order each rate, one fresh GPU
worker at a time, 900-second deadline each. Stop on infrastructure or numerical
failure; retain completed and failed artifacts. Never overwrite a run.

Reuse the twelve archived full GELU, full SwiGLU, calibrated narrow and
BlockShuffle controls from the completed three-rate cohort. Their data,
initialization and optimizer behavior are verified before reuse. This is equal
new per-recipe tuning; it does not erase historical architecture search or
provide simultaneous paired speed measurements. All 22 pre-existing variants
must reproduce the archived tiny CPU parameters, logits, loss, common gradients
and matrix-work counts exactly after factory extensions.

New narrow models retain dense width calibration, with eager activation backward.
New BlockShuffle models retain factor-specific initialization/LR and gate
recomputation, using non-reentrant checkpointing for the richer activation.
The baseline's native SiLU derivative cannot evaluate a learned curve. This
execution change is explicit: it recomputes only activation/product, preserves
all parameter derivatives, and adds overhead. No previous fused kernel is used.

## Selection and promotion

Select each recipe's lowest FINAL validation NLL; tie-break by smaller global
peak rate. Record grid-boundary winners without extending this grid. All trials
must have finite recorded gradients/activations/slopes. Any promoted recipe must:

- Retain at least 70% fewer unique FFN parameters than the full references.
- Have NLL within 1% relative of BOTH independently selected full controls.
- Beat selected calibrated narrow and improve at least 0.2% relative over its
  own selected unmodified compressed base. This threshold rejects negligible
  changes as a reason to spend a larger budget.
- Keep allocated training peak within 10% of BOTH selected full references.

These gates only earn more study; they do not establish gold acceptance. Record
inference/training throughput as descriptive serial measurements. No speed
promotion follows from cross-date timing or low parameter count. Added shape
weights are counted; their pointwise work is not mislabeled as matrix FLOPs.

## Shape and mechanism evidence

Before training: independent FP64 equations/parameter derivatives, tiny CPU/BF16
initial-output/common-gradient preservation, checkpoint equivalence, actual
weight counts, real-token backward and exact validation target accounting.
After each run: source/config/data hashes, checkpoint/history/initial+final
layer statistics; sampled activation slopes, control saturation and all shape
parameters. After selection: plot learned corrections and derivatives, quantify
shape movement, and reset learned curves in the selected checkpoint to separate
its dependence on the correction from the original unmodified training recipe.
A reset is a post-training intervention, not a retraining ablation.

A winner must next earn independent seeds, longer training, TinyStories transfer,
and simpler/fixed-coordinate controls. A failure stays negative for this frozen
setting; do not relabel a toy-fitting gain as language-model success.
