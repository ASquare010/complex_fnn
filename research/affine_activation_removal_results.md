# Removing the affine activation after training

**All three checkpoints pass the frozen removal gates.** Zeroing and removing
the learned correction costs only 0.0058% to 0.0199% NLL. Every projection and
other common weight stays bitwise unchanged. The resulting models use the plain
BlockShuffle architecture, with 2,801,664 FFN weights and 9,099,648 total.

[Frozen H042 plan](affine_activation_removal_plan.md),
[raw decisions](../results/affine_removal_v1/result.json), and
[original three-seed result](affine_activation_replication_results.md).

| Seed | Learned affine NLL | Removed correction NLL | Relative cost | All gates |
|---|---:|---:|---:|---|
| 17 | 4.749981965 | 4.750811172 | +0.017457% | PASS |
| 29 | 4.757793810 | 4.758068841 | +0.005781% | PASS |
| 43 | 4.754844169 | 4.755788750 | +0.019866% | PASS |

Mean NLL after removal: **4.754890**.
Each seed remains within 1% of both full controls, beats calibrated narrow and
improves at least 0.2% over its original selected plain BlockShuffle recipe.
Every reset cost is below the predeclared 0.1% allowance.

## What was checked

Each fresh GPU worker reproduced the original full-validation NLL within 1e-7,
then evaluated all 322,688 validation targets after reset and after stripping.
Reset and stripped sampled logits are bitwise equal; their full NLLs are equal.
Source checkpoint hashes remain unchanged. Native BF16 and the frozen WikiText
data/tokenizer are preserved. No training or optimizer updates occurred.

Deployment copies are saved under `results/affine_removal_v1/s{seed}/stripped_checkpoint.pt`.
They contain model/config, source step 800, the original checkpoint hash and
explicit intervention metadata. They contain no optimizer state and must not
be described as plain models trained from initialization.

The [removal helper](archive/retired/src/core/activation_retrofit.py) refuses nonzero curve
parameters, so resetting a learned correction is an explicit intervention.
It copies common weights into a new plain Transformer without reinitializing
their scale. The two CPU tests cover both compressed bases and trained weights.

## Interpretation and next controls

The final correction contributes little to these checkpoint predictions. This
supports investigating how the training recipe changed the common weights; it
does not prove the activation caused that change. H041 selected different
global learning rates for plain and affine models. The [same-rate comparison](affine_activation_rate_control_plan.md)
fills that gap before interpreting an activation-specific training mechanism.

The three successful deployment copies earn a separate numerical and timing
audit using the existing four-kernel inference implementation. No serving speed,
lower training memory, training speed or whole-network stability result follows
from removal alone. TinyStories transfer and gain/bias retraining ablations
remain outstanding.
