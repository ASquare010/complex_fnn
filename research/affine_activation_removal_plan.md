# H042: frozen removal of the learned correction after training

Written after H041 passes all three seeds, before any 800-step correction reset.
At200steps, resetting the selected affine correction cost only0.001251% NLL.
At800steps the learned slopes are larger and this earlier result may not hold.
Hypothesis: the extra training parameters improve the learned projection weights,
while the final correction can be removed with little quality cost. A positive
result would permit the existing plain BlockShuffle inference implementation.
It would not prove that the correction caused the training gain.

## Fixed diagnostic

Use all three original native affine WikiText800-step checkpoints (seeds17,29,43)
from H039/H041. No retraining or parameter updates. Fresh GPU worker per seed,
900s deadline, sequential execution. Verify original checkpoint hash and full
322,688-target validation NLL within1e-7 of its recorded value. Use the same
pinned data and BF16 evaluation. Official test remains unfetched/unscored.

Copy every non-curve parameter for comparison. Zero all gain/bias theta vectors
in memory, keeping every other weight bitwise unchanged. Evaluate the full
validation set. Then create a plain `blockshuffle_swiglu` Transformer, copy the
common learned weights exactly and drop the zero curve parameters. Verify
identical sampled logits and full-validation NLL between the reset and plain
models. Original checkpoint files must remain unchanged. Save a new deployment
checkpoint outside results/runs, with source hash and source_step800, no
optimizer state and no claim of training the plain recipe from initialization.

The public removal helper refuses nonzero theta parameters; reset is an explicit
intervention. Do not repeat width initialization or transform projection factors.
The resulting plain model has2,801,664 FFN weights and9,099,648 total, exactly
the unmodified BlockShuffle parameter count. This is post-training removal,
not a claim of lower training memory or a replacement for gain/bias retraining
ablations. Learned shapes, source checkpoints and negative results stay retained.

## Acceptance and follow-up

For EACH seed, require reset NLL <=1.001*original learned NLL (at most0.1% cost),
>=70% FFN parameter reduction, <=1% relative NLL increase against BOTH full
controls, beat calibrated narrow and >=0.2% improvement over that seed's
unmodified BlockShuffle. Require finite logits/loss, exact common weights,
reset/plain sampled-logit equality and full-NLL equality. Preserve every failure.

All three must pass before a separate fused-inference audit of these deployment
copies is earned. This diagnostic measures final-checkpoint dependence on the
correction, not training causality, population generalization, latency or
full-network gradient stability. TinyStories transfer and gain-only/bias-only
training ablations remain outstanding regardless of this outcome.
