# FFN literature follow-up during the basis-readout screen

Read on 2026-10-04 after the basis-readout recipe was frozen. This note adds
context, not a change to the registered experiment or its acceptance criteria.

[StructuredFFN](https://arxiv.org/abs/2406.16450) already studies low-rank and
block-diagonal FFN parameterizations trained from scratch, including a temporary
dense training path. Grouped matrices and compact training are established ideas;
that paper does not establish the performance of this six-response readout.

[Linear recoverability of FFN blocks](https://arxiv.org/abs/2606.19379) measures
the best affine approximation of pretrained FFN input/output activations. It
reports sharply different recoverability across blocks and a limited additional
fit from a low-rank bilinear residual probe. Its methodological point is useful:
weight-spectrum energy does not establish activation-map retention, and an
undertrained linear probe can misrepresent what an affine map can explain.
Its pretrained models, scale and diagnostic differ from our from-scratch screen.
These are author-reported results, not independently verified measurements here.

[Fuzzy-logic FFNs](https://arxiv.org/abs/2606.31845) already combine bounded
memberships through intersection and set difference. Its sequence-quantifier
extension crosses the current token-local FFN scope. Logical operators therefore
cannot be presented as an undiscovered direction merely by renaming them.

Research implications, not validated findings about our models: separately
measure a trained block's linear and nonlinear input/output structure before
assuming multiplication captures its useful residual. If a later diagnostic is
run, preregister closed-form affine fitting on training activations, independent
held-out activation samples, per-output errors and sample/ridge sensitivity.
Do not select architecture hyperparameters using test activations or reinterpret
high aggregate fit as a guaranteed language-loss result. No diagnostic fitting
or model change runs concurrently with the active GPU language campaign.

Novelty remains unverified. The current experiment's concrete question is whether
its allocation of weights to a grouped six-response readout beats equally sized
ordinary dense FFNs under fixed data/backbone/compute comparisons. A new practical
result could matter even when its mathematical ingredients have precedents.
