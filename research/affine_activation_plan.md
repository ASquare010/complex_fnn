# H037: frozen affine activation ablation

Written before affine LM training. The selected native rational BlockShuffle
checkpoint improved WikiText NLL by 1.216% over its selected unmodified base,
but exceeded memory limits. Its learned correction has 99.7684% affine energy
on the uniform [-6,6] diagnostic grid. This is not a data-weighted result.
Hypothesis: a much simpler learned slope and offset can reproduce that gain
with lower execution cost. This is a retraining ablation, not a novelty claim.

Use phi(z) = SiLU(z) + 0.25*(tanh(theta_gain)*z + tanh(theta_bias)).
Eight groups, independent parameters per layer, all theta initially zero;
16 new weights/layer, 128 across eight layers. Parameters receive the global
learning rate and no decay. No new initialization or shape hyperparameter search.
Preserve common projection weights, optimizer assignments, BF16 rounding,
calibrated narrow initialization and BlockShuffle checkpoint recomputation.
Implementation and the scoped origin-Jacobian proof are in
[model notes](../src/learnable_activation_ffn/model.md).

## Fixed experiment

Two recipes (calibrated narrow and BlockShuffle) each receive peak rates
0.0003, 0.0006, 0.0012. d384/L8/H6/context128/vocab4096; hidden304 and2048,
respectively. WikiText-2 pinned train/validation cache, B16/BF16, seed17,
200 steps =409,600 sampled training tokens/trial, six trials =2,457,600 tokens.
Full validation:322,688 targets, including final nine-window batch. Official
test remains unfetched/unscored. Ascending rates, alternating recipe order;
one fresh GPU worker at a time, 900s deadline. Retain every failure/run.

Reuse the twelve archived original control trials and six relevant rational
trials, without new tuning. Verify control source archives, data hashes,
26 old CPU variant snapshots and initial real-token BF16 logits/common gradients.
Shared training loop, projections, optimizer and data stream remain unchanged.
Equal new per-recipe tuning does not equalize historical architecture search.

Select lowest FINAL NLL per recipe, breaking ties by lower rate. Record grid
boundary selection; do not extend this grid. Promotion requires finite recorded
activations/slopes/gradients, >=70% FFN weight reduction, <=1% relative NLL
increase against BOTH selected full controls, beat selected calibrated narrow,
>=0.2% improvement over its own selected unmodified compressed base, and allocated
training peak <=1.1 times BOTH full references. Additionally require NLL <=1.002
of that base's selected native rational variant. This earns further study only;
it is not convergence, independent replication or SOTA evidence. Serial timing
is descriptive and does not justify a paired speed claim.

## Mechanism and follow-up

After selection, reuse the existing full-validation curve-reset intervention
for both affine checkpoints (zero all theta, no other weight changes/retraining).
Record learned gains/biases and curves. Separately, using the selected native
rational BlockShuffle checkpoint, sample preactivation/correction pairs from the
first eight training windows (no labels or fitting the model). Fit per-layer/group
affine functions by CPU FP64 least squares, using deterministic stride to cap
at 65,536 samples/group/layer. Report uncentered correction energy explained
and RMS fit error, explicitly scoped to this training-prefix sample. This audit
addresses the uniform-grid limitation and cannot establish training causality.

Only a promoted recipe earns independent seeds, 800-step training and TinyStories
transfer. A negative result rejects this frozen affine recipe, not every possible
learnable activation or learning rate. Compiler refinement is a separate
execution hypothesis; do not change this eager training cohort while it runs.
