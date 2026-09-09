# H055: Existing native gate recomputation for overcomplete headwise FFNs

Freeze after H054 registers the module and fails the full-GELU eager memory
allowance by 4.270 MiB. Reuse the existing RecomputedSwiGLU autograd function;
change no projection, parameter, initialization, activation formula, router,
data, global LR, factor LR or decay policy. No checkpoint/compiler variant sweep.

## Implementation and preservation

Add an opt-in native gate-recomputation flag to OvercompleteHeadwiseFFN, default
false. During training with gradients enabled, save gate/value inputs and
recompute SiLU in backward; eval/no-grad and the default eager path stay the same.
Factor the existing trainer's gate setup into one helper supporting this new
native-only case and preserving all older BlockShuffle/learnable-activation rules.
Use that helper in both the real trainer and the qualification worker.

Preserve all 32 existing default model signatures (weights, logits, losses, all
gradients, parameter counts/FLOPs) in a CPU snapshot captured before edits.
FP64 direct-versus-recomputed forward/all derivatives and input gradcheck must
pass at the small standalone geometry. Unsupported setup combinations must fail.
No new registered model variant or serialized configuration field.

Extend H054's GPU worker with explicit frozen-root/plan/config-factory and
optional arithmetic-check inputs, retaining its complete memory/checkpoint
procedure and default behavior. The exact H054 source remains archived. This
avoids copying a second full pipeline loop. All new source is frozen before work.

## One matched GPU cell

One fresh candidate at the exact H054 geometry and seed17, with only
recompute_gate=true and gate_recompute_method=native. Same native BF16, batch16,
context128, 20 constant-LR .0012 AdamW updates, .1 parameter decay, uniform LR,
clip1, and same first CUDA WikiText batch repeated20 times. New exposures40,960;
no validation/test scoring. Retain H054's eager candidate and both full controls.

Before the training-peak interval, compare initialized eager and recomputed
whole-model fixed-batch loss and every parameter gradient on the same weights
and data, without optimizer updates. Initial losses match within1e-7; every
parameter-gradient relative L2 error must be finite and <=.02. Restore native
recomputation before training and do not consume another sampler batch. Preserve
these diagnostics and the setup's baseline allocated memory. They are warmup
checks, not a speed measurement.

The existing whole-model BF16/FP32 logit limit .05, finite weights/gradients/Adam
moments/layer diagnostics, nonzero first gradients, first-batch hash, checkpoint
logit round trip, actual optimizer groups, data/source/checkpoint integrity and
20-step history checks all remain. New initial fixed-batch loss must also match
H054's eager candidate within1e-7. Later updates are not required to be bitwise
identical. Neither early loss nor this fixed-batch result selects a quality model.

Require native peak strictly below its eager candidate AND <=1.1 times EACH
retained H054 full control, with the same >=70% FFN-weight reduction. Only a
pass earns a separately frozen balanced language-quality screen. Actual screen
memory still has to pass; no speed, convergence, SOTA or breakthrough claim.

CPU preflight ceiling180seconds; one GPU cell ceiling300seconds. Full regression
tests run after GPU work. Retain exact H054/H053 outcomes and all attempts. No
automatic retries, overwrite, extra rates or activation repairs. Unexplained
process failures stop for diagnosis and explicit continuation. If this fails,
do not start the quality screen under this plan.
