# H054: Complete-Transformer overcomplete integration and memory qualification

Freeze after H053 passes its standalone constructive/initialization/BF16 checks.
Its isolated eager FFN peak is 20.37% above full SwiGLU. A complete-model test
is required before any quality screen; no recomputation, router, activation or
optimizer adjustment will be added to this integration round.

## Integration contract

Register overcomplete_headwise_swiglu using the existing ModelConfig fields:
groups means H FFN heads, mixer groups are 2*H, expanded width m=4*d/3 and hidden
means private width k. Require d divisible by 3 and both d/m divisible by 2*H.
The explicit experiment remains d=384,m=512,H=8,G=16,k=200,L=8,attention heads=6,
context=128,vocabulary=4096: 2,801,664 FFN weights and 9,099,648 total weights.

If hidden is zero, choose the largest positive multiple of eight fitting the
30%-of-full-FFN weight ceiling, with a minimum of eight for tiny diagnostic
models. Explicit hidden values take precedence. This is deterministic budget
accounting, not a tuning sweep; the frozen experiment specifies hidden=200.
Report matrix FLOPs as twice the projection parameter count per layer; nonlinear
and optimizer costs remain excluded. No new serialized ModelConfig fields.

Keep all existing initialized weights, logits, losses, all parameter gradients,
counts and FLOPs exact on a saved CPU signature for all 31 registered variants.
Capture before editing shared code. Add only the factory/count/validation/FLOP
paths for this variant; adjust the generic test fixture's new case to valid
rectangular dimensions. Existing test cases stay identical.

The standalone module's numerical behavior and seed-local initialization stay
unchanged. Ensure the complete decoder initializes it AFTER generic nested
BlockShuffle factors, so those factors cannot overwrite its isometric gains.
Verify every layer matches a separately initialized standalone module with the
same name/seed/residual scale, and all non-FFN parameters match full SwiGLU.
The standalone class docstring may be updated to reflect registration.

## Frozen GPU qualification

Fresh processes, order full SwiGLU, candidate, full GELU. Seed17, native BF16,
B=16,T=128,20 constant-LR .0012 AdamW updates, betas .9/.95, eps1e-8, decay .1,
clip1, uniform FFN LR and ordinary parameter decay. No gate recomputation.
Use the first CUDA batch from the existing WikiText training sampler seed10017
and repeat it for all20 updates. Each cell must match the known first-batch hash
258e6c6132c5ac9d202dd965ee64a530055aa414fd62b396b79c4d1be392362e.
Each has 2,048 unique training targets and 40,960 token exposures; all three
sum to 122,880 exposures. No validation or test scoring and no quality selection.

Check whole-model initial BF16-versus-FP32 logits (relative L2 <=.05), finite
initial/final layer diagnostics, finite gradients/weights on all20 updates,
every parameter's nonzero gradient on step1, actual optimizer groups, source,
checkpoint and data/sampler hashes. Save the trained qualification checkpoint
and restore it without recalibration to verify exact fixed-batch logits.

Measure allocated training peak across forward, loss, backward and AdamW after
resetting the allocator peak following initial precision/diagnostic checks.
Keep parameter, gradient and optimizer allocations included and save baseline
allocation. Diagnostics outside that interval do not count as training peak.
Do not claim timing superiority from this short cross-process qualification.

Require candidate allocated peak <=1.1 times EACH freshly measured full control
and >=70% fewer FFN weights, plus every integrity/numerical check. Passing earns
only a separately frozen balanced quality screen. Actual long-trial memory
will still need verification. If memory fails, stop before LM screening and
separately justify any execution change; do not modify this frozen cell.

CPU preflight ceiling 180seconds; each GPU cell ceiling 300seconds. One GPU cell
at a time. Full tests run after GPU work ends. Retain every outcome and source
archive. Numerical failures fail the qualification; unexplained process failures
stop for diagnosis and explicit continuation, with no automatic retries and no
overwrite. Existing 165 LM/profile runs, H052 negative and H053 proof remain.
