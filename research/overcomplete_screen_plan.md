# H056 - Overcomplete headwise WikiText quality screen

Status: frozen before scoring. H055 earned this screen by passing complete-model
memory with existing native SwiGLU gate recomputation. Its fixed-batch losses
were not held-out language measurements.

Question: does the explicitly more expressive rectangular construction translate
into competitive held-out NLL at the same 70.3125% FFN-weight reduction?

## Fixed experiment

Three new seed-17 trials, peak LR 0.0003 / 0.0006 / 0.0012 in that order.
Each trains 200 steps, batch 16, context 128: 409,600 sampled targets per trial,
1,228,800 total. Use the unchanged trainer's 20-step warmup and cosine decay to
0.1 times peak. Model: d=384, eight layers, six attention heads, vocabulary 4096;
overcomplete_headwise_swiglu, expanded 512, eight private heads of width 64,
private hidden 200, mixer groups 16. FFN weights 2,801,664; total 9,099,648.
Native BF16 on the RTX 4070 Laptop; four CPU threads; one GPU cell at a time.

Use existing ffn_lr_mode=fan_in, ffn_decay_mode=parameter, weight decay 0.1;
recompute_gate=true, gate_recompute_method=native, activation_backend=eager.
The two input-mixer factors have LR scales 8/8; output factors 8/(32/3).
Private gate/value/down tensors remain at scale 1, as do non-FFN parameters.
This policy differs from H055's uniform-rate execution qualification. It follows
the retained structured baseline policy, not a claim of optimal private-head
optimizer geometry. No new optimizer, activation, router or parameter is added.
Actual groups, finite updates and training memory are checked in the new trials.

Use the pinned data/wikitext2_v1 train-only BPE cache and seed-10017 CUDA sampler.
Evaluate all 322,688 validation targets at initialization and steps 1/50/100/150/200
(the unchanged trainer repeats final evaluation for metrics). The final partial
batch contains nine windows. No official test targets are fetched or scored.
Validation is used for selection: this is a development screen, not an unbiased
test estimate. Preserve data, source, checkpoint and final sampler hashes.

## Retained comparisons and source compatibility

Reuse the twelve H035 full SwiGLU/full GELU/calibrated narrow/plain BlockShuffle
trials at exactly these three rates, plus the three H047 router-free headwise
trials for context. Verify their retained artifact hashes, source archives,
configuration, actual optimizer groups and sampling state. Reconstruct current
optimizer groups: changing the dense controls' uniform flag to fan_in must have
no effect on actual groups; preserve their width calibration and decay policies.
The narrow down projection retains its existing width LR correction. BlockShuffle
already uses fan_in/native recomputation. Private head tensors get the ordinary
base rate in both headwise recipes. These are matched-budget recipe comparisons,
not an isolated mixing/expansion ablation: head geometry and initialization differ.

Check current computational sources against the passing H055 test snapshot;
check the old training-step AST, evaluation functions and shared forward/loss
against retained source archives. Existing H055 32-model exact-signature evidence
is retained, and no existing computation is changed in this round. Verify the
exact validation stream and shared non-FFN initial weights before dispatch.

The minimal frozen_train_worker receives only hashed protocol configurations,
source qualification and output paths. Its parent launcher is standard-library
only and records each process before importing Torch. Never overwrite a cell or
retry automatically. A failed or incomplete cell remains visible and blocks
promotion until its cause and any separately authorized continuation are recorded.

## Selection and decision, fixed before scoring

Select each recipe's finite completed trial by lowest final validation NLL;
exact ties select the lower peak LR. Every allocated candidate cell must complete
and have finite diagnostics before a positive decision. The selected candidate
must have >=70% fewer FFN weights, <=1% relative NLL cost against BOTH selected
full controls, strictly lower NLL than selected calibrated narrow, and allocated
training peak <=1.10 times BOTH full controls. These are the original engineering
gates, not statistical significance. Report the separate 0.2% narrow margin and
comparisons to plain BlockShuffle and square headwise as diagnostics, without
silently adding or removing primary gates after seeing results. Report same-rate
comparisons, all three trajectories, clipping, parameters, FLOPs and measured
throughput/memory. Cross-session timing is descriptive, not a speedup claim.

A pass earns a separately frozen longer comparison; it establishes neither
convergence nor seed robustness, scale transfer, easy optimization or novelty.
A boundary-rate winner is not an optimal rate. A failure rejects promotion of
this recipe at this budget; do not automatically repair it with an activation,
extra optimizer search or a longer run. Inspect recorded diagnostics to decide
whether a distinct testable mechanism remains. Retain H051's longer-training
failure, all activation failures and H054's eager-memory failure.
