# Fixed conditioning and fused execution: joint validation

Hypothesis H031. The previously selected 0.1 singular-value floor can improve
trained down-projection conditioning while preserving the larger BlockShuffle
model's quality when served through the existing four-kernel implementation.
Freeze this plan before evaluating the combination; do not tune the floor or
kernel tiles against its results.

## Cases and controls

Use width 384, eight layers, hidden width 2048, eight groups, 800-step checkpoints
for seeds 17, 29 and 43. Each seed has two fresh, isolated worker processes:
original weights and the fixed 0.1 floor. Six cases total; no training.
Only blocks.*.ffn.down.second.weight may change. Preserve aggregate Frobenius
norm through the existing CPU FP64 SVD transformation; store the resulting FP32
factors and their hashes. Original checkpoints stay unchanged. Prior random
perturbation controls are documented in conditioning_results.md; this round
checks transfer and execution compatibility, without adding a new strength search.

For each case, measure CPU FP64 materialized down-projection spectra and the
existing fixed-batch CPU FP32 loss-direction gradients before replacing FFNs.
These are projection and sampled-direction diagnostics, not a proof of full
network conditioning or nonvanishing gradients. The square-factor bound at
strength 0.1 is condition <= 10 in exact arithmetic; the rectangular product
has the bound derived in conditioning_plan.md. Its input nullspace remains.

## Quality and numerical comparison

Use the unchanged TinyStories cache, tokenizer and target positions. Score the
original prefix (32,768 targets) and already-scored disjoint tail (167,936 targets)
separately. The tail is now a diagnostic region, not a fresh holdout. Native
CUDA BF16 evaluation is the reference for each case's own weights. Retain
per-window tail losses and exact target counts. Compare original native scores
to the completed validation-tail audit, allowing absolute NLL drift <= 0.0002.

First check fused eager execution on the prefix and three distinct real input
batches. Then compile the same model with default Inductor, fullgraph=True,
dynamic=False, no eager fallback, unchanged BM32 kernel tiles. Compare fresh
input logits before and after four warmup calls to that case's native outputs
kept on CPU. Require finite logits, max absolute error <= 0.15, RMS <= 0.01,
and absolute prefix/tail NLL difference <= 0.001. No original/floored logit
comparison is used to judge kernel accuracy: a weight edit is a separate effect.

Retain the combined option only if, for every seed:
- the worst down-projection condition improves by at least 10x;
- native and compiled floored NLL each increase by at most 0.1% versus original
  native NLL, separately on prefix and tail;
- compiled floored NLL stays within 1% of both full baselines and beats calibrated
  narrow SwiGLU on both regions using the locked same-seed reference scores;
- all numerical and finite-gradient checks pass, parameter objects/counts are
  unchanged by replacement, other weights are bitwise unchanged, and original
  checkpoint hashes match before and after.

## Execution and evidence

Use UV with the compile extra, four CPU threads, CUDA BF16, TF32 disabled,
workspace-local compiler caches and one GPU worker at a time. Limit each fresh
worker to 420 seconds; retain logs and failures. Stop on infrastructure or
numerical failure, and record quality rejections without retuning. Record source
archive/hash, this plan hash, environment, input checkpoint/data hashes, changed
factor tensors/hashes, spectra, gradients and raw losses. No paired timing or
memory comparison is made in this round, so the earlier seed-17 serving result
must not be generalized to the combined option or the other seeds. The complete
research objective still needs equal tuning, convergence, broader language data,
credible novelty and appropriately scoped stability evidence.
