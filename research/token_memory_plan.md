# H101: memory first, token-local recomputation

Frozen before GPU measurements on 2026-09-10. The user's current priority is
lower measured VRAM; the older 70% FFN parameter target remains a separate,
unmet architectural objective. This study changes execution, not capacity.

## Hypothesis and scope

After whole-block checkpointing, the vocabulary projection/loss can limit peak
memory. Token-chunked FFNs alone may therefore have little full-model benefit.
Test loss chunking, FFN chunking and their combination separately. This is a
new experiment in this repository, not a claim to have invented chunking.

For a deterministic tokenwise FFN f and row partition X=[X1;...;Xk],
f(X)=[f(X1);...;f(Xk)]. For unmasked next-token targets,
L=sum_i CE_sum(Xi W^T, yi)/N. Differentiating finite sums gives the same
mathematical parameter and input gradients. No gradient term is dropped.
These identities do not imply bitwise floating-point equality, convergence,
absence of vanishing gradients, or improved representational capacity.

An FFN chunk of C tokens requires O(C h) expanded feature workspace instead of
O(N h); a loss chunk requires O(C V) logits instead of O(N V). Saved inputs,
parameters, gradients, optimizer moments, attention and allocator workspaces
remain. Recomputing checkpoints adds matrix work; concatenation retains O(N d).

## Prior art checked

- [Reformer](https://arxiv.org/abs/2001.04451), section 3, explicitly describes
  feed-forward chunking. Its reversible layers/attention are not adopted here.
- [Cut Your Losses](https://arxiv.org/abs/2411.09009) avoids materializing the full
  classifier logits using specialized kernels. Our simple native PyTorch token
  chunks still materialize C-by-V logits and use no gradient filtering.
- [PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
  provides non-reentrant recomputation. This study uses the installed API.

## Qualification before resource measurements

Check all five retained variants, uneven chunks, noncontiguous input, exact
parameter/state-key preservation, evaluation behavior, invalid settings,
double-precision loss/input/parameter gradients and numerical gradcheck.
CUDA BF16 full-model relative L2 gradient error must be <=2% per tensor
(denominator floored at 1e-8), relative scalar loss error <=0.1%; CPU double
comparisons use rtol=1e-9, atol=1e-11. Preserve failures; do not relax tolerances.

## Resource experiment

Use existing Transformer, data, initialization, optimizer groups and diagnostics.
Five variants: full GELU h1536, full SwiGLU h1024, narrow GELU h456, narrow
SwiGLU h304, BlockShuffle SwiGLU h2048/g8. Width384, L8, six heads, V4096.
Counts are checked from actual models. Two shapes: B16/T128 and B8/T512. BF16 autocast,
FP32 weights/AdamW, TF32 off, four CPU threads, sequential RTX4070 workers.

Six policies, equally available to every architecture:
native; whole FFN checkpoint; whole block checkpoint; block plus FFN chunks;
block plus loss chunks; block plus both. Fixed chunk size512, no size search.
Seeds17/29/43. Each case gets12 paired WikiText-2 training updates, constant
LR0.0006, AdamW betas(.9,.95), eps1e-8, matrix decay0.1, clip1. First four are
warmup; time eight updates with CUDA synchronization, excluding diagnostics.
This is a short execution-fidelity/resource assay, not converged language training.
Keep all losses, pre-clip norms, gradient finiteness, allocation/reservation,
parameter/gradient/optimizer bytes, forward/backward/optimizer time separately,
final common unchunked validation on eight fixed batches. Final checkpoints are
saved for native references; all endpoints receive hashes and an immediate,
independent forward score check. Other final tensors are released after checking
to bound disk use; they cannot later be independently rescored without rerunning.
No official test split. Compare each execution policy to its own architecture.

## Gates and decisions

Qualify numerical checks first. A useful memory option must reduce peak allocated
training memory >=15% versus the strongest existing checkpoint policy at each
shape, across all three seeds, while median update time increases <=25% versus
that same policy. It must preserve final validation NLL within1% for each seed
and finite training diagnostics. Report mean/median/sample variance and every
case; no aggregate may hide a failing architecture/shape. A scoped pass can be
retained as an opt-in tool, with no universal pass implied.

FFN-only, loss-only and combined ablations determine which component matters.
If chunked FFN adds cost without memory improvement over loss-only, eliminate
that fixed combination. Do not change batch, dtype or attention to obtain a win.
No activation search, new corpus, kernel or long training is allocated here.
The next architectural experiment must use the improved memory baseline if one
qualifies. A memory result alone does not satisfy the original breakthrough goal.
