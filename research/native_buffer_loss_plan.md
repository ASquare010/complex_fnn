# H127: native-arithmetic classifier buffer reuse

H126 is progress, but its long-run gate failed. Do not spend more seeds on that
combination. Test a different storage path that retains the native full-batch
matrix products and reduction kernels, without chunked reduction, compact
RMSNorm, or gradient staging. No parameter or novel-function claim.

For logits Z=HW^T, compute L=log_softmax(Z) into the private Z buffer. Native
NLL supplies loss and G=d(loss)/dL. Compute native log-softmax backward into
G itself, then use the ordinary full-batch products dH=GW and dW=G^T H.
Two NxV buffers suffice in backward, versus native log-probability, NLL-gradient
and logit-gradient storage. Forward avoids a separate logits/log-probability
pair. Saved inputs/log-probabilities are never mutated in backward; caller
inputs and upstream derivatives remain unchanged. Exact arithmetic is a
hypothesis to check, not assumed from algebra. Backend aliasing is qualified
only on this installed runtime/shapes; first derivatives, FP32/FP64 only.

Prior art: [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) avoids the full
logit matrix; [efficient_cross_entropy](https://github.com/mgmalek/efficient_cross_entropy)
already overwrites logits with gradients. This experiment retains the matrix
and reuses native PyTorch kernels. See [CUDA SoftMax source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu).
This is implementation research, not a new FFN/activation discovery.

Prospective budget and gates:
- Eight CUDA operator comparisons: four (tokens,vocabulary,width) shapes
  (3,7,5), (9,1024,32), (11,4096,48), (5,4097,16), each FP32/FP64;
  native and candidate each one backward = 16 backwards. Include ignored
  targets, noncontiguous inputs, scaled logits, negative upstream scaling.
- Require bitwise loss and both gradients, finite outputs, no input mutation.
  Six FP64 directional derivative checks use 12 extra forwards, tolerance 1e-7.
  Save every result. If qualification fails, stop before whole-model work.
- If qualified: reuse H123's unchanged probe loop, ten repeats/three warmup,
  two corpora and two arms, reversed corpus order = 40 model backwards.
  Both use the same H121 native FP32 step-800 checkpoints and checkpoint-input
  CPU offload, ordinary RMSNorm, resident gradients, original optimizer state.
  No updates. Native control versus buffer reuse. Save four gradients.
- Independently replay four native full-model gradients. Require bitwise loss
  and every gradient versus each saved artifact, exact states/data provenance,
  and zero allocator boundaries. Total maximum 60 backwards, 0 updates;
  163,840 model probe targets + 16,384 replay targets are diagnostics only.
- Diagnostic peak allocated memory <=0.9x native; median event and wall time
  <=1.15x native, per corpus; pinned host peak <=128 MiB. Include construction,
  all repetitions, transfers, serialization and state checks. Report all raw
  phases, mean, median and variance; these probes are not complete training.

A pass earns complete-update testing with original clipping/Adam. A failure
is retained and blocks promotion. No reruns of completed cases, threshold
changes, maintained-default changes or rescue of H126. The full VRAM,
quality and parameter-efficient FFN goal remains open.
