# H116 — full-job resources after the decoder precision diagnosis

Prospective bounded screen. The preceding H115 turn made progress: its fixed
diagnosis and independent forward/stored-arithmetic audit qualified two FP32
decoder modes for resource measurement. H114 remains failed. The original
research objective remains lower measured VRAM with preserved quality and
practical runtime; architectural parameter efficiency is still unproven.

## Hypothesis and controls

Can FP32 decoder execution preserve H115's gradient fidelity while classifier
chunking reduces actual full-job allocation by at least 15%, for no more than
25% median extra update time relative to the practical BF16 reference?

Use exactly H115's six source checkpoints: four correlated narrow GELU fixtures
(WikiText-2 and TinyStories seed61, old block/loss-chunk states at step800), plus
full GELU and SwiGLU seed17 at step3200. These are not six independent seeds.
Retain each original model, batch/context, tokenizer, corpus cache and data
stream (seed +40000). No architecture, parameter count or parameter dtype changes.

Five policies, all with whole-block checkpointing:

1. `bf16_default_native`: BF16 decoder and native BF16 classifier, default SDPA.
2. `fp32_default_native`: FP32 decoder and native FP32 classifier, default SDPA.
3. `fp32_default_chunks`: FP32 decoder and checkpointed 512-token FP32 classifier chunks.
4. `fp32_math_native`: FP32 decoder and native FP32 classifier, forced math SDPA.
5. `fp32_math_chunks`: FP32 decoder and FP32 classifier chunks, forced math SDPA.

The paired native FP32 policies isolate classifier chunking. The BF16 policy
measures the complete practical tradeoff, including decoder precision. Do not
report that comparison as an FFN architecture improvement. Default/math context
applies to both training forward and backward, including recomputation. TF32
stays disabled. Do not enable deterministic algorithms or modify cuBLAS workspace
configuration. Rotate/reverse policy order by fixture before execution.

## Budget, initialization and evaluation

Thirty cases, each50 updates: **1,500 optimizer updates**. Load identical source
weights and Adam state for all five arms of each fixture. Use the H114 short
continuation recipe: constant LR0.0006, AdamW betas(0.9,0.95), eps1e-8,
existing decay groups and clip1. Record source learning rates; full controls'
old terminal LR is deliberately replaced equally. This is continuation screening,
not fresh language training or evidence of long-run quality.

An initial complete loss/backward probe adds one backward per case (30).
CPU-double qualification checks two shapes × GELU/SwiGLU × five policies against
four native references: **24 CPU backwards**. Verify masked-target handling,
loss/gradient agreement at1e-10 and unchanged parameter identities. No new custom
autograd is introduced. Main execution has **1,530 backwards**. Planned audit
adds24 FP32 initial-gradient backwards and zero optimizer updates. Total budget
including qualification and planned audit:1,578 backwards. Save all failures;
no automatic scientific retry or threshold relaxation.

All arms share the frozen memory-aware BF16/default classifier-chunk evaluator
at initial/final checkpoints, preserving full attention context and target
weighting. This common scorer compares weights rather than changing scoring
precision per training policy. Final sampled layer diagnostics also use the
same BF16/default forward. All those allocations count in job peaks. Independent
native scoring must verify the streamed scores; it runs in a separate audit
process and is not misrepresented as part of the production job peak.

## Measurement

One GPU worker, UV-managed Python/PyTorch, four CPU threads. Each case includes
model construction, loaded optimizer, corpus cache, initial evaluation/probe,
50 updates, final evaluation/diagnostics and checkpoint export. Record every
memory interval before resetting peak counters; count normal workspaces and
subtract nothing. Verify zero allocated/reserved memory between cases.
Report allocated and reserved separately; driver/context memory is excluded.

The first20 updates warm up; all final30 contribute to timing. Use synchronized
wall time and CUDA events separating forward/backward/optimizer. No gradient
hooks or forward hashes in measured passes. Batch hashing, scalar diagnostics
and disk logging occur outside the timed update. Therefore report loop wall
time separately. Use three consecutive10-update mean times to test stability;
no outlier removal. Save per-update losses, preclip norms, sampler/batch hashes,
full endpoint model/Adam and initial gradient tensors. Use one shared study
runner for all architectures and execution policies; maintained code stays unchanged.

## Frozen decisions

For each scope (two narrow corpora and two single-fixture full controls) and
each candidate chunk policy, require:

- Finite weights, gradients, optimizer moments and sampled activations.
- Initial loss relative error <=1e-6, full-gradient relative L2 <=0.002 and
  each parameter <=0.02, relative to same-backend native FP32. Norm floors remain
  1e-12 globally and1e-8 per tensor. Precision changes do not require identical
  gradients to the BF16 reference; disclose that intervention explicitly.
- Every candidate job-allocated peak <=0.85 times both BF16 native and
  same-backend FP32 native peaks.
- Median warmed update-wall ratio <=1.25 versus both references; all relevant
  three-block timing stability ratios <=1.25.
- Every final common-scorer NLL <=1.01 times both references after50 updates.
- Independent FP32 initial-gradient replay <=1e-5 global and1e-4 per parameter,
  and loss relative error <=1e-6. Do not require bitwise identity. BF16 reference
  gradient replay is outside this gate; H115 already showed its variability.
- Native validation agrees with streamed scores within1e-6 relative error;
  complete batch streams, source and endpoint model/Adam hashes verify.

A pass earns only a prospective broader replication/duration decision, not
automatic fresh training in this protocol or a global breakthrough. Report
mean, median, sample variance (null for one fixture), ranges, individual outcomes
and the Pareto tradeoffs. Reject or hold failed configurations; complete the
fixed grid for ordinary threshold failures, stop for runtime/nonfinite/integrity
errors and preserve partial evidence.

[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
distinguishes asynchronous timing, allocated/reserved memory and TF32 choices.
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend-dependent numerics. This applies established checkpointing,
precision and classifier-chunking techniques; it makes no novelty claim.
