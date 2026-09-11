# H114 — FP32 classifier chunks in complete model continuations

Prospective protocol; freeze code, data and checkpoints before any model execution.
H113 proved an autocast accumulation mechanism and qualified FP32 chunks only for
this screen. Its isolated timing was highly variable. H112 BF16 chunks saved about
33% of whole-job allocated memory but failed the two-corpus quality gate.

**Hypothesis.** Keeping the decoder in BF16 and evaluating only the vocabulary
projection/cross entropy in FP32, in checkpointed 512-token chunks, can preserve
the whole-job allocation benefit with acceptable full-model gradient error and
runtime. This changes numerical execution, adds zero parameters, and is not a
new activation or a claim of novel classifier chunking.

## Fixed fixtures and budget

- All 12 H112 step-800 models **with their saved Adam states**: WikiText-2 and
  TinyStories, seeds 61/73/89, each original block/BF16-chunk training policy.
  Same narrow GELU, width 384, hidden 456, eight layers, six heads, vocabulary
  4096, batch 8, context 512. The two starting policies are correlated fixtures,
  not six independent seeds per corpus.
- Two H078 step-3200 full controls with saved Adam states: GELU hidden 1536 and
  SwiGLU hidden 1024, seed 17, WikiText-2, batch 16/context 128. These add model
  breadth, not multi-seed full-model evidence. No context extrapolation.
- Four executions per starting state: native BF16 classifier, BF16 chunks,
  native FP32 classifier, FP32 chunks. All use identical whole-block decoder
  checkpointing, FP32 weights/gradients/Adam and BF16 decoder autocast.
- 50 sequential optimizer updates per execution: 20 warmup, 30 measured, grouped
  into three blocks of ten. Total **56 cases, 2,800 optimizer updates**. These
  are short continuations of existing trained models, not fresh training runs.
- All continuations use fixed LR 0.0006, AdamW betas (0.9,0.95), epsilon 1e-8,
  matrix decay 0.1, scalar decay zero and clipping at norm 1. Saved moments and
  optimizer step are retained. This deliberately replaces H078's terminal LR;
  the replacement is identical across methods and will be recorded.
- A new sampler seed (training seed + 40000) is shared within each fixture;
  it does not resume the source sampler. Preserve every batch hash.
- One initial probe backward per case with no update, then restore sampler
  state. Save all parameter gradients on CPU. An independent audit compares
  the FP32 methods and verifies deterministic recapture. Total profile backward
  count is 2,856 including the probes, before independent audit work.
- Full validation before and after every continuation uses H111's BF16
  classifier-chunk evaluator, identical for all methods. Independently rescore
  source and final models through native BF16 forward. No test-set scoring.

## Measurements and fixed gates

Record every warmup and measured update, loss, preclip gradient norm, layer
gradient statistics, parameter/gradient/optimizer bytes, finite states, batch
hashes, final checkpoint and complete validation NLL. Save both allocated and
reserved CUDA peaks for **every phase before resetting counters**. Whole-job
maximum includes construction/data/Adam loading, both evaluations, initial
probe, all warmup and measured updates, and final diagnostics/serialization.
CPU copies are not extra CUDA reference models. Count workspaces normally;
subtract zero bytes. Driver/context allocations are outside PyTorch counters.

CUDA events mark forward, backward and optimizer boundaries without a host
synchronization between phases; synchronize at the end of each update. Record
wall time separately. Event elapsed time includes stream scheduling/submission
gaps and is not pure active-kernel time. First 20 updates are excluded from
timing summaries but included in memory, training and token totals. No outlier
removal. Report all 30 measured timings, means, medians, variance, and each block
of ten. A fixture's timing is stable when the ratio of largest to smallest block
mean wall time is <=1.25. Wall timing includes forward/backward/clipping/Adam and
final synchronization, excludes sampling, hashes, validation and disk logging;
also record complete case elapsed time.

For FP32 chunks to earn a fresh multi-seed LM experiment in a scope (each narrow
corpus, each individual full architecture), require **all**:

1. Finite losses, weights, gradients and Adam tensors in every case.
2. On each unchanged initial model: relative loss difference <=1e-6, global
   gradient relative L2 <=0.002 and every parameter relative L2 <=0.02 against
   native FP32 classifier. Denominator floors: 1e-12 globally, 1e-8 per tensor.
   The decoder remains BF16 in both methods. No demand for identity to BF16.
3. Every paired whole-job allocated peak <=85% of native BF16 classifier.
4. Median paired **median update wall-time** ratio <=1.25 versus native BF16,
   and the timing stability rule passes in both reference and candidate for
   every fixture. Event timings are corroborating diagnostics, not a substitute
   for wall timing. A stability failure requires a separate measurement design,
   never removal of inconvenient steps.
5. Native BF16 validation NLL after 50 updates <=1.01 times paired native BF16
   continuation for every fixture. This is a short continuation check, not a
   convergence, equivalence, held-out-test or long-duration quality proof.

Ordinary gate failures do not truncate the grid. Runtime failure, source drift
or nonfinite training stops execution and preserves partial outputs. No automatic
retry/restart of completed cases. Fresh model/optimizer per case, one GPU process,
rotating/reversing method order across fixtures. At case boundaries only, collect
Python garbage, clear autocast and retained cuBLAS workspaces, empty the allocator
and require zero allocated/reserved bytes. CUDA context persists across cases;
this is not fresh-process isolation. Current UV-managed Python and installed
PyTorch are used; four CPU threads, TF32 disabled. Direct UV-managed executable
avoids the previously recorded UV-dispatcher panic without changing packages.

## Interpretation and prior work

For token count N, vocabulary V and chunk C, FP32 logits require 4NV bytes when
materialized, versus 4CV per chunk, before cross-entropy intermediates. Chunking
keeps the exact sum/count loss in real arithmetic; floating point gradients may
differ. Doubling classifier precision can still reduce memory when C is small
relative to N. This arithmetic is a motivation, not a whole-job memory proof.

H113's independent FP64 gradient checks and rounding witness are the mechanism
evidence. [PyTorch autocast](https://docs.pytorch.org/docs/2.14/amp.html) documents
disabled subregions; [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
and [events](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)
support synchronization/timing and allocator accounting. Classifier-memory work
already exists, including [Cut Your Losses](https://arxiv.org/abs/2411.09009).
These ordinary PyTorch chunks are not its fused kernel. No SOTA or novelty claim.

Do not edit maintained model/trainer code or earlier frozen studies. Publish a
compact source/result bundle, readable report and plots. Preserve large tensors
locally with hashes. The broad VRAM/architecture research goal remains open.
