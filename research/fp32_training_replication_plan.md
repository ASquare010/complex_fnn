# H117 — fresh three-seed replication of the FP32 memory recipe

Prospective hypothesis following H116, whose completed resource screen made
progress toward the primary VRAM objective. Its passing narrow-model settings
earn this larger test. All prior failures, sources and results remain unchanged.
No claim of a new FFN or completed architectural target is made.

## Question and fixed allocation

Does default-attention FP32 decoder/classifier chunking preserve quality over
800 updates from random initialization across three seeds and both corpora,
while retaining at least 15% lower full-job allocation and practical update time?

Use the unchanged narrow GELU model: width 384, hidden 456, eight layers, six
heads, vocabulary 4,096, batch 8 and context 512. It has 9,099,648 total and
2,801,664 FFN parameters. Keep the hashed WikiText-2 and TinyStories caches and
their respective frozen train-only tokenizers. These are previously used
development validation splits, not a new held-out test. Do not pool raw NLL
between tokenizers or access a test split.

Fix seeds **101, 113, 127** before execution. These differ from the H112/H116
selection seeds; do not select or drop seeds after results. Three policies:

1. `bf16_default_native`: BF16 decoder and native BF16 classifier.
2. `fp32_default_native`: FP32 decoder and native FP32 classifier.
3. `fp32_default_chunks`: FP32 decoder and checkpointed 512-token FP32 classifier chunks.

All use default SDPA and whole-block checkpointing. Native FP32 isolates the
effect of chunking; BF16 native tests the complete practical recipe. Do not
interpret the precision intervention as an FFN architecture improvement.

**18 fresh trials × 800 updates = 14,400 updates / 58,982,400 target presentations.**
Every arm of a corpus/seed starts from the same CPU-generated step-zero model
and empty Adam state. Initialization source and recipe are frozen first; save
and hash six initial checkpoints before CUDA training. Independent audit must
regenerate those initial states and verify them exactly. Policy order rotates
by seed/corpus, then reverses on alternating fixtures. One GPU worker at a time.

## Training, evaluation and artifacts

Keep the established fixed recipe: LR 0.0006 for all 800 updates, AdamW betas
(0.9, 0.95), epsilon 1e-8, matrix decay 0.1, existing no-decay groups, norm clip1,
no width calibration, and FP32 weights/Adam moments. TF32 stays disabled, with
four CPU threads and no deterministic-algorithm/workspace-policy changes.
Sampling uses seed +10000, independently of policy. No hyperparameter tuning.

The common scorer remains streamed BF16/default evaluation with unchanged
attention context and valid-target weighting. Score complete validation at
steps 0, 200, 400 and 800. Save complete model/Adam/sampler states at 200, 400
and 800, plus initial gradient tensors and per-update histories. Final sampled
layer diagnostics use the same BF16/default forward in all arms. Record
per-layer gradient statistics at 1, 20, 200, 400 and 800. Final NLL at step800
is primary; do not select a best checkpoint. Earlier scores describe convergence.

An initial full loss/backward probe per trial adds 18 backwards, resets the
sampler, and makes no optimizer update. Reuse H116's complete CPU-double
qualification (20 comparisons, four references, 24 backwards) of the unchanged
loss adapter. Main execution therefore uses **14,418 backwards**. Planned audit
adds 12 FP32 initial-gradient replays and zero optimizer updates: **14,454 total
backwards** including qualification. A source/runtime/nonfinite failure stops
and preserves partial artifacts; there is no automatic scientific retry.

## Measurement and independent audit

Count the entire job: construction, optimizer, corpus caches, initial evaluation
and probe, all updates, intermediate evaluation/checkpoints, final evaluation,
diagnostics and serialization. Record each interval before resetting memory
peaks, charge normal workspaces and subtract nothing. Verify zero allocated
and reserved bytes between trials. Report allocated versus reserved distinctly;
driver/context memory remains outside the PyTorch allocation measurement.

Twenty updates warm up. All final 780 contribute synchronized wall and CUDA-event
update timing. Batch hashing, scalar diagnostics and serialization are outside
the timed update; retain complete-loop wall time separately. No gradient hooks
or forward hashes occur in timed updates. Three consecutive 260-update mean
wall times test stability. No outlier removal or thermal-based run selection.

Freeze the complete study and audit before independent scoring/replay. Audit
six regenerated initial checkpoints, all 54 trained checkpoints, 18 saved
gradient hashes and all 14,400 sampled batches. Native unchunked BF16 scoring
checks six unique initial states and all 54 trained states: **60 native scores**.
Twelve independent FP32 initial backwards verify replay numerically; BF16
gradient replay is explicitly outside that gate. Audit allocations are separate
from the reported training-job measurement.

## Decisions fixed before results

For every seed in each corpus require the chunked candidate to have:

- Full step800 common-scorer NLL <=1.01 times both references, verified by native scoring.
- Full-job allocated peak <=0.85 times both references.
- Median warmed update time <=1.25 times both references, individually for every seed.
- All relevant three-block timing stability ratios <=1.25, and finite recorded
  losses, gradients, optimizer moments, weights and sampled activations.
- Initial loss relative error <=1e-6, global gradient relative L2 <=0.002 and
  per-parameter error <=0.02 versus native FP32 at identical starting weights.
  Keep norm floors 1e-12 globally and1e-8 per parameter.
- Independent FP32 replay <=1e-5 global /1e-4 per parameter and <=1e-6 loss error.
- Native/streamed score relative error <=1e-6 and exact initialization,
  checkpoint, state and batch-stream integrity checks.

A corpus qualifies only when all three seeds pass. The two-corpus component
qualifies only if both do. Complete the fixed grid despite ordinary quality,
memory or runtime gate failures. Report every seed, means, medians, sample
variances, ranges and paired ratios. Three seeds provide limited statistical
power; neither a passing mean nor tiny NLL changes prove a general quality gain.

A pass establishes only a scoped 800-update memory/quality component. Larger
models, other contexts/corpora, inference performance and the parameter-efficient
architectural goal remain separate requirements. No default is changed here.
Use the existing UV-managed runtime, preserve compact exports and raw local
artifacts, and document any native runtime failures without claiming a cure.
