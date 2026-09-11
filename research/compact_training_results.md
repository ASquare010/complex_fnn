# H125: complete-step compact training screen

**Both corpora pass the predeclared memory, timing, quality and numerical gates.**
Compact RMSNorm plus gradient staging preserves the memory saving through
AdamW, global clipping, evaluation and serialization. This is a short
continuation result, not a fresh-run or multiseed breakthrough.

| Corpus | Baseline peak MiB | Candidate peak MiB | Saved | Event ratio | Wall ratio | Final NLL baseline / candidate |
|---|---:|---:|---:|---:|---:|---|
| wikitext2 | 281.462 | 242.552 | 13.82% | 1.0467x | 1.0443x | 4.963374 / 4.963405 |
| tinystories | 276.214 | 236.373 | 14.42% | 1.0557x | 1.0497x | 2.627666 / 2.627682 |

Both arms use chunked FP32 loss and checkpoint-input offload. The candidate
adds analytical RMSNorm backward and pinned gradient staging, then restores
all CUDA gradients before the unchanged global clipping and default AdamW
step. Evaluation uses the ordinary RMSNorm path, including BF16. Parameter
count remains 9,099,648; there is no parameter reduction or new activation.

Complete-job peaks include initialization, ten warmups, all training phases,
evaluation, transfer/restoration, clipping, updates, serialization and checks.
Combined pinned-host allocated peak is 112.100 MiB; active/cached allocation is
recorded separately. CUDA allocated/reserved memory excludes driver/context
and other processes. Timing uses the last 20 of 30 complete updates; means,
medians, sample variances, split-half stability and individual rows are saved.
This instrumented continuation does not prove sustained production throughput.

## Evidence

- Four runs continue the same two trained checkpoints from step 800 to 830:
  **120 updates, 491,520 training targets, 122 backwards including qualification**.
- Two tiny-model backwards activate global clipping. Independent FP64 clipping
  error is at most 8.781e-08. FP32/BF16 evaluation outputs match exactly.
- Independent NumPy AdamW checks pass: maximum first-update parameter relative
  error 3.593e-08, moment error 3.790e-08; first raw gradient discrepancy
  2.662e-07. First/final steps, states, moments and finiteness are checked.
- All 120 sampled batches and sampler states reconstruct exactly. Four fresh
  native validation passes agree with the recorded streamed scores within
  4.880e-09 relative error. Study evaluation contributes eight full scores.
- Candidate restoration runs once per backward: all 50 parameters and
  36,398,592 gradient bytes restored before clipping. Qualification verifies
  bitwise staged round trips. No completed scientific case was repeated.
- Twelve tensor artifacts, 628 complete-job memory intervals and twelve zero
  GPU boundaries are preserved. All 61 maintained file hashes remain unchanged.

The unchanged H120 training and audit functions are reused through scoped
factory/loss/backward adapters. The adapter restores patched methods on exit.
It supports one backward per batch; accumulation, distributed execution and
concurrent streams remain outside the tested scope. No maintained default
was changed.

## Next required evidence

Run longer matched training from the original initialization, then replicate
seeds and scale. Include the conventional native-loss control: matching the
chunked baseline alone cannot resolve H117's earlier native-versus-chunked
quality failure. Check endpoint quality and actual complete-job memory in every
run. The successful short gate does not rescue H117/H119 or establish the
original architectural/parameter-efficiency goal.

[Plan](compact_training_plan.md), [adapter](../results/compact_training_v1/adapter.py),
[audit](../results/compact_training_v1/audit.py), [receipt](../results/compact_training_v1/receipt.json).
