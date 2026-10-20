# Three-model comparison: attention residuals do not fix fact recovery

Evaluation date: 2026-10-14. The user intentionally ended attention-residual
training at37,045/48,762 updates and explicitly does not want further training.
The checkpoint and run status remain paused; no training was resumed. Compare
this retained checkpoint with the completed plain64 and original512 checkpoints.

The attention-residual model slightly improves language prediction over plain64
at the same37,000-step validation point. It does not establish better factual
reading. All three recover0/103 notebook answers exactly and0/32 changed facts.
The new model's higher missing-fact score comes with substantially more false
claims that an available fact is missing. Retain the original512 as the strongest
language model among these retained checkpoints.

## Training budget and fair comparison

| Property | Original512 | Plain64 | Attention residual64 |
|---|---:|---:|---:|
| Evaluated `last.pt` step | 48,762 | 48,762 | 37,045 |
| Scored training-label exposures | 361,357,104 | 361,357,104 | 274,371,917 |
| Trainable parameters | 23,209,216 | 23,209,216 | 23,217,408 |
| Microbatch / effective batch | 32 / 32 | 16 / 32 | 16 / 32 |

All logged example indices, labels, learning rates and sampler cursors match
through step37,045. Data, tokenizer, frozen encoder, fresh seed47 core
initialization and original two-epoch schedule match. The new model adds8192
depth-query parameters and stops11,717 updates before the older models. Therefore
the retained-checkpoint generation/test comparison is not an equal-update
architecture ablation. Older checkpoints at step37,000 were not retained for a
generation comparison. Saved development logs permit this equal-step comparison:

| Development at exactly37,000 updates | Original512 | Plain64 | Attention residual64 |
|---|---:|---:|---:|
| CE, lower better | 1.846516 | 2.259391 | 2.239513 |
| Token accuracy | 60.3123% | 52.7197% | 52.9645% |

Attention residuals gain0.2448 percentage points over plain64 here and improve CE
on14/17 development sources. That is a small gain in one initialization, not a
general architecture result. The memory source has slightly higher token accuracy
but worse CE at this point: plain64 CE0.41448/accuracy88.8332%, attention residuals
CE0.44303/accuracy89.0530%. Template token prediction is not factual recall.

## Fresh test evaluation of retained checkpoints

All7,646 original test segments and1,757,421 assistant labels were evaluated,
with the same frozen FP32 encoder codes and original segment-level grouping.

| Test metric | Original512 | Plain64 | Attention residual64 |
|---|---:|---:|---:|
| CE | 1.806954 | 2.216948 | 2.240588 |
| Token accuracy | 60.9990% | 53.5087% | 53.0441% |
| Perplexity | 6.0919 | 9.1793 | 9.3989 |

The stopped attention-residual model is0.4647 percentage points below the
completed plain64 model and7.9550 points below original512. Its test CE is worse
than plain64 on16/17 sources; the memory source alone has slightly better CE
(0.37015 versus0.38239) and virtually identical token accuracy (90.1055% versus
90.1136%). These final-budget differences do not overturn the modest matched-step
validation improvement shown above.

A5,000-draw paired document bootstrap for new-minus-plain64 gives CE difference
[0.02116,0.02682] and accuracy[-0.5138,-0.4153] percentage points. Against
original512: CE[0.42534,0.44236], accuracy[-8.1174,-7.7982] points. These describe
document-sampling uncertainty for the retained checkpoints, not training-seed
uncertainty or a prediction of what further training would do. The test split
was previously used in this research and is not a newly blinded benchmark.

## Fact recovery

All312 historical prompt cases were reused without selecting by results. Greedy
generation uses the model's own64/512 raw boundary, live encoding of completed
generated blocks, a64-new-token cap, equal PAD/BOS suppression and EOS stopping.
No reference answer tokens are supplied during generation.

| Probe | Original512 | Plain64 | Attention residual64 |
|---|---:|---:|---:|
| Correct old memory: exact notebook answer | 0/103 | 0/103 | 0/103 |
| Changed old fact: exact updated answer | 0/32 | 0/32 | 0/32 |
| Both original and changed answers correct | 0/32 | 0/32 | 0/32 |
| Truly missing fact: correct missing response | 13/25 | 17/25 | 22/25 |
| Available fact falsely called missing, lower better | 49/103 | 65/103 | 87/103 |
| Simple color, short prompt | 8/8 | 0/8 | 0/8 |
| Simple color, old compressed fact | 0/8 | 1/8 | 0/8 |
| Simple color, old fact codes blocked | 1/8 | 1/8 | 0/8 |

The false-missing metric requires the exact benchmark missing-answer template,
with EOS: “I don't have a saved location for that notebook.” It is a narrow,
auditable count, not a general semantic abstention classifier. The new model
uses that response on87/103 answerable questions. Manual review of its other16
answers found wrong locations, wrong people, repetition, or irrelevant text.
It also has zero complete expected-location substring matches, so its zero exact
score is not explained solely by harmless punctuation or wording differences.
Plain64's sole location substring hit names Bilal when the question asks Vera;
that remains an incorrect factual answer.

For example, with the true fact that Vera's notebook is in the yellow drawer in
the kitchen, attention residuals produce “Vera's notebook in the purple cabinet
in the kitchen.” On the short Amina/blue color prompt, it begins “Amina's favor
of the Big Bang Theory” and repeats. Short color prompts contain59 tokens:
their facts initially remain raw for both models, but64-token models cross their
boundary after five generated tokens.

Blocking the old notebook fact codes gives exact answerable scores1/103,0/103,
0/103 and missing-response scores10/25,0/25,0/25. Attention residuals emit EOS on
only1/128 blocked notebook cases, versus128/128 with correct memory. Thus memory
affects behavior; this is not evidence that it is read correctly. Changing the
old value changes5/32,13/32,10/32 outputs respectively, but none of those changes
produces a correct original-and-updated pair.

Blocked controls hide only fact-containing old code slots in every layer while
preserving positions and all later question tokens. The slots also contain
neighboring formatting text. Changed facts preserve the complete question token
sequence but can shift earlier prefix lengths/positions. These are synthetic
notebook and small color tests, not broad real-world reading benchmarks.

## Implementation checks and evidence

Evaluation uses the public attention-residual forward/packed paths, including
all learned depth queries. Its16 query-vector norms range2.17–5.92. An independent
single-window reference with depth mixing gives identical checked logits;
blocked-content mutation checks and four independent serial generation replays
per model pass. The old evaluator's unbound plain forward path is not used for
the attention-residual model.

Both older models reproduce all312 prior generated outputs and full test
metrics exactly. Each checkpoint's Adam counters match its reported step;
recorded source hashes pass. Original and plain64 last weights equal their best
weights; the new last checkpoint is37,045 and its best-dev checkpoint is37,000.
Only the predefined `last.pt` checkpoint is tested here; no checkpoint is selected
using test results.

No optimizer was constructed or updated. The evaluator finished with exit0,
released the shared lock, and verified unchanged model/encoder/checkpoint/source
content. A separate artifact check recomputed test totals and checked source
archive bytes and checkpoint hashes. No further training is planned by this task.

- [Evaluator](../tools/compare_token_memory_attnres.py)
- [Full comparison](../dump/token-memory-attnres-comparison-20261014-step37045/report.json)
- [False-missing analysis with case IDs](../dump/token-memory-attnres-comparison-20261014-step37045/abstention_analysis.json)
- [Artifact verification](../dump/token-memory-attnres-comparison-20261014-step37045/verification.json)
- [Attention-residual outputs](../dump/token-memory-attnres-comparison-20261014-step37045/attnres64-predictions.json)

All prompts, all three models' outputs, per-segment statistics and source archive
are retained in the same directory. Attention-residual checkpoint SHA256:
`70b666f9e601fc33235d16f930a328a24e79062204acd61e20e60300cb7d8a1d`.
