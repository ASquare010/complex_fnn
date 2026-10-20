# Final 64-token rollover comparison (2026-10-14)

The original 512-token model performs better. The user completed the fresh
64-token run at step 48,762. A read-only comparison of the final checkpoints
finds worse language prediction on every development and test source, without
reliable recovery of old facts. Keep the original as the language foundation.
This does not prove that every shorter-context design will fail.

## Matched training, verified from the actual runs

Both models have 23,209,216 trainable parameters, the same fresh seed47
initialization procedure, frozen original encoder, v4 data and tokenizer,
assistant-only CE, FP32, AdamW, learning-rate schedule, and two epochs. Every
one of the 48,762 logged updates has identical sample indices, target counts,
learning rate and sampler cursor: 361,357,104 assistant-label exposures total.
All Adam counters are 48,762; final model tensors equal best-checkpoint tensors
for both runs. Checkpoint sidecars and all recorded source hashes pass.

The intended change is raw-context rollover, 512 to 64. The new model moves
completed text into frozen vectors sooner; the underlying text history is the
same. Microbatch32 versus16 and on-demand versus cached encoder grouping can
change floating-point rounding. This is one initialization, not a multi-seed
architecture ranking.

## Language prediction

| Metric | Original, raw512 | New, raw64 |
|---|---:|---:|
| Development CE, lower better | 1.804710 | 2.214284 |
| Development token accuracy | 61.0132% | 53.5046% |
| Test CE, lower better | 1.806954 | 2.216948 |
| Test token accuracy | 60.9990% | 53.5087% |
| Test perplexity | 6.0919 | 9.1793 |

Development results are the saved full validation metrics. Test metrics were
freshly evaluated on all 7,646 original segments and 1,757,421 assistant labels,
using the same frozen FP32 codes for both arms. The new model loses 7.4903
percentage points of test accuracy; its test perplexity is 50.68% higher.
All 17 sources have worse CE and accuracy on both splits. A 5,000-draw paired
document bootstrap gives new-minus-original test CE [0.4019, 0.4184] and accuracy
[-7.6528, -7.3319] percentage points. These intervals measure document sampling
uncertainty, not training-seed uncertainty. The test split has been used in prior
research and is not a newly blinded benchmark.

Representative test sources:

| Source | Original CE / accuracy | New CE / accuracy |
|---|---:|---:|
| Chat | 1.5288 / 65.57% | 2.0541 / 55.64% |
| Stories | 1.5048 / 62.92% | 1.8246 / 56.57% |
| Memory | 0.2407 / 90.90% | 0.3824 / 90.11% |
| Reasoning | 1.6315 / 63.46% | 1.9115 / 58.23% |
| Education | 3.1569 / 39.61% | 3.4805 / 34.84% |

The high memory-source token accuracy includes predictable template words and
teacher-forced answer prefixes. It is not an exact factual-recall score.

## Free factual recall

Reuse all 128 notebook cases from the original final evaluation: 103 answerable
and 25 missing. Same prompt token IDs, greedy decoding, maximum64 new tokens,
and EOS rule in both arms. PAD/BOS outputs are suppressed equally. Generation
encodes completed real-token blocks live and clears the raw buffer at each
model's own boundary; it never feeds reference answer tokens back.

| Condition | Original | New |
|---|---:|---:|
| Correct old memory: exact answerable notebook answers | 0/103 | 0/103 |
| Correct old memory: correct missing-fact answers | 13/25 | 17/25 |
| Changed old fact: exact updated notebook answer | 0/32 | 0/32 |
| Block fact codes: exact answerable notebook answers | 1/103 | 0/103 |
| Block fact codes: correct missing-fact answers | 10/25 | 0/25 |
| Simple color: short prompt, correct color | 8/8 | 0/8 |
| Simple color: old compressed fact, correct color | 0/8 | 1/8 |
| Simple color: old fact codes blocked, correct color | 1/8 | 1/8 |

The new model's one matching notebook-location substring belongs to the wrong
person: expected “Vera keeps the notebook in the red cupboard in the hallway.”,
generated “Bilal keeps the notebook in the red cupboard in the hallway.” It is
not a correct fact answer. Both models score zero on correctly answering both
members of each original/changed-fact pair.

The new model improves missing-fact exact answers by four cases in this small
selection. One of eight old color answers is also correct, but one is correct
with the relevant codes blocked. These results do not establish reliable
extraction of factual content. Blocking facts substantially changes the new
model's behavior, including stopping: only11/128 blocked notebook generations
emit EOS, versus127/128 with correct memory. Dependence on memory content is
different from correctly reading that content.

The short color prompts are59 tokens, so all input facts initially remain raw
for both models. The new model crosses its64-token boundary after five output
tokens and often drifts during the answer. For example, the original answers
“Amina's favorite color is blue.”; the new model begins “Amina's favoritely a
fascinating story about a young girl named Lucy.” This is an observed failure
in a small diagnostic, not a general claim about every short prompt.

Blocked controls mask only old code slots intersecting the fact message, in
every layer, preserving all positions and the entire later question. Those
slots also contain neighboring formatting text. This differs from historical
controls that blocked all memory. Changed-fact controls preserve the full
question token sequence but may change prefix lengths and positions. Synthetic
notebook/color tests are not general real-world reading benchmarks.

## Verification and evidence

No optimizer was constructed or updated by this comparison. Both checkpoint
files, archived sources and encoder remain unchanged. Public model-forward
parity, blocked-content invariance and serial replay pass. A separate artifact
check recomputed saved test totals, checked archived source bytes, and reproduced
all128 historical original-model answers under the correct-memory condition.
Final comparison worker completed with exit0 and released the shared lock.

- [Evaluation tool](../tools/compare_token_memory_rollover.py)
- [Full metrics and checks](../dump/token-memory-rollover-comparison-20261014/report.json)
- [Artifact verification](../dump/token-memory-rollover-comparison-20261014/verification.json)
- [Prompt cases](../dump/token-memory-rollover-comparison-20261014/cases.json)
- [Original outputs](../dump/token-memory-rollover-comparison-20261014/original512-predictions.json)
- [New outputs](../dump/token-memory-rollover-comparison-20261014/rollover64-predictions.json)

Per-segment sufficient statistics and a source archive are retained alongside
these files. Original checkpoint SHA256:
`e17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d`.
New checkpoint SHA256:
`047d5bc295e837663fd294ffa968c6286198211cd73c25141063dc0ff88a1cb0`.

Both test passes took about25 seconds with cached codes. The312-case generation
passes took27.42/34.70 seconds, including input and live rollover encoding, with
unequal stopping lengths. These are this evaluator's timings, not a general
inference-speed claim. The pasted training rate is about3 updates/s; a matched
end-to-end training-time comparison is not established.

Reducing the raw window increases the need to use vectors, but ordinary
next-token CE still permits prediction from language patterns without recovering
the required fact. This run supports retaining the original512 window. Evidence
supervision remains a research direction; no new training is launched here.
