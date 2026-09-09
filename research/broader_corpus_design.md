# Next experiment design: broader language data

Status: DESIGN, not an executed experiment or a frozen training protocol.
The TinyStories three-seed and joint conditioning checks now pass locally.
Another corpus and controlled training budgets are the next substantive gaps.

## Candidate data source

Use the official [Salesforce WikiText repository](https://huggingface.co/datasets/Salesforce/wikitext/tree/f776294184f13b8ff2337b3841cf9269a6216d1e/wikitext-2-raw-v1),
pinned to revision f776294184f13b8ff2337b3841cf9269a6216d1e, with the
wikitext-2-raw-v1 configuration. Its [pinned data card](https://huggingface.co/datasets/Salesforce/wikitext/blob/f776294184f13b8ff2337b3841cf9269a6216d1e/README.md)
describes Wikipedia articles and preserved case, punctuation and numbers; the
raw variant precedes replacement of out-of-vocabulary tokens. The card lists
36,718 train rows, 3,760 validation rows and 4,358 test rows. These are rows,
not independent articles. Retain attribution and the listed CC BY-SA 3.0/GFDL
metadata with downloaded data. The older Salesforce blog link redirects to
its general AI page, so use the pinned repository as the reproducible source.

This corpus is a practical change from short synthetic stories. It cannot by
itself establish robustness over arbitrary language domains or reasoning tasks.
Our 4,096-token BPE and context-128 protocol will not yield directly comparable
perplexities to the dataset's published word-level benchmarks.

## Preparation before allocating GPU compute

1. Download and hash official train/validation files at the pinned revision.
   Inspect row and article boundaries before defining concatenation and EOS
   handling; preserve article text and split boundaries. Keep the test split
   unscored during recipe selection. Do not assume rows are whole documents.
2. Define and check train/validation deduplication. Train a shared 4,096-entry
   byte BPE on training text only. Retain source, tokenizer and token-array hashes,
   token counts, duplicate accounting and a reproducible reconstruction check.
3. Freeze the evaluation windows before any model is scored. Use enough validation
   text to avoid making choices from only a few articles, and reserve an unscored
   region or official test split for the final locked comparison.
4. Check finite backward, real parameter counts and identical common initial
   tensors for the existing four recipes on this cache. Shared training, sampling,
   attention and numerical settings stay fixed. Models train from scratch.

## Comparison to freeze after data inspection

Carry full SwiGLU, full GELU, calibrated narrow SwiGLU and native BlockShuffle
at width 384 / eight layers. The existing FFN initializations and optimizer
multipliers are explicit parts of each recipe. Do not silently add the
post-training floor or fused inference to architecture-training comparisons.

Allocate the same small global-learning-rate grid and screen budget to every
recipe. Freeze candidate promotion and tie-breaking rules before scoring.
Report all trials and choose by the same criterion. This equalizes new tuning
work; it does not erase the unequal historical architecture search.

If the candidate earns longer training, use all controls and at least three
paired seeds. Separate the fixed-token transfer comparison from a convergence
study: 800 steps alone did not establish convergence on TinyStories. Define a
bounded longer schedule, checkpoints and a quantitative stopping criterion
before running that stage. Retain parameter/compute accounting, loss curves,
activation/gradient diagnostics, clipping frequency and memory. Speed claims
need paired measurements on the new trained checkpoints.

No broader-corpus files have been downloaded or scored yet. The FlashMHF
comparison in [current comparator audit](current_comparator_audit.md), longer
context tests, convergence and verified novelty also remain outstanding.


Implementation update: the pinned cache has been prepared and independently reconstructed. The [twelve-trial WikiText result](wikitext_screen_results.md) is complete and fails quality promotion. Earlier design statements describe the pre-experiment state; the official test remains unfetched and unscored.
