> Historical failed baseline. Its implementation has been retired; current model results are in [Step 2 results](step_2_results.md).

# Paragraph compressor: reconstruction experiment

Status: implemented and verified; the eight-example memorization gate and the
approved 2,000-update FineWeb-Edu pilot are complete. **Exact recovery did not
generalize: 0/1,000 validation paragraphs matched exactly.** This is a functioning
experimental autoencoder, not a successful universal or lossless compressor.

## Final full validation

| Measurement | Result |
| --- | ---: |
| Complete validation paragraphs / targets including EOS | 1,000 / 93,516 |
| Reconstruction NLL | 2.327738 |
| Free-generation exact text match | 0/1,000 (0%) |
| Token edit distance / reference tokens | 0.666447 |
| Character edit distance / reference characters | 0.580567 |
| Mean cosine similarity between valid memory slots | 0.129667 |
| Complete model parameters | 3,227,136 |
| Peak allocated / reserved memory sampled during training | 221.30 / 418.00 MiB |
| Maximum-length preflight allocated / reserved memory | 194.98 / 232.00 MiB |
| Summed training-update time, excluding evaluation/checkpointing | 409.82 seconds |
| Resumable checkpoint file bytes (including optimizer) | 39,080,631 |

Hardware: RTX 4070 Laptop GPU, PyTorch 2.14.0+cu132. Parameters are FP32; matrix
operations use BF16 autocast. Seed 17, microbatch four, four-step accumulation,
AdamW 3e-4, weight decay 0.01, warmup 100 updates, gradient clipping one.
The 6 GiB allocated-memory budget passed without reducing microbatch size.
These are this pilot's observed resources, not a speedup over a matched baseline.

## Does the decoder use its memories?

The same 64 shortest validation paragraphs were evaluated under all three
conditions. Shuffled donors are rotated within groups of 16 sorted by length;
lengths/masks can still differ, so this is not an exactly length-matched control.
These sampled diagnostics are separate from the full-validation result above.

| Memory condition | Reconstruction NLL | Exact match | Token edit rate | Character edit rate |
| --- | ---: | ---: | ---: | ---: |
| Correct | 1.782000 | 0/64 | 0.584167 | 0.512495 |
| Zeroed | 7.863860 | 0/64 | 1.570974 | 1.263459 |
| Shuffled | 8.283787 | 0/64 | 0.996360 | 0.872483 |

Correct memories help substantially, but exact reconstruction still fails. Edit
rates can exceed one when generation inserts many extra tokens. Teacher-forced
loss improvement alone does not establish dependable free generation.

## Names, numbers, negation and repetition

Two fixed probes per category produced 0/2 exact matches in each category for the
fresh pilot model. They are short diagnostic examples, including inputs below the
training minimum of 32 tokens; this is not a calibrated semantic-fidelity score.
All input/output pairs remain in the evidence. For example:

> Input: The order was 17 batteries. Nine arrived. Eight are missing.

> Output: The order was bies. 17 17 are Nine arrived. Nighting. missing miss.

This input contains 19 BPE tokens and produces three FP32 memory vectors. The
vector/mask payload is **3,075 bytes**, versus **60 UTF-8 bytes** or **76 bytes**
of int32 token IDs. The model reduces sequence positions but does not demonstrate
smaller raw-text storage. Metadata/file overhead and decoder weights cost extra.

The main unresolved issues are unseen-paragraph recovery, order/detail retention,
generation drift, and compression quality versus slot count. No additional tuning
or longer run is implied by this report. The reserved test set remains unused.

[Pilot evidence](../records/paragraph-pilot-v1.json) ·
[Notebook execution check](../records/paragraph-notebook-check-v1.json) ·
[Data integrity audit](../records/paragraph-data-audit-v1.json)

## Protocol and implementation

The architecture is a two-block bidirectional encoder, learned pooling over
eight-token spans, and a two-block autoregressive decoder with cross-attention
to pooled memories. Width is 256, with four heads and Curve-Wide FFNs throughout.
Input paragraphs are limited to 256 tokens. Slots are determined by length;
fact/concept roles, learned semantic segmentation and universal lossless
compression are not established. The decoder sees no original encoder states.

The selected corpus is the streamed `HuggingFaceFW/fineweb-edu` `sample-10BT`
at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The bounded subset contains
50,000 train / 1,000 validation / 1,000 reserved test paragraphs, selected from
20,742 documents. Source documents are assigned before extracting newline-delimited
paragraphs, with exact paragraph deduplication across splits. Filtering to 32–256
tokens biases the subset toward short paragraphs; it is not a universal-language test.

Tokenizer training uses only the training-document candidate pool; final training
paragraphs are a subset of that pool. Validation/test text does not fit the tokenizer.
Tokenizer round trips preserve exact selected text, including spaces and punctuation.
Source and split hashes live in the local dataset manifest.

The separate memorization run recovered all eight short training examples exactly
at update 300. The pilot starts again from a fresh initialization, not that checkpoint.
[Memorization evidence](../records/paragraph-memorization-v1.json).

Reconstruction NLL is conditioned on memories encoded from the target paragraph.
It must never be ranked against ordinary next-token language-model NLL.
See [Step 2 results](../../../docs/step_2_results.md) and the
[interactive notebook](../notebooks/paragraph_compressor_inference.ipynb).
