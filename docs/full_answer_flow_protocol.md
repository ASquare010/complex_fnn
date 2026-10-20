# Full-answer frozen-codec protocol — original preparation

This document preserves the original preparation protocol addressing the limited
supervision in the first-eight-token pilots. Subsequent bounded training used a
[reversible literal end-marker view](full_answer_flow_results.md), followed by a
[clean-code loss comparison](data_prediction_flow_results.md). Neither completed
comparison established useful vector generation. The counts and EOS padding
below describe the original preparation, not the later derived view. This does
not authorize restarting the paused full-v4 production run. The earlier decoder-reader pilot
and its independent verification are complete and did not show a held-out gain.
The parameter constraint remains 23,209,216 trainable parameters, with the
original encoder and decoder frozen. The earlier archived preparation protocol
remains a historical source snapshot; this document records subsequent work.

## Prepared data

`tools/audit_full_answer_flow_data.py` selects whole dialogues by a seeded hash
within each of the 17 original v4 sources. It retains every assistant label in
each selected dialogue, including short answers and real EOS labels. Prompt
spans are encoded in blocks of at most 64 tokens; previous assistant spans use
independent eight-token codes, matching the proposed output format. A short tail
is EOS-padded for the codec only; padding does not become an extra CE target.

| Split | Dialogues | Assistant target tokens | Vector targets |
| --- | ---: | ---: | ---: |
| Train | 4,352 | 2,578,931 | 325,409 |
| Development | 489 | 283,282 | 35,709 |
| Test | 504 | 284,085 | 35,830 |

This is 256 training dialogues per source and up to 32 per source in each
evaluation split. Some sources have fewer available held-out dialogues. Equal
dialogue allocation does not imply equal token allocation; report both pooled
and per-source results. Long examples contribute more supervised tokens.

Evidence is in `dump/full-answer-flow-data-audit-20261013`. The audit checks all
selected labels against the unchanged binary tokens/masks, block ordering, tail
padding, unique IDs, split separation, and source-file hashes. It excludes the
1,024 test IDs already used in the bounded probabilistic/flow experiments.
Near-duplicate text has not been ruled out. This selection is a development
protocol, not a permanently pristine benchmark.

An earlier selection missed the first 256 probabilistic-pilot test IDs and had
25 overlapping selected test dialogues. It was preserved under
`dump/full-answer-flow-data-audit-initial-exclusions-20261013`, then replaced
with the corrected selection above before any encoding or training. Do not use
the earlier selection for a new result.

The old pilots contain only 16,384 distinct target positions and 1,647 distinct
eight-token answers. Repeating those examples for thousands of updates does not
provide broad language exposure. Conversely, this larger selection is still a
bounded diagnostic, not evidence of adequate pretraining for a general chatbot.

## Causal packed computation

`tools/parallel_flow_queries.py` implements the independent-query function while
processing a dialogue's history once. Each query sees only the history before
its own target block and its own noisy active state. Queries cannot attend to
one another, and history cannot attend to queries. Later teacher-forced answer
blocks are available only to later queries.

Three CPU tests in `tests/test_parallel_flow_queries.py` check predictions and
all gradients against separately truncated forwards, future/other-query/padding
invariance, and invalid visibility rejection. These tests do not establish GPU
capacity or throughput. The dense attention mask still has quadratic cost;
packing alone must not be advertised as a measured speedup. The subsequent
GPU checks below address parity and capacity, not end-to-end training speed.

Before training, perform a zero-update full-size CUDA comparison of predictions,
parameter gradients, and total losses on real short and long dialogues. Include
Adam-sized buffers and frozen-decoder backward cost when measuring VRAM. Keep
MSE normalized by real vector targets and CE by real assistant tokens across the
whole effective batch. Decoder CE can be evaluated in chunks, with gradients
accumulated on output leaves and chained through the shared history once.

Cache immutable encoder codes for this bounded selection if useful. Do not build
a full-v4 prefix-state cache: position states can instead be expanded once for
the current batch. Measure encoder preparation, expansion, training, sampling,
and output decoding separately, then report end-to-end cost.

## Comparison and evaluation requirements

Freeze the data, loss schedule, update budget, sampling order, and solver-selection
rule before starting. Use a raw-token control with the same source information,
target positions, trainable budget and supervised-token exposure. The existing
48,762-update v4 model is a quality reference, not a matched-budget control.

Compare deterministic vector prediction with a conditional stochastic head.
Keep output coordinates and input representation fixed within that comparison.
Do not simultaneously introduce a new codec, head, reader, loss and dataset and
attribute any gain to a single change.

Evaluate complete answers and multiple-block predicted-vector feedback, including
EOS, repetition and short tails. Report teacher-forced and self-fed results
separately. Use held-out grounded questions with correct, changed, wrong and
blocked **old facts**, keeping the current question identical. Include unseen
values, wording and record formats. Score answer facts and abstention as well as
single-reference text agreement; alternative valid continuations need not match
the reference word for word.

For stochastic output, report several draws and distribution-sensitive metrics.
Decoder CE on one sampled vector is not autoregressive LM perplexity. Sampling
speed must include all denoising steps and frozen decoder work. Expanded memory
positions trade attention compression for easier access; any later selective
expansion or resampling is a separate experiment.

There is currently no evidence that this protocol, flow matching, or the frozen
codec can beat the token reference at general language generation. A useful next
result must transfer to held-out complete answers and preserve factual memory,
not merely memorize another small training set.

## Completed zero-update preparation

`tools/full_answer_flow_common.py` adds packed vector loss and a raw-token control.
The control has exactly 23,209,216 active trainable parameters using a tied
4096-by-256 vocabulary table, a biased input projection and eight 1646-wide FFNs.
Its FFN allocation differs from the vector model's 1773/1774 widths; it is not
the old GELU baseline architecture. Both retain the same attention/Branch-FFN
family. Raw CE sees earlier ground-truth tokens inside each eight-token block;
vector prediction models the block jointly. Their losses are not identical
conditional likelihoods even though original assistant labels match.

Nine CPU tests across `test_full_answer_flow.py` and
`test_parallel_flow_queries.py` passed. Full-size CUDA checks in
`dump/full-answer-flow-preflight-20261013/preflight.json` passed for short and
long real dialogues: packed versus independent query gradient differences were
below 5e-8, as were chunked versus naive raw-CE gradient differences. The
models, codec and protected source/data/checkpoint hashes were unchanged.
There were zero optimizer updates.

The repeated longest real dialogue (2509 assistant tokens, 314 vector targets)
used 2.434 GiB allocated/2.598 reserved for compact-vector microbatch8, including
Adam-sized moment buffers. Raw microbatch2 used 2.697/2.842 GiB. These isolated
forward/backward probes exclude optimizer-step temporaries, evaluation, saving
and initial encoding. They are not a training throughput or ETA guarantee.

`tools/prepare_full_answer_flow.py` completed in 187.281 seconds and saved bounded
immutable codes in `dump/full-answer-flow-data-20261013`. Counts match the table
above. Normalization uses all selected training vector targets only; dev/test
statistics are excluded. Sources, original data and reference weights passed
end-of-job hash checks. No optimizer was created. Preparation session32138
finished exit0; there is no worker left from this job.

Genuine target-code decoder checks scored 282324/283282 dev tokens and
283103/284085 test tokens correctly, with 34762/35709 and 34859/35830 exact blocks.
These include real EOS labels, excluding EOS fill. Memory-source agreement is
952/1049 dev and 941/1038 test: aggregate near-perfect reconstruction does not
justify claiming perfect per-source recovery. This report has not classified
those errors into EOS versus answer-content errors; inspect before selecting
an exact-output memory benchmark.

No bounded full-answer training schedule or winning architecture is selected by
these preparation results. Sampling, complete self-fed generation and resumable
matched training still need implementation and verification before launching.
