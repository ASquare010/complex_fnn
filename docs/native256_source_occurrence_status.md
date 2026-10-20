# Source-occurrence training test

18 October 2026. Goal remains ACTIVE / UNMET. This is a new training objective
on the completed pooled-key architecture. The completed ordinary-CE pilot is
reused as the control; its checkpoint is not continued or changed. Native256,
the original models/corpus/caches and the frozen encoder remain untouched.

The proposed difference is supervision of the correct source occurrence instead
of only the summed probability of its word ID. Existing name/value answer pieces
point to that person's original statement. Missing facts point to the existing
NULL choice. Source coordinates are training-only targets, never model inputs.

| Audited original TRAIN supervision | Count |
|---|---:|
| Memory dialogues |39,270|
| Positive dialogues |31,411|
| Missing dialogues |7,859|
| Name/value source terms |415,989|
| Missing NULL terms |7,859|
| Original segments containing terms |39,294|
| Terms in matched first1,000 updates |16,834|
| Aligned segments in first1,000 updates |1,553|
| Original assistant labels in first1,000 updates |7,416,257|

Twenty-four answers cross original segment boundaries, explaining the difference
between dialogue and aligned-segment counts. All original segments and labels
remain unchanged; there are zero extra training examples or corpus epochs.

Preparation and the independent ALL-case audit passed. The auditor separately
parsed original questions, fact sentences and answers, checked every source and
target token ID/causal coordinate against original bytes, and reproduced the
entire sparse CSR table. No reconstruction or student-success filtering was used.

Five CPU tests pass: coefficient-zero exact ordinary Adam parity; independent
mean-embedding objective and all-parameter gradients; detached auxiliary core;
global microbatch normalization/future/padding/NULL controls; serialized two-toy-
epoch Adam resume; and genuine original answer/source bytes. Two additional
comparison tests reject fixed-record shortcuts, missing EOS and incomplete outputs.

Fresh initialization, RNG and orders match the completed control exactly, with
empty Adam and23,209,216 trainable reader parameters. The9,426,432-parameter encoder
is separate and frozen. Inference is the exact same pooled-key model, raw64 and
one global attention/KV position per64 old tokens. Literal IDs remain added
metadata in existing FP32 values, not proof that the old semantic encoder is
lossless. The auxiliary branch detaches incoming hidden states and embeddings;
ordinary CE still trains all weights. This does not guarantee language preservation.

Current run: `dump/native256-source-occurrence-unified-v2`.
Preserved zero-update export: `dump/native256-source-occurrence-unified-v1`.
Data: `dump/native256-source-occurrence-data-20261018`.
Protocol: [prospective design](native256_source_occurrence_protocol.md).

The v1 full data/initial and all production-packet audits passed. Its GPU gate
stopped at a bitwise coefficient-zero gradient assertion, before any update.
Eight repeat passes diagnosed ordinary CUDA gradient variation3.725290e-9 and
zero-auxiliary variation5.587935e-9, with exact loss/norm/weights/empty Adam.
The failed export and all sources remain unchanged. The corrected v2 export
records a1e-7 GPU gradient tolerance while retaining exact CPU Adam parity and
the full independent CPU/GPU/causal/capacity gates. See
[numerical protocol](native256_source_occurrence_cuda_protocol.md).

V2 preparation and data-proof transfer passed: all old input/cache/source hashes
and fresh state/orders were checked, the full immutable block audit was explicitly
inherited, and all39,294 production alignment packets were rechecked. The GPU
numerical gate passed with microbatch gradient error2.607703e-8 and unchanged
weights/empty Adam.

Inspection caught a capacity selector limitation: largest document end chose a
short tail, so that sample was not the maximum training shape. A separate
[capacity gate](native256_source_occurrence_capacity_protocol.md) now passed on
an actual mixed batch8/raw512/memory39 with119 alignment terms. Ordinary/alignment
allocated peaks are5,135,798,784/5,136,044,544 bytes, including two moment-sized
buffers. Three-sample body medians are0.22536/0.22914 seconds, excluding actual
Adam step/scratch/data IO/eval/save. These are not end-to-end speed evidence.

The independent full21-layer CPU gate passed: objective error3.33786e-7 and
all-parameter gradient error3.017485e-7, with unchanged weights and protected
earlier files. The finalizer bound BOTH numerical and maximum-shape receipts in
`preflight.json` before the first update.

Actual update1 and its independent audit passed:7,035 original assistant labels,
40 auxiliary terms from3 positive segments, all Adam counters1 and exact sampler/
LR/cursor/recipe/source checks. Initial language CE7.8881953954696655 exactly
matches the ordinary pooled-key control's first update. No missing example
appears in this first batch; the full prepared table retains all missing cases.

Training reached1,000 and paused, about1.25 updates/s including runtime overhead.
The independent actual-exposure audit passed7,416,257 assistant labels/32,000
segments, all Adam counters1,000, exact original orders/LR/cursor/recipe/sources
and protected earlier files. Auxiliary exposure is1,553 segments:1,243 positive
and310 missing,16,834 terms; global normalization and counts were checked.
This is not two original epochs done. No completed control was extended.

Full GPU evaluation and all independent checks are complete. The verified result is42.651722% token
accuracy/CE3.178110542, versus ordinary control44.164101%/CE3.062043972, a language
cost of about1.51238pp. Memory-source teacher accuracy is92.7987% versus95.1816%.
Positive complete facts are20/285 versus23, missing0/63 versus13. Notebook full
answers are15/256 for both, proper completions76/256 versus77; all-four0/64 in
both formats and both arms. Paired language difference−1.512379pp has fixed-doc
95% interval[-1.559751,−1.464757]. Both generation replay verifiers passed.

The separate conditional diagnosis and its cached replay also passed. First-name
LM accuracy81/285 falls to63/285; later teacher-prefix improvements are not
free binding. Copy gates are already high. Exact coefficient audit finds NULL
source loss weighted3.221x first-name loss, and NULL word-copy supervision
suppresses a valid acknowledgement "I" copy without recovering missing answers.
These are diagnostic observations, not causal sufficiency/convergence proof.

All workers are terminal. Preserve this completed checkpoint; do not adopt or
automatically extend it. Full outcomes and evidence are in
[verified results](native256_source_occurrence_results.md). The full goal remains
ACTIVE / UNMET; no fresh complete two-epoch/parity result is established.
