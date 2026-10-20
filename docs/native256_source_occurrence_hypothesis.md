# Possible next loss: identify the source occurrence, not only its word ID

18October2026. The training-only objective and labels are implemented, all labels
and full-model preflight gates passed, and the new bounded training is live after
an independently verified first update. Quality evaluation is still pending. See the
[prospective protocol](native256_source_occurrence_protocol.md) and
[execution status](native256_source_occurrence_status.md). The pooled-key1,000
control remains immutable: all-four0/64,121/128 question changes keep the same
value. The hypothesis below is not a demonstrated fix. No completed run is extended.

The copy head mathematically sums probability over EVERY occurrence of the same
token ID before next-token CE (`LexicalReader.distribution`, scatter_add). If
three records all contain " in the", correct word prediction does not identify
the queried record. After teacher forcing supplies the correct color, short
context may select later words without ever binding the original question to its
owner. Existing95% memory-source teacher accuracy/poor all-four free answers are
consistent with this ambiguity, but do not establish it as the sole cause.

If33-token owner keys remain weak, a focused training-only test is source-position
alignment from the EXISTING original memory metadata/history/answer. Supervise
the actual queried record's Name and value-token occurrences, including common
value words, rather than the sum over duplicate word IDs. Source coordinates must
map to genuine already-visible input tokens and exactly equal the original target
ID. Keep missing cases: supervise NULL at the first missing-answer decision.
Count all extra derived alignment labels explicitly; same examples/epochs does
not mean identical supervision to plain CE.

Preserve original LM CE, batches/orders/LR,23M/fresh47/two-original-epoch schedule,
frozen encoder,64:1 global positions and zero additional inference operations/KV.
The source-location target is TRAINING ONLY, never an input/gold feature in the
reader or generation. Precompute sparse row/target/source coordinates on CPU;
do not re-encode targets or filter by reconstruction/reader success. Audit every
metadata/query/response/source/token/causal coordinate independently.

An initial isolation can stop alignment gradients at the captured query hidden
states and shared word embedding matrix, recomputing only the small reader branch
with those inputs detached. Ordinary LM CE still updates all model weights.
This protects the core from DIRECT extra gradients but does not guarantee language
preservation: changed readouts/copy scores can still harm generation. Supervise
both existing read units, reuse actual source-valid/visibility masks, and keep
global normalization across microbatches. Missing/positive row means should be
reported explicitly, rather than letting long answers silently dominate NULL.

First test coefficient0 exact parity with current updater, independent objective/
all-parameter microbatch gradients and actual serialized Adam/cursor resume. Then
genuine-source zero-update capacity/future/padding/null checks. A fresh bounded
comparison must use the same original exposures as pooled-key control. Require
free Name+value+EOS, correct both-swapped/all-four units, missing/no-match, held
values/names/wording and language replay performance. Lower aligned CE is not
itself retrieval. Earlier name-loss and interaction pilots harmed language or
failed generalization; their green training loss is not adoption evidence.

If no query-name signal exists in the detached LM state, reader-only alignment
may still fail. That result would motivate a separate direct causal query-token
summary, rather than silently mixing multiple architecture/loss changes here.
