# Source-occurrence supervision, prospective protocol

18 October 2026, before label preparation or optimization. The completed
pooled-key1,000 pilot remains immutable and is the ordinary-word-CE control.
No native256/original source, run, checkpoint, corpus or cache is edited.

The narrow hypothesis is that the existing copy CE does not identify the right
record when many source occurrences share a token ID. Derive extra TRAINING-ONLY
source coordinates from the ORIGINAL training memory dialogues. Positive targets
are the queried name's and location phrase's BPE pieces, aligned to the same
pieces of its original fact statement. Include common words such as "in/the".
Every source ID must equal the actual existing assistant target ID and precede
the query's visible64-token boundary. Missing cases supervise the existing NULL
source at the first substantive answer token. No success/reconstruction filter,
extra dialogue, extra corpus pass or new input feature is permitted.

Save a sparse CSR table indexed by original segment. Each term records local
target coordinate, document source coordinate (-1 means NULL), and positive or
missing kind. Audit EVERY original memory dialogue and derived coordinate against
independently parsed questions, source sentences, answers and original token/mask
bytes. Preserve token-boundary artifacts; unequal source/answer IDs must fail
preparation rather than silently substituting a different source occurrence.

The new reader is exactly the completed pooled-key architecture:23,209,216
trainable reader parameters, separate frozen9,426,432-parameter encoder, raw64,
one global attention/KV position per64 old tokens, two read layers. The variant
still adds literal IDs in low FP32 mantissa bits; this is not evidence that the
old semantic codec is lossless. No inference operation or KV array is added.

Objective: unchanged global original assistant CE +0.1*alignment. At each read
layer, alignment is negative log probability of the correct SOURCE occurrence
(or NULL), averaged over selected pieces within each original segment. Average
these segment means across all aligned segments in the effective batch, then
average the two read layers. If there are no aligned segments, auxiliary loss is
zero. Count/log positive and missing segment means separately; they have equal
per-segment weight and are not artificially resampled. Normalize globally across
microbatches. This is extra derived supervision, not identical supervision to CE.

The alignment branch detaches its incoming transformer hidden states and shared
word embeddings, then recomputes only the small reader probability calculation.
It updates source/query normalization, query/key transforms and recency. Ordinary
CE still updates every parameter. This does not guarantee language preservation:
the changed reader remains part of generation. No target coordinate enters the
ordinary forward input, public inference or generation.

Fresh seed47, original orders/batches/assistant labels/LR/FP32/AdamW0.01/clip1 and
48,762-update/two-original-epoch schedule. First bounded pilot is1,000 updates,
matched to the COMPLETED pooled-key1,000 control; do not train another control or
continue its checkpoint. Initial tensors/RNG/empty Adam must match exactly. Pause
after actual update1 for independent label/order/optimizer/source audit, then
resume only this new run for999 new updates. Sources freeze before export.

Before optimization require coefficient0 exact ordinary-updater parity, an
independent explicit mean-embedding probability/objective/all-parameter gradient
reference, microbatch global normalization, detached-core checks, causal/future/
padding/NULL validation and actual serialized Adam/cursor resume. Full-size
zero-update GPU tests must check genuine source coordinates, gradient agreement,
unchanged initial weights/Adam and chosen-microbatch capacity. Measure auxiliary
body overhead without claiming end-to-end speed. No second GPU worker.

Evaluation uses the same full original dev, positive/missing/changedvalue/query/
blocked prompts, historical and seven legacy groups, numeric probe and controlled
notebook BOTH questions against BOTH swapped memories, including full and proper
known-name completion formats. Require independent aggregate/public CPU replay,
notebook replay and paired fixed-checkpoint comparison. Lower alignment loss,
single-answer gain or one-question memory-swap gain alone is not success. Robust
binding, unseen-value/generalization and language preservation remain required.

The final goal still requires full original2epochs and original test/novel literal
evaluation against genuine raw1 and the separately labeled strong mixed512
reference. This pilot neither completes that goal nor proves a capacity limit.
