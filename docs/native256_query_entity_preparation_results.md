# Query-identity reader: implementation and zero-update evidence

18 October2026. The previous visible turn reported a paused completed experiment
and made no new experimental progress. This resumed turn made progress: a new
reader, independently audited original-data labels, actual all-case production
input coverage, objective tests, full-size CPU/GPU and capacity checks and an
equal-budget speed comparison. The full goal remains ACTIVE / UNMET.

The new implementation is `src/models/token_memory_64_query_entity_v2`. It adds
trainable entity/topic selection from causal recent surface IDs and biases the
existing local reader toward old sentences containing the selected words. It
preserves per-unit ordinary pointer EOS/NULL mass and starts presence vectors
at zero. Query-role and word geometry use public input bytes; gold query/source
coordinates are training targets only. The encoder remains separate/frozen.

| Constraint | Verified implementation |
|---|---:|
| Trainable reader parameters |23,209,216|
| FFN hidden units removed to fund query path |6 (3,072 weights)|
| Global OLD attention/KV positions per64 BPE tokens |1|
| LM active raw window |64|
| Additional persistent KV bytes |0|
| Additional fixed model-buffer bytes |16,384|
| New corpus optimizer updates |0|

Literal IDs are added metadata in the existing FP32 word-bit values. This is not
proof that the old semantic encoder alone is lossless. The transient query reader
looks back128 literal IDs, requires a complete serialized User header, and keys
complete ASCII alphanumeric words of at most eight BPE pieces. This is a lexical
sentence cue with generic fallback, not an unrestricted relation/alias reader.

All39,270 ORIGINAL TRAIN questions produced independent query-slot labels. There
are62,822 critical positive source terms (one name/one value per31,411 positive
case), spanning31,413 original segments;7,859 missing cases have query labels
and NO generic-NULL source loss. Critical pieces are the first divergence against
the distractor records, before a differing gold prefix can identify the answer.
All bytes/coordinates/CSR tables were independently rederived. The actual
production-packet audit verifies both query candidates for every39,270 case and
the exact old entity key in every31,411 positive case. It checks all31,413 source
segments, including spillovers. Untrained selector hits18,084/39,270 entity and
17,136/39,270 topic are diagnostics, not output-quality results.

Prospective objective: unchanged global assistant CE +0.05 query-slot CE +0.02
critical-source CE, with global segment/field/read-layer normalization. Auxiliary
embeddings, incoming core state and old pointer/norm computations are detached;
only new selectors/gates receive auxiliary gradients. Presence/core weights use
ordinary CE. Nine repaired V2 invariant tests and four actual-objective tests
passed, including independent explicit query math/gradients, probability
conservation, causal/future/padding/answer exclusion, unchanged cache arrays/bytes,
cached/public/chunked/real64 rollover and serialized toy two-epoch Adam resumes.

V1's first auxiliary test exposed accidental registration of a temporary embedding
Parameter. V2 fixes only transient context assignment and preserves exact V1
initial tensors/RNG. V1 source/zero-update receipts and failed gate snapshots are
retained. Subsequent preflight input assumptions were corrected from actual
dataset evidence: no <=128 segment contains both query/source, and no source-
bearing raw512 segment has39 old blocks. No optimizer updates occurred in those
failed gates. A separate independent FP64 audit diagnosed a9e-6 relative CPU FP32
gradient-norm reduction discrepancy. Loss/raw/clipped gradients agree; norm is
now checked relatively, with all evidence bound to the new gate.

Full21-layer actual objective CPU/GPU clipped-gradient maximum error5.141e-7;
GPU allbatch/microbatch maximum1.863e-8. CPU/GPU objective difference4.65e-7.
All initial weights unchanged and Adam empty. Maximum-shape capacity uses a
genuine mixed original batch (one long-history language row, seven source rows)
at[8,39,256] memory/[8,512] raw: median body337.123ms,5.488GiB allocated/6.018GiB
reserved. An eight-source-row invalid-padding stress reaches the same shape:
median329.257ms,5.287GiB allocated/5.773GiB reserved. Its original15-block packet
and padded39-block packet objective sums differ at most0.0001221. Padding is
explicitly invalid; these are not eight genuinely39-block source histories.
Both checks include two moment-sized buffers, exclude Adam step/scratch/E/IO/
evaluation/save, and establish capacity rather than full training throughput.

The equal-budget speed comparison uses default23,209,216-weight ordinary Span
and query V2 on identical genuine inputs, alternating measurement order:

| Timed operation | Ordinary Span | Query V2 | Ratio |
|---|---:|---:|---:|
| Assistant CE forward/backward/clip, effective micro8 |202.301ms|248.793ms|1.230x|
| Cached one-token feed, batch1 |46.546ms|77.562ms|1.666x|
| Cached one-token feed, batch8 |46.788ms|77.293ms|1.652x|

All timed model weights remain unchanged. Cache bytes are identical3,441,232
at batch1 and27,529,856 at batch8 for that actual15-old-block query plus one
reserved code slot/raw64. These are body/one-token timings excluding initialization,
prefill/encoder/IO/Adam/evaluation/save and full-generation rollover. The new path
is materially slower. The preliminary219->256ms shared-core measurement used a
slightly smaller ordinary control and is superseded for equal-budget reporting.

No corpus training, free-answer comparison, full DEV improvement, final TEST or
novel-literal student scoring has occurred for V2. Correct implementation and
input coverage are not evidence that query selection/generalization will learn
or that the compute cost will be justified. Next: a separately guarded fresh47
matched pilot under the original48,762-update/two-original-epoch schedule, with
full free facts/all-four/missing/language checks, then a verified final full-budget
comparison only if supported. Native256 and all completed earlier runs are retained.

Receipts:

- `dump/native256-query-entity-labels-20261018/independent-verification.json`
- `dump/native256-query-entity-surface-audit-20261018/report.json`
- `dump/native256-query-entity-norm-audit-20261018/report.json`
- `dump/native256-query-entity-v2-preflight-capacity-verified-20261018/report.json`
- `dump/native256-query-entity-v2-speed-20261018/report.json`
- `dump/native256-query-entity-preparation-verification-20261018/report.json`

The final combined audit PASSED: all2,323 previous protected files, including
native256, are unchanged; source/data/gate/speed receipts were rehashed. All
workers are terminal. Sources and completed receipts are
immutable. Do not resume any earlier failed/completed run or score reserved TEST
inputs with this untrained prototype. The new corpus export/trainer is not created
yet. Full parity, efficiency and the two-original-epoch final comparison remain
outstanding.
