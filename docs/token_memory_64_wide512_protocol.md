# Width512 backbone / width256 local reader protocol

18 October2026. New separate package `src/models/token_memory_64_wide512`.
All original/native256/growing-retirement/checkpoint/corpus/cache files remain
unchanged. No existing completed pilot is extended. Goal remains ACTIVE/UNMET.

The hypothesis is that the width256/21-layer reader is an inefficient allocation
for the broad language target. This candidate restores a width512/eight-layer
backbone close to the strong original reference while retaining256-wide local
dictionary work. It does not assert that projection caused earlier failures.
The existing old256 encoder stays frozen and separate. Fresh seed47 is required.

Reader budget is exactly23,209,216:22,312,448 core,764,416 local/query reader,
132,352 memory norm/projection/gain/type adapter. Core has eight heads, two local
reads after layers2 and5, and total FFN hidden width11,525. Raw generation still
has BOS plus at most63 actual tokens, with64 actual tokens compressed on rollover.
One global memory/KV position represents64 old words. No decoder expansion or
additional source-history cache is allowed. Original global RoPE coordinates
are retained to avoid adding a position change to this architecture test.

Old codes remain256 at input; payload IDs are read before normalization and
projection. Existing exact word-ID metadata is disclosed: it is an added literal
payload, not proof of neural lossless compression or byte compression versus IDs.
At the two read layers, the payload is reinserted into the first256 channels of
the existing512-wide V slot. The remaining256 V channels are unchanged. The local
dictionary uses the first256 channels of the shared512 token embedding, while
query/output states remain512. Temporary word scoring still costs work.
Token/role metadata has the inherited ASCII/eight-BPE-piece limitation. Fixed role
tables are model buffers, not history or KV entries. No new STOP action/loss or
conditional EOS override is included: those diagnostics did not preserve quality.

Adapter gain starts at1/sqrt(256), so a normalized old code projected with .02
weights has roughly the fresh token embedding's RMS scale. Measure actual cached
input statistics in GPU preflight; this initialization is not a quality claim.

The raw1 reference uses the SAME parameters/initial tensors and one global
position per old word. Its source state includes the full512 token embedding
as a residual plus the shared adapter applied to its first256 channels; every
adapter parameter is used. It can retain the full word representation and is
not a compressed64 surrogate. Query/source coordinates use words_per_memory
explicitly:64 for compressed,1 for raw1. Both modes keep raw64 generation.
The reference uses additional global attention/KV by design; it is a control.

Before training, require CPU tests of exact budgets/initial tensors, independent
word geometry and raw1/64 query-coordinate parity, packed versus individually
unfolded public predictions/all-parameter gradients, future/padding isolation,
public/cached/chunked/rollover equality, exact retained payloads and unchanged
cache attributes. Actual mixed-objective normalization, microbatching and exact
serialized Adam/RNG/cursor resume must also be tested. Then require full-size
GPU all-parameter gradient/capacity/speed gates with no optimizer updates and
unchanged initial weights. Compare to the preserved V2 reader at realistic
shapes, include encoder/optimizer/I/O exclusions in component timing claims.

Eventual training must use a NEW run/format and both fresh47 arms, original
780,186 segments/epoch,180,678,552 original assistant labels/epoch,48,762 updates
over two original epochs, effective batch32, FP32 and original LR/AdamW schedule.
Existing query/source auxiliary labels derive solely from original TRAIN data;
both arms receive the same examples/auxiliary targets and global denominators.
No new QA text, reserved TEST exposure or synthetic reading labels are permitted.
Any within-update layout optimization needs full GPU proof and explicit recipe/
actual microbatch provenance; it must not cross original update boundaries.
Prepare source/data/evaluation/verification closure before starting the run.

Final assessment requires full original dev, full two-epoch original exposure
audit, strong matched raw1 reference, original512 reference, natural full replies,
changed values/queries, four-way binding, missing/blocked/shuffled controls,
historical and reasoning probes. Reserved student TEST is scored only after the
full exposure gate. Lower teacher loss or a passing code test is not success.
