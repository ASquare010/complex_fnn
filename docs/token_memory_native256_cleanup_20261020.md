# Native256 canonical cleanup — 20 October 2026

User-directed architecture migration: preserve the existing low-bit format, make
src/models/token_memory the only compressed-memory implementation, remove the
256-to512 input projection, keep23,209,216 reader parameters and two original
epochs, provide readable progress/periodic evaluation and a training command.
This explicitly supersedes historical source-freeze instructions for active
source paths. No old checkpoint, encoder, corpus, label table or cache was edited.
No production training was started.

## Architecture and comparison boundary

The new model is256-wide end to end, eight blocks/four heads, tied256-word
embeddings, total FFN hidden width38,062, read blocks2/5, raw64 and one memory
position/64 old tokens. Small RMS normalization/channel scales/type offsets
match input amplitude without a dense projection. The source-word metadata is
extracted first; the existing FP32 format and read-layer V reinsertion are kept.
Local token-candidate work remains. Budget belongs to the reader; encoder is
separate. This is fresh seed47, not an edited wide512 checkpoint.

Original TRAIN assistant targets, per-epoch orders47/48, effective32,48,762updates,
361,357,104 label exposures, frozen FP32 cache, query .05/source .02 auxiliaries,
AdamW3e-4 to3e-5, warmup200, decay .01/clip1 and FP32 are preserved. Trainer uses
length-first microbatch4; full original DEV every1,000 updates and final; saving
250. Exact resume preserves weights, Adam state, RNG, cursor and sampler arrays;
source/input/format/recipe guards reject stale checkpoints.

Step3A establishes compact global positions/KV and a working route for some
memory reading. Exact literal-ID storage succeeds; perfect factual retrieval is
NOT established. The completed512-wide study had67/285 exact familiar answers
versus27/285 raw1, but language62.322818% versus63.709511%, poor transfer and zero
reserved exact answers including known controls. Total KV at2,048old tokens was
3.094MiB versus72.003MiB (23.27x); request latency was0.10–3.68% slower. Full detail
in [completed comparison](wide512_completed_comparison_20261020.md). These are
historical results for wide512, not native256 measurements or a vanilla baseline.

The optional optimization discussed with the user is not implemented: a fixed
4096-token by8-score inference lookup table is128KiB. Its entries depend only on
embedding/selector weights, so could be calculated once per loaded model and
refreshed after weight changes. It does not grow with conversation/KV. A speed
benefit requires output-parity checks and measurement.

## Source cleanup and retention

Before edits,576 source/test/doc/config files were zipped and every archived
member checked against its SHA256. Local archive (ignored by Git):
`dump/token-memory-cleanup-20261020/sources-before-cleanup.zip` (91,654,623bytes).
SHA256: `49d49423e81ac3718f57e70f51476e918e94b408147eadcae1890c7ca826465a`.
`manifest.json` records per-file hashes. The completed wide512 run also retains
its original sources.zip, paired results, verification receipts and checkpoints.
Restoring an old study must use its archived source in a separate workspace;
its source guards intentionally do not accept this canonical migration.

Removed16 src/models/token_memory_* variant directories after checking resolved
paths and every retained source hash against the archive. Removed256 named
native256/wide512/token_memory64 experiment helper/test files and22 obsolete
legacy-memory commands/tests. Detailed absolute removal lists are recorded in
verified-model-removals.json, verified-variant-helpers.json and
verified-legacy-memory-helpers.json beside the archive. Experimental .md records
remain under docs. Original raw baseline and frozen codec remain, with raw data
and attention utilities independent of the retired memory implementation.

Automatic approval review rejected the initial broader cleanup list because it
included generic utilities. The completed cleanup was narrowed to verified
memory variants and their dedicated helpers; corpus and codec tools were kept.
Old chat entry points route to the new canonical CLI. Unrelated pre-existing
edits are excluded from the commit.

## Verification

CPU checks cover exact parameter count/fresh initialization, native256/no adapter,
causal packed versus isolated outputs/all parameter gradients, future/padding
isolation, cached/public/chunked decoding, empty/blocked memory, unchanged KV
schema, exact payload and64-token rollover; generic raw1 coordinate control;
CE/auxiliary denominator/detachment correctness; serialized Adam replay; actual
CLI pause/resume versus uninterrupted tiny execution and readable periodic eval.
Tiny synthetic updates are test fixtures, not a production experiment.

GPU zero-update capacity check PASS on four actual original training examples
at512 raw positions/39 memory positions: peak allocation2.684GiB, cold
forward/backward1.858s. No optimizer state/parameter changes. This is a capacity
check, not steady training throughput, request latency or a quality result.
Original corpus/cache/label hash verification PASS on all TRAIN/DEV assets.
Both complete sampler arrays match the original wide512 orders exactly.
Final retained-repository test suite:28 passed,2 skipped (including11 new
canonical model/objective/trainer/generation checks). No production updates.

Training command and resume instructions:
[src/models/token_memory/README.md](../src/models/token_memory/README.md).
