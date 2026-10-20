# Original test inputs, independent of pilot selection

18October2026, before preparation. The pilot uses original TRAIN and development
only. Prepare all ORIGINAL test prefix codes with the same separate frozen
9,426,432-parameter encoder, on CPU, into a NEW directory. No native/old cache,
student model, source corpus or current GPU training file is modified. No student
test prediction or metric is computed, and no decoder-success filtering occurs.

Input metadata/label hashes are checked for provenance; labels never train a
model here. Keep FP32 raw semantic codes, one per completed64-token block. The
student's normal literal-ID packing remains a later RAM-only input operation.
Cache identity binds original corpus/tokenizer/layout/preparation sources and
encoder SHA. Record exact encoder weight/buffer hash before/after, zero updates,
all source/input/cache hashes and source archive.

Independent CPU audit reads ORIGINAL test segments/token/mask bytes directly,
derives each contiguous document prefix without Dataset.block_tokens, verifies
all assistant label counts, and re-encodes every prefix block using groups127
rather than256. It checks cache values, all hashes and unchanged encoder state.
Later GPU agreement on representative source blocks is required before final
student scoring. This input-only preparation is not quality or losslessness
evidence and cannot justify an architecture decision from held-out scores.

Student original-test scoring is reserved for actual step48,762 checkpoints with
completed original two-epoch/361,357,104-label training audits. That final evaluator
and same-format novel-literal tests remain separate work; no pilot may silently
substitute development or a synthetic subset for the full requested test scope.
