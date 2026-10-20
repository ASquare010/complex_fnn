# Exact sparse-vocabulary/vectorized lexical kernel

Prospective18October2026. The first full GPU gate verifies the23M lexical reader
but finds added decode overhead: approximately51ms versus37.5ms conventional
native1 at2,496 old words. The cache is4.47MB versus112.86MB, and batch4 forward
is51.2ms versus133.4ms. These are untrained loaded-input measurements, not quality
results. Original/reference lexical sources and saved exports remain immutable.

Implement a separate `token_memory_64_lexical_fast` sibling with EXACT same
parameters, state-dict names, fresh seed47 weights, equations, word-bit input
format and existing KV arrays. No new trainable weights or persistent buffers.
The only changes are execution: normalize/gather the vocabulary rows actually
present in the old IDs, fold nine query-context projections into one batched
vocabulary GEMM, gather/sum nine scalar predecessor scores together, and form
read vectors by mass over those unique vocabulary rows. Full output logits still
cover all4096 IDs. Every old word remains a candidate and each candidate/neighbor
retains original causal masks. No approximate top-k or discarded words.

Unique rows and neighbor-index arrays are TEMPORARY, rebuilt from existing V
metadata each call; no source ID/feature bank may persist on the cache. There is
no normalization across vocabulary rows, so unused/future IDs cannot change
the mathematical visible-row scores. Different GEMM/reduction shapes may cause
small FP32 roundoff; test future derivatives and all-parameter parity, not only
one forward. Source-table uniqueness costs extra sorting/synchronization work and
must be measured. Do not assume performance from fewer multiply-adds.

Before training: CPU exact state/budget, independent reference reader/LM forward
and all-parameter gradients with activated read gates, metadata and cache array
parity, future-source/raw-key isolation, actual64-word rollover, microbatch and
serialized Adam resume. Then full-size GPU reference parity, independently audited
genuine-source inputs/gradients/cache, and identical complete timing phases against
the frozen first kernel, native64, lexical/native1. Preserve the original export
and source snapshots; never edit live/guarded reference files or resume old models.
No model is trained until those gates pass. Final broad quality/fresh two-original-
epoch/23M-reader/64:1 objective remains active and unmet.
