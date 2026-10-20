# Width512 backbone with width256 local reader: budget-only candidate

18 October2026. Actual CPU module construction totals23,209,216 parameters:
22,312,448 core;764,416 two-local-reader/query weights;132,352 memory adapter.
Core: eight layers, width512, eight heads, total FFN hidden width11,525. Local
source keys/values remain256, using256 channels of the reader token embedding;
queries/output states are512. Adapter: old frozen256 code RMSNorm,256->512
projection,512 channel gains and512 memory-type values. Encoder remains separate.
This retains one global memory position per64 old tokens. For103 total positions,
KV tensors alone calculate3,375,104 bytes versus4,429,824 in21-layer native256.

This is ONLY a parameter inventory, not a complete model. No forward, cache,
causal/gradient proof, training, quality or latency benchmark exists. Do not
claim it is faster or better. It is a candidate for bringing the global backbone
closer to the strong original512-width reference while keeping local dictionary
work at256, rather than doubling that work. Projection is not established as
the root cause of the prior failure; the successful existing V2 literal reader
already demonstrates narrow retrieval without decoder expansion.

Implementation before training would need exact FP32 payload extraction BEFORE
normalization/projection, reinsertion in the first256 channels of the existing
512-wide read-layer V slots, eight-layer dynamic-width cache storage, unchanged
raw64 rollover and no additional history bank. Initialization must check semantic
memory/input-embedding scale explicitly. All existing model sources/checkpoints
must remain immutable; use a new package/run. Raw1 needs genuine one-position-
per-word input, common generic word-coordinate handling and an adequate reference,
not a weak early checkpoint. Full original two-epoch evaluation is still required.

Actual module inventory: dump/native512-local256-budget-20261018/report.json.
