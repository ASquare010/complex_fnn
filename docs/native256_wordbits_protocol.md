# FP32 word-bit residual: storage gate only

Prospective2026-10-18. New component `token_memory_64_wordbits`; no LM, encoder,
old model/cache/source is edited, extended or trained. No production reader yet.
The goal remains full23M-reader/two-original-epoch/64:1 quality parity, unmet.

The previous name-anchor curriculum is independently evaluated separately. Its
known-name gain and held-name/position limitations motivate CANONICAL lexical
keys without relearning the codec inverse. Test storing exact token metadata in
the existing256-FP32 vector, preserving all semantic dimensions. Four least
significant mantissa bits of four floats store one16-bit payload.256 floats store
64 payloads:12-bit vocabulary ID,2-bit INPUT-known speaker role,1-bit message-start
and1 reserved bit. Never encode future EOS/message-end/answer or gold entity spans.

Semantic values change by at most15 ULP (normal-value relative bound15*2^-23).
This is a precision tradeoff, not exact preservation of original floats. IDs are
exact. No extra vector/attention/KV position or stored ID bank. FP32 cache and
bitwise load/store are mandatory; normalization/casts destroy payload. Incoming
codes can carry metadata, and a future reader must reinsert it AFTER learned V
projection into existing V allocations. Semantic gradient uses an explicit
straight-through identity approximation; forward is entirely bitwise so zeros/
subnormals preserve payload. This is not byte compression versus raw ID streams.

Before any integration/training: all4096 IDs at all64 positions, all65536 payload
combinations, sign/exponent preservation, finite/zero/subnormal behavior, selected
word extraction, actual existing-cache copy/bytes, CPU/CUDA exactness and semantic
roundoff bound. Test selected frozen-codec word decoding before/after packing and
same-weight native hidden/logits on genuine cached inputs. Report numerical and
prediction differences, including failures; no success filtering. No quality or
speed claim from this storage proof. Unique format/new cache required; old code
formats/checkpoints must never be silently used as this format.

Future reader remains UNIMPLEMENTED: learned query-span/intent extraction, exact
full-span/boundary matching (not just first BPE), field selection, cross-block
masking, generic language readout and token-generation state. These need actual
23M allocation, unchanged KV count, full-model benchmark before training, matched
fresh controls and broad Name/value/position/reasoning/language checks. Word
roundtrip alone is not retrieval or goal completion.
