# Selective trained decoder: zero-update architecture preflight

The broad trained-readout gate was progress, not parity: the small student
recovers only a fraction of words, while the ORIGINAL decoder evaluated on one
selected feature recovers almost every non-special test token. Inspection shows
the codec decoder has only ONE block. This new probe reuses that trained block
selectively instead of expanding64 positions into global LM attention.

Create a separate src/models/token_memory_64_selective package; do not edit native,
old helpers, weights or caches. No subagents/growing families/new LM training.
The full23M/two-original-epoch/64:1 goal stays active and unmet.

The readout selects125 rows of the trained256→8000 unpack matrix for one requested
position, applies the shared125→256 map, then the trained pointwise decoder block
and word head. With one position, attention has one key and weight1: fuse its
value/output matrices into one linear map and remove Q/K/softmax entirely.
Keep its trained RMS norms, FFN and final word head. This is a frozen nonlinear
position-specific reader, not the previously unsuccessful unpack-only head.

Also implement an UNTRAINED query-conditioned address scorer. Fold a query's
125-dimensional key through the frozen unpack weights, then score codes to obtain
64 SCALAR address scores per block, without constructing64 feature vectors per
code. Query-dependent offsets let a name match address a nearby value. Add an
internal NULL score and strict memory-valid/per-query-visible masks. These scores
are transient classifier outputs, not extra global attention positions or KV.
No retrieval success may be attributed to this untrained controller.

Provide a cache STORAGE prototype with one256-dimensional key and one256-value
per code. Values hold original FP32 codes for selective readout; do not keep a
second code, raw history, decoded features or token-ID cache. This replaces an
existing value slot; it is NOT compatible with silently reading a normally
projected/normalized V as the original code. Real LM integration must implement
that value route and prove prefill/incremental parity and full byte accounting.
The storage prototype alone is not such an integration or speed proof.

Build parameter-only native-shaped fixtures with reduced FFN rows to pay for
the frozen readout and controller. Count every stored/frozen readout weight.
Check both23,209,216-reader and23,209,216-TOTAL-including-encoder conventions;
the latter uses the actual9,426,432 encoder weights and a much smaller LM FFN.
These fixtures have no usable LM forward and establish arithmetic feasibility
only. No claim of quality at either allocation; budget preference remains pending.

CPU tests: selected-vs-original pointwise decoder logits, folded-vs-expanded
address algebra, offset/NULL/memory/future masking and gradients, actual cache
storage/readout roundtrip with no additional retained code, both parameter sums.

Then zero-update GPU/CPU audit on all512 dev and512 test blocks and all2048 prior
fact cases, with oracle positions derived from actual source text. Save full
predictions; compare to the original one-position decoder and full decoder,
retain errors, report non-special and exact-body/word metrics. Correct, zero and
shuffled source controls on fact cases. Gold positions are deliberately supplied:
this is word recovery, NOT free answers, learned address selection, binding,
reasoning or language quality. All pretrained vocabulary resources are disclosed.

Benchmark the complete selective readout and folded address+selected readout at
one query/memory counts1/39, batch1/16, versus full64 decoder. Exclude nothing
inside those components, but explicitly exclude the main LM, encoder, integrated
KV management and address SUCCESS. Verify frozen decoder/reader tensors and
protected old hashes. Independent recomputation through the original pointwise
decoder on deterministic samples and all-case metric/source audits must pass.

Next only after this gate: integrate a trained question-to-address controller,
language/copy objective and actual cache route in a separately budgeted model;
test held names/values/counterfactuals and language replay before final fresh
matched two-epoch runs. Oracle word accuracy alone cannot satisfy the goal.
