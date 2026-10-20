# Wide512 implementation and zero-update preflight

18 October2026. New separate package src/models/token_memory_64_wide512 is
implemented. Exact23,209,216 reader parameters; frozen256 encoder separate;
width512/eight layers/eight heads,256 local dictionary features and two local
reads. Raw64 generation and one global position per64 old tokens are retained.
Payload IDs are extracted before projection and reinserted into existing512-wide
V slots; no extra source-history bank. A genuine raw1 control uses the same
parameters/initial tensors and full512 word-embedding residual, one position/word.
The inherited literal token-ID metadata and ASCII/eight-piece limitations remain.
No STOP override/extra STOP loss is included. Old/native runs remain unchanged.

All eight frozen CPU gates and full-size GPU gates passed. Independent packed
versus public predictions/all-parameter gradients, future/padding isolation,
chunked/empty/blocked caching, exact payloads/rollover, raw1 word-coordinate parity,
actual mixed-objective normalization and exact serialized Adam resume are tested.
A new empty-memory reshape bug was fixed before the frozen ledger/no training.
Both complete source and2,748 protected original input checks passed.

Full-size CPU/GPU gradient maximum compressed7.00e-7, raw1 8.46e-7; GPU microbatch
maxima2.24e-8 and3.91e-8. Initial weights unchanged and Adam empty in both arms.
GPU public/cached log-probability max difference1.91e-6. Genuine prefixes are
8/8 codes for compressed and512/512 word positions for raw1.

Six alternating fresh-model cached-token samples on the same genuine batch2
prefix give prior V2 median79.449ms, wide512 34.003ms (2.34x). Allocated KV/cache
bytes6,366,356 to4,850,836,23.80% less; attribute sets unchanged. This excludes
cache setup, prefill, rollover and encoder; it is not a trained-quality or
end-to-end latency comparison. Extra fixed role tables are model buffers.

Measured incoming encoded-vector RMS7.681925 becomes memory-state RMS.019990,
versus raw-embedding RMS.019978. This documents the normalization/gain treatment
being tested; scale mismatch alone is not established as the causal failure.
Raw1 memory-state RMS.028062 includes the full word-embedding residual.

Maximum-label genuine stress case uses512 assistant labels/example and the full
old prefix: compressed micro12 peak6,944,930,304 allocated/7,899,971,584 reserved
bytes; raw1 micro4 peak7,138,488,320/7,411,335,168. Moment/scratch-sized buffers
are included; optimizer states/steps are not created. Earlier mixed stress also
passed compressed8/12 and raw1 1/2. No production optimizer updates occurred.

An additional actual original effective32 update preserves all7,035 LM labels,
3 query and3 source segments across layouts. All parameter gradient differences
are at most2.24e-8. Three-sample body medians: compressed original micro8 .959s,
length-sort micro8 .715s; raw1 original micro1 3.263s, memory-sort micro4 1.820s.
These are one-batch forward/backward/clip timings excluding Adam, encoder, I/O,
eval and saves. They are not full-training throughput claims. The earlier full
2-epoch CPU layout analysis checks update multisets/label totals, not wall time.

Candidate training plan: compressed length-first/micro8; raw1 memory-first/micro4,
with original effective32 update boundaries, sample multisets, labels and global
loss denominators preserved and actual micro-orders logged/audited. A new full
trainer/data/initial-state/evaluation/source closure is still required BEFORE
production training. Full language evaluation must use the same copy distribution
as training in BOTH arms; the old generic evaluator's raw1 gen-only branch is
inapplicable to this new matched control and must not be reused silently.

Artifacts: dump/token-memory-64-wide512-preflight-20261018/{cpu,gpu}/report.json
and dump/token-memory-64-wide512-layout-preflight-20261018/report.json.
No quality result yet. Full fresh two-original-epoch comparison, strong matched
raw1 reference, original512 reference, factual/counterfactual/missing/blocked/
shuffled/historical/reasoning and gated reserved TEST checks remain outstanding.
