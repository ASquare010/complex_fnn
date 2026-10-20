# Read-only shallow-reader budget inventory

18October2026. While the paired STOP pilot trains, a CPU-only module inventory
checked a possible efficiency alternative. No production model, forward, cache,
trainer or inference outputs were created; no ongoing sources/runs were changed.

An8-layer/8-head/512-wide native branch core, FFN hidden total10,980, semantic
memory256-to512 adapter and two512-wide local query readers plus STOP fit exactly
23,209,216 trainable reader parameters. Encoder is separate. Actual constructed
module counts (not a forward implementation):

| Part | Parameters |
|---|---:|
| Core |21,754,368|
| Memory RMSNorm |256|
| Memory projection |131,072|
| Memory type |512|
| Local reader/query/STOP modules |1,323,008|
| Total |23,209,216|

For equal39 old global positions and64 raw positions, K/V tensors alone would
occupy3,375,104 bytes per sample, versus4,429,824 for21 layers/width256. Ratio
`8*512/(21*256)=0.761905`; total cache auxiliaries are excluded. Global old-token
compression can remain64:1; the shape calculation adds no attention positions.

This falsifies the assumption that a narrower reader necessarily uses less KV:
layer count also matters. It does not predict actual latency or quality. Any
implementation must preserve exact literal metadata before projection/normalization
and reinsert it in existing V storage, prove causality/cache equivalence, measure
full-size memory/speed, and retain the frozen encoder/native run. A512-wide reader
requires its own fresh initialization/training; no old trained tensors were used.

The current STOP pilot continues unchanged. Broad factual/language/reasoning and
full fresh two-original-epoch parity remain the goal. Artifact:
`dump/native512-query-reader-budget-20261018/report.json` (all parameter shapes
and source hashes; explicitly inventory-only/no forward/no speed-quality claim).
