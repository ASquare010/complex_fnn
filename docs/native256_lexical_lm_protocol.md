# Lexical LM integration: zero-update verification before training

18 October 2026. New separate `token_memory_64_lexical_lm` implements the component
described in [lexical-reader protocol](native256_lexical_reader_protocol.md).
Old sources/runs/checkpoints/caches and native256 remain immutable. No growing
models, no subagents. The name-gradient-isolation pilot is a separate experiment.

The model has21 layers/width256/4heads, FFN total31374 and559616 lexical-reader
weights, exactly23209216 parameters, all trainable. Encoder is separate/frozen.
Two read units after layers7/14 derive lexical features transiently from exact
IDs in their EXISTING projected memory V values. The custom bitwise forward with
identity straight-through semantic gradient reinserts metadata AFTER projection.
The cache has exactly the old K/V/valid/current-raw64 arrays, with no retained
word IDs, source feature bank or extra memory positions. The compressed branch
uses64 words/value. A genuine1:1 reference mode uses one embedding/global K/V
position per source word with the SAME readout and initial weights. Its larger
cache/attention cost is explicit. No old raw-history sequence is given to the
compressed main transformer. Exact IDs are newly added lexical metadata, not
evidence of recovery by the original learned encoder.

Main packed causality retains the original raw64 windows. Old source keys only
see earlier source positions. Local word contexts use causal predecessors only;
every candidate is masked by the query's visible physical memory count/validity.
The generic soft pointer copies vocabulary IDs and returns NULL mass to the
generator; no gold name/value/span/field or fixed answer list enters inference.
Rollover packs the64 actual generated words, excluding BOS, into one newly
encoded code (or64 reference word slots), clears only the old raw window and feeds
BOS. All source metadata must come from observed tokens. Unique word-bit cache
format is mandatory; original FP32 code caches cannot be passed as already-packed
word-bit values. They may only be read to create separate tagged new caches.

CPU gates cover actual full-size budget/fresh64-vs1 tensors, activated (nonzero)
read/recency gates, public/cached/prefill/token output parity, metadata recovery
from actual existing V allocations, exact cache array inventory/bytes, future/raw
padding/all-parameter gradients and independent isolated causal queries,
empty/blocked memory and actual64-token rollover in both modes, microbatch
gradients and actual8-step serialized Adam resume. Fresh zero gates alone are
insufficient evidence for integration correctness.

Before ANY lexical LM training: full-size GPU tests with fresh shared tensors,
activated temporary gates for parity (restore fresh weights afterwards),
genuine-source compressed and1:1 inputs, independently constructed memory IDs,
public/packed/cached checks including rollover/empty memory, all-parameter
microgradients, exactly unchanged initial state and zero optimizer updates.
Independently inspect source/weights/masks/outputs/cache state. Benchmark complete
forward, source prefill, cached token decode and64-token rollover including frozen
encoder against matched native256 and1:1 controls. Use same histories, query
tokens and validity; alternate timing order, report samples, peak memory, exact
cache bytes and all exclusions. Do not infer speed from lower source position
counts or a component matmul. No training until these checks pass.

The full-size gate uses one genuine2560-token training document (39 completed
blocks plus64 current words), independently checked against the immutable full
FP32 code cache and re-encoded with frozen E. Both1-block and39-block histories
are contiguous suffixes of that same document. Batch1/4 timing uses identical
history/query tokens, two warmups/five alternating-order saved samples. Models
are original native64, lexical64, lexical1 and a conventional native1 control,
all23209216 parameters. The native benchmark-only cache adapter must match the
original Native tensors/public math; its1:1 mode supplies ordinary word embeddings
without lexical parameters or word-bit V modifications. Native1 input transport
parses literal IDs; no old ID bank remains on its cache. Lexical64/1 parity uses
temporarily activated read/recency gates, restored to exact fresh weights before
timing. Cached prefill includes source plus63 current inputs; decode consumes the
64th input. Rollover includes the actual frozen encoder for64 modes and skips it
for1:1 modes. All caches reserve one more completed64-word block and raw64.
Heavy lexical training-body profile uses four copies of a real original512-token
segment and its original assistant labels, with two Adam-moment-sized buffers,
no optimizer update. Fullsize CPU independent replay, literal bit extraction,
all-parameter CPU/GPU gradients, cache byte/phase counts and all timing medians
are rederived. These loaded-input timings are not full-training/quality evidence.

Subsequent new pilot/data/trainer/evaluator are not implemented here. They need
fresh source manifests and broad held-out names/values/multiword/position/binding/
missing/corrections/multiblock and language/reasoning checks. Completed narrow
name pilots must not be extended. Final acceptance remains the full user goal:
fresh two-original-epoch/23M-reader/64:1 matched comparison with a strong genuine
uncompressed baseline, broad quality and complete measured efficiency.
