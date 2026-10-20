# Lexical64 GPU correctness, memory and speed: verified, no quality result

18 October2026. Both the reference kernel and its exact optimized implementation
passed full-size GPU gates and independent CPU audits. No lexical model has yet
completed training or demonstrated factual/language parity. The overall goal
remains **ACTIVE / UNMET**.

All four comparison models have23,209,216 reader parameters, FP32, seed47. The
encoder is separate. `native64` is the existing native architecture with original
semantic codes; `lexical64` adds exact-ID residuals and the generic local reader;
`lexical1` is the same lexical architecture with ordinary word slots;
`native1` is a conventional uncompressed native core without lexical parameters.
A benchmark-only cached native adapter matches original Native tensors, public
math and rollover; it does not modify any deployed model or checkpoint.

The source is one actual2560-token training document:39 complete old64-token
blocks and64 current words. Both short/long histories are contiguous suffixes.
Semantic cache vectors match a frozen-E re-encoding within1.43e-5; source tokens,
cache layout/bytes and masks were independently rederived. The benchmark is not
a quality evaluation or decoder-success-filtered sample.

## Full-size numerical evidence

| Check | Reference kernel | Optimized kernel |
|---|---:|---:|
| GPU cached versus packed maximum LP difference |1.91e-6|1.91e-6|
| GPU rollover maximum LP difference |9.54e-7|1.91e-6|
| All-parameter microbatch gradient difference |4.17e-7|3.28e-7|
| Future-code/raw-input gradient difference |2.44e-6|2.15e-6|
| Independent original-CPU versus GPU gradient difference |4.77e-6|5.72e-6|

Future continuous source gradients are zero. Empty/blocked source behavior stays
finite. Exact IDs are recovered from both actual existing read-layer V arrays.
Actual64-word rollover excludes BOS, encodes once for64 modes and never calls E
for1:1 modes. All initial weights are restored exactly; zero optimizer updates.
Independent CPU checks replay full-size public/cached/rollover cases and gradients,
recompute timing medians and byte counts, and verify protected sources.

The optimized kernel keeps the EXACT parameter/state/buffer layout and equations.
It projects only vocabulary rows present in memory (54 short,782 long in this
document), batches nine predecessor lookups and aggregates read vectors over
those unique rows. It drops no word candidates and retains no derived source bank
on the cache. Independent explicit equations and full gradients agree; GPU
optimized-versus-reference maxima are1.91e-6 LP and2.38e-6 gradients. Sorting and
temporary index/table work are included in complete timings.

## Complete loaded-input timings

RTX5070, one pinned CPU core, current PyTorch/FP32 environment. Two warmups and
five alternating-order measured rounds; medians in milliseconds. Source/current
tokens and cache capacity are matched. Prefill means old source plus63 current
inputs; decode consumes the64th input; rollover includes E where applicable.

| Batch/history | Phase | Conventional1:1 | Optimized64:1 | Reference64:1 kernel |
|---|---|---:|---:|---:|
|1 /2496 old words|Forward|37.44|44.77|52.03|
|1 /2496 old words|Prefill|74.03|86.18|93.54|
|1 /2496 old words|Decode|37.30|45.02|51.49|
|1 /2496 old words|Rollover|77.93|100.71|107.08|
|4 /2496 old words|Forward|131.56|45.33|52.11|
|4 /2496 old words|Prefill|146.67|87.61|94.30|
|4 /2496 old words|Decode|37.90|45.34|52.14|
|4 /2496 old words|Rollover|78.84|100.69|107.35|

Batch4 forward is about2.9x faster and prefill1.67x faster than conventional1:1.
Single-token decode is approximately20% slower in this setup, and rollover has
the frozen encoder cost. **There is no universal latency win or64x compute claim.**
These are warm loaded-input timings, excluding disk/tokenization and quality
evaluation. They do not establish complete training throughput or gains at every
batch/history length. All samples, not just medians, are retained in reports.

The compressed allocated cache is4,473,448 bytes versus112,856,128 bytes for1:1,
with2496 old words, raw64 and capacity for one additional completed block. This
is64x fewer OLD KV positions and about25.2x less WHOLE cache including the fixed
raw window/reserve. Cache arrays are unchanged; no extra ID/key/value bank is
retained. Temporary work still consumes additional activation memory.

## Training-body capacity

Four/eight copies of one genuine512-token segment with39 codes and355 assistant
labels/example. Optimized body medians: .13746s at micro4 and.146917s at micro8.
Recorded allocated peaks:2.57181GB/4.68634GB; reserved2.91504GB/5.34774GB. Two
Adam-moment-sized buffers are included, but Adam step/scratch, data IO, E,
validation/save are excluded. **An idle reference-model copy remains live through
the probe's loop/loss variables**, adding about92.84MB plus buffers; the probe's
short wording that heavy memory isolates the optimized model is imprecise.
The reference copy performs no timed training work. These samples do not isolate
a kernel-only throughput gain across different microbatch sizes.

The separate conventional raw training-capacity gate also passes with full
prefix gradients:2496 old words +512 current,355 labels, micro1 median.184386s,
3.15676GB allocated/3.27785GB reserved including two moment-sized buffers. No
prefix detachment, optimizer updates or weight changes. Micro2 must pass the
chosen full-run preflight before use.

## Artifacts and next step

Reference export `dump/native256-lexical-lm-preflight-20261018`: GPU3583 and
independent48797 completed successfully; all2053 protected earlier files intact.
Optimized export `dump/native256-lexical-fast-preflight-20261018`: GPU48389 and
independent26051 completed successfully; all2074 protected earlier files intact.
Raw capacity export `dump/native256-lexical-full-capacity-20261018`:39135 complete.
All these workers have terminated. Sources/exports are frozen.

The new full comparison is being prepared separately: exact original unified-v4
two epochs/48,762 updates/361,357,104 assistant labels, fresh23M readers, ordinary
CE only, optimized compressed64 versus conventional raw1. No extra QA training
data. It must still pass data/initial/actual-microbatch checks and first-update
audit, then complete broad factual, language, reasoning and independent final
evaluation. See [full comparison protocol](native256_lexical_full_protocol.md).
