# Pooled source keys:1,000 verified, no robust binding win

Latest18October2026: corrected original-corpus v2 pilot completed and independently
training/result-audited1,000 updates/7,416,257 original labels. All Adam/order/LR/
cursor, aggregate/source/public CPU/notebook checks pass.60 uncached public CPU
and32 notebook replays match GPU, with blocked-swap invariance. Initial state is
fresh47, not a continuation from the old trained reader. Matched comparisons to
both controls are saved; all original exposure/prompt/source guards pass. See
[verified raw/old-reader comparison](native256_lexical_control_1000_results.md).

| Same1,000-update outcome | Old nine-token keys | New pooled33-token keys |
|---|---:|---:|
| Full-dev token accuracy |44.136780%|44.164101%|
| Full-dev CE |3.065599|3.062044|
| Positive full factual replies |21/285|23/285|
| Missing-information replies |0/63|13/63|
| Changed-value both-correct pairs |1/64|2/64|
| Changed-query both-correct pairs |0/64|0/64|
| Controlled full replies |22/256|15/256|
| Controlled known-name completions |78/256|77/256|
| Known-name both-memory swap pairs |1/128|11/128|
| Full / completion all-four units |0/64 /0/64|0/64 /0/64|

Language difference+0.027321pp has fixed-checkpoint paired document95% interval
[-0.011717,0.064771]. Positive-fact difference+0.701754pp has paired source
interval[-2.807018,3.859649]. These do not establish a reliable language/positive
fact gain. Missing answers improve; changed query exact drops3/64 to1/64.
Known-name swap-pair difference+7.8125pp has unit interval[3.90625,12.5], but this
is not robust question binding: the model returns an identical value for both
queried people in121/128 same-memory comparisons(old120/128), and the same UNIQUE
source owner in107/128(old117/128). Both questions are correct in only1/128 same-
memory tests in either model. A fixed record choice can pass swaps for one name;
the two diagnostic CPU tests explicitly demonstrate that shortcut. All-four stays
zero, including raw-name/no-roll and whole-query-raw/no-roll strata.

All seven legacy groups remain0/512. Historical positives regress8/103 to3/103,
changedfacts3/32 to0/32; historical missing1/25 to8/25. Numeric0/53/EOS4(cap256)
is a limited probe, not general reasoning parity. Recent-color3/8 has zero EOS,
so it is not strict complete recovery. Blocked factual/controlled values remain
zero, with all64 matched blocked question/format swapped outputs identical.

Conclusion: the missing owner feature was real, but this pooling change does
not fix robust binding at this budget point. It costs about18% body time/703MiB
temporary peak, so do not adopt or automatically extend this arm. More context
alone is not established as sufficient. Full original two-epoch goal remains
unmet; these finite early outcomes do not prove a universal capacity limit.

Authoritative export `dump/native256-lexical-span-unified-v2/comparison-1000`
binds all fixed checkpoints/audits/prompts/sources and paired statistics.
`compressed64/evaluation-1000/notebook-binding/choice-analysis.json` contains the
descriptive record/value choices. No original/native/source/cache was modified.
The zero-update setup and cost evidence below is preserved as preparation history.

18October2026. New separate model `src/models/token_memory_64_lexical_span`.
The current lexical64 model fails robust notebook name/value binding, and an
input-only audit finds no owner-name token in any of570 discriminating nine-word
keys. Complete names need11..20 positions. This prototype keeps nine learned key
slots but pools causal ranges out to33 tokenizer-token positions. Older helper
names use "word" for a tokenizer token; these counts include BPE pieces and
special tokens, rather than natural-language words. Semantic/query states still exist
in the old model; this is a targeted feature hypothesis, not a proven sole cause.

Exactly23,209,216 trainable reader parameters, separate frozen9,426,432-parameter
encoder, identical fresh47 tensors/RNG/buffers, raw64 and two read layers remain.
Global attention still uses one position per64 old tokens, with no new cache
array. FP32 literal-ID metadata remains added information, rather than proof
the original semantic encoder is lossless. Pooling adds temporary computation.

## Verified checks

Six CPU model/reference tests pass, including exact full-size initialization,
independent actual embedding means/input/parameter gradients, cache/public and
actual64 rollover, future isolation, empty/blocked cases, microbatch gradients
and serialized8-update Adam/cursor resume across two toy epochs. An additional
genuine-input full21-layer CPU check gives exact loss and maximum gradient
difference3.576279e-6, with unchanged weights and zero optimizer updates.

GPU gate and independent CPU verifier both finished successfully. Full-size
GPU microbatch gradient error2.980232e-7, future-gradient error1.907349e-6,
future semantic-input gradients exactly zero. Independent explicit mean-embedding
CPU-to-GPU all-parameter gradient error4.529953e-6. Public/cached/real rollover
checks pass at1/39 blocks, including actual frozen encoder calls and metadata.
All2,113 protected earlier files remain unchanged.

## Cost on genuine loaded inputs

RTX5070,FP32,TF32 disabled; untrained matched initial models. Three warmups then
six measured rounds with rotated/reversed model order.2496 old tokens=39 codes.
Inference peaks include all three resident models; times include the vocabulary
output. No concurrent training or audit ran during these measurements.

| Long-history operation | Original lexical64 | Pooled-key64 | Conventional raw1 |
|---|---:|---:|---:|
| Batch1 full forward |54.241ms|55.095ms|63.353ms|
| Batch1 prefill |103.403ms|104.045ms|101.561ms|
| Batch1 next-token decode |52.836ms|55.867ms|45.151ms|
| Batch4 full forward |53.757ms|55.035ms|237.190ms|
| Batch4 prefill |101.755ms|103.874ms|253.093ms|
| Batch4 next-token decode |53.117ms|55.172ms|43.579ms|

Pooled keys retain a large batch4 loaded-forward advantage against raw1(about
4.3x in this run), but next-token decoding remains slower. Small median changes
are one-machine measurements without seed/hardware confidence intervals. Real
rollover includes the encoder for compressed modes and skips it for raw1;
batch4 medians are125.884/121.766/91.975ms respectively, with no general rollover
speed claim from that small difference between compressed modes.

Whole batch1 allocated KV/cache at this history, including raw64 and one extra
block reserve:4,473,448 bytes for BOTH lexical64 models;112,856,128 for raw1.
This is25.23x for the whole cache and64x for the OLD source positions. It is not
64x whole-cache or bitstream compression.

Isolated training bodies start at ZERO allocated GPU bytes and include two full
moment-sized buffers plus recreated library scratch. Only one model is resident.
They use one genuine repeated512-token/39-code example and three timed samples.

| Body | Original lexical64 | Pooled-key64 |
|---|---:|---:|
| Batch8 median loss/backward/clipping |236.476ms|279.020ms|
| Batch8 peak allocated |4,593,019,392 bytes|5,330,576,896 bytes|
| Batch8 peak reserved |5,207,228,416 bytes|6,310,330,368 bytes|

The measured batch8 body adds17.99% time and703.39MiB temporary peak allocation;
new allocated peak is4.9645GiB. Actual Adam step/scratch, encoder, loading,
evaluation and saving are excluded. These are NOT end-to-end updates/second.
Batch4 bodies also pass, but old samples vary from166.9 to319.8ms, so do not use
their three-sample median as a robust speed ratio.

## Preserved failed export and scratch diagnosis

The first span GPU gate intentionally stopped before isolated body profiling
because64MiB remained allocated after releasing models. A separate genuine
forward/backward diagnosis showed this was cuBLAS workspace: clearing only
that runtime scratch released all64MiB, with unchanged32MiB cuBLAS/1MiB
cuBLASLt settings and unchanged weights. The corrected v2 wrapper clears it
between bodies, retains the exact-zero assertion, and includes recreated
scratch in each body's peak. The failed export remains protected and is not
called a successful gate. No optimizer update occurred in any benchmark.

Evidence: `dump/native256-lexical-span-preflight-v2-20261018/report.json` and
`independent-verification.json`; first export `native256-lexical-span-preflight-
20261018/failure.json`; diagnosis `native256-lexical-span-workspace-20261018`;
full CPU proof `native256-lexical-span-cpu-preflight-20261018`.

## Fresh original-corpus pilot

The current entrypoint is `tools/run_native256_lexical_span_full_v2.py`, with run
`dump/native256-lexical-span-unified-v2`. Every saved initial tensor and RNG matches
the actual previous fresh run, and the orders file is byte-exact. The all-data
audit passed: all original rows, labels,4,987,062 codes and48,762 original
batch orders/label counts/LRs, with exact fresh initialization/empty Adam.
Three parent routing/fresh-state/provenance tests and two new
real-data/nested-restoration tests pass. V1's data audit caught a raw control-batch
restriction before training; its zero-update export/sources are preserved. V2
only allows that data comparison inside the original batch function, while its
model factory still rejects raw1 training. It keeps original data,
orders/assistant CE/FP32 AdamW/LR and the original48,762-update two-epoch schedule.
First target is1,000 for comparison to the frozen lexical64 and raw1,000 arms,
with actual first-update and full data/gradient/optimizer audits before resuming.
No span training has started. Current raw1 evaluation owns the GPU; native256 is
untouched. See [pilot protocol](native256_lexical_span_full_protocol.md) and
[data-routing correction](native256_lexical_span_full_v2_protocol.md).

No full two-original-epoch quality/adoption win exists for pooled keys. Any later result
must include free answers, missing information, counterfactual all-four binding,
held literal values, language and reasoning; full goal still needs two ORIGINAL
epochs and final original-test evidence.
