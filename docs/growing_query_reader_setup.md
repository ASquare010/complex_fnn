> **Current status — 16 October 2026:** Growing/vector-generation experiments are retired. Their code, checkpoints and caches have been removed at the user's request. The text below is historical; use the [retirement record](retired_vector_experiments.md) for preserved results and the current token-memory64 direction.

# Query-conditioned growing memory: implementation and benchmark

Implemented2026-10-16. No corpus optimizer updates or new training run launched.
The active raw64 cross-attention process PID12572 continued; its26 guarded Python
sources were checked unchanged. The existing growing run was not resumed.

## Architecture

The full-corpus `growing_token_memory` trainer now imports `query_reader.Model`
through `full_model.Model`. Its historical `transformer.py` is preserved unchanged
as the original comparison implementation. New readers operate at layers4/8,
with width16 per attention head. Eight heads each have64 channels.

For a query q and visible value v_i, the contribution is
`v_i + B tanh(v_i[:16] + A q)`, weighted by ordinary rotary attention scores.
The query-conditioned operation applies to completed codes and the active prefix's
self-value, without supplying target tokens or other query snapshots. The current
query is unrotated inside the reader. The reader pools16-wide features before the
output projection, rather than projecting every query-memory pair to full width.

Readers add32,768 parameters. FFNs at layers4/8 shrink1534->1518 hidden units,
freeing exactly32,768. The total remains23,209,216. Other shared core tensors match
fresh47 exactly; resized FFNs and new matrices are deterministically initialized.
The encoder remains frozen; no reconstruction decoder is used. The data, labels,
cache layout, batch order, two-epoch schedule and global assistant-token CE stay
the same. This is a capacity reallocation, not an attention-only fixed-FFN ablation.

One completed code remains one attention position. Per-layer K/V shapes remain
`B,8,M,64`, both FP32. Across8 layers,39 codes occupy1,277,952 KV bytes, identical
to the original. The reader's channels come from existing V; no persistent reader
features are added. Training activations and computation do increase.

At the two reader layers, each query attends to M old codes plus its own prefix
key: `Q x (M+1)`. The other six layers retain the original fused masked attention.
Memory/query projections and FFNs are batched together during training. Completed
memory states are causal and independent of later queries, so generation keeps
the original cache format. The active prefix still re-encodes emitted tokens and
commits at64. This does not remove the runtime encoder cost.

## Verification

22 CPU tests pass across `test_growing_query_reader.py`, `test_growing_token_full.py`
and `test_growing_token_memory.py`. They cover an independently implemented dense
reference, all parameter/input gradients, empty/blocked/future/padded memory,
independent queries, full/cached gradient equality, unchanged cache sizes and
contents, code-specific query conditioning, chunk equivalence, actual-model live
rollover, exact parameter/shared-core identity, cache corruption/migration guards,
and actual tiny synthetic two-epoch Adam/model/RNG/cursor resume. Synthetic CPU
optimizer checks do not train the corpus model.

The full-width GPU real-dev check uses16 queries from segment7713, zero optimizer
updates, and a12% allocator cap. Maximum logit error against the dense reference
is5.9605e-7; all-parameter gradient max error6.7055e-8; full/cached logit error
8.5682e-7. Every new reader matrix has positive gradient norm. Parameters remain
unchanged. Report: `dump/growing-query-reader-gpu-parity-20261016.json`.
The diagnostic's first attempt stopped on a CPU tensor in its reference helper;
that helper was fixed to allocate on the input device. No model/optimizer changes
were made to resolve it. The rerun and all nine reader CPU tests passed.

## Timing

`tools/benchmark_growing_query_reader.py` reads existing verified caches only.
It verifies the original first32 seed47 samples (7,035 labels),34 selected train
prefix shards, and completed-code cache hashes. It tests two heavy real segments
and synthetic39-memory/512-query input, reserving two Adam moments and one
parameter-sized update scratch buffer. No optimizer is constructed or stepped.

Final measurements: RTX5070, FP32/TF32 disabled, micro1, effective32, three rounds
with alternating model order, shared with the user's active trainer.

| Model | Median32-example forward/backward | Range | Median cached query |
|---|---:|---:|---:|
| Original growing |3.467s|2.199–4.071s|37.43ms|
| New layout, reader disabled (timing ablation) |4.365s|3.444–4.385s|43.87ms|
| New query reader |3.408s|3.374–4.342s|46.41ms|

These numbers are noisy: the reader-disabled arm being slower than the enabled
arm prevents a clean incremental-overhead estimate. Do not claim a speedup or
infer useful tokens/sec. Training timing includes cached-tensor upload and
forward/backward, excludes disk loading, encoder, optimizer, validation and saves.
Cached-query timing repeatedly predicts one fixed query and excludes live prefix
encoding/rollover. New peak allocation746.86MiB, reservation864MiB, includes the
Adam-sized reserve buffers and heavy cases. Not a capacity claim for micro4.

The initial implementation separated all completed-memory and query layers,
producing an8.498s median versus3.311s original. That implementation was replaced
after the benchmark; its report is retained as diagnostic history. The final
implementation batches shared projections/FFNs, retains ordinary fused attention
at six layers, and vectorizes reader features across the original segment.

Final report: `dump/growing-query-reader-benchmark-optimized-20261016.json`.
All model states, selected cache files and active-run source hashes were verified
unchanged by these measurements. No claim about improved loss or fact recovery
is made before training.

### Idle-GPU rerun after the hybrid completed

The completed hybrid was verified at48,762 updates; no training process was active.
The same benchmark ran under the shared worker lock, microbatch4/effective32,
three alternating rounds, FP32/TF32 disabled and a20% allocator cap. First32
samples and7,035 labels match the earlier benchmark. All source/cache guards pass.

| Model | Median32-example forward/backward | Range | Median cached query |
|---|---:|---:|---:|
| Original growing |0.5057s|0.4501-0.6106s|18.36ms|
| New layout, reader disabled |0.6272s|0.4776-0.7463s|26.40ms|
| New query reader |0.7267s|0.7114-0.7416s|28.62ms|

The reader took about44% more forward/backward time than the original in this
workload, and about16% more than the disabled-reader timing arm. Timings still
have run variance and cover one effective batch, not the corpus. All earlier
exclusions apply: no disk loading, encoder, Adam, validation, checkpoint or live
prefix-encoding cost. No end-to-end speed or quality claim follows from these
numbers. Persistent KV remains exactly1,277,952 bytes for39 codes in all arms.

The benchmark's real microbatch4 workload peaked at1.376GiB allocated/1.600GiB
reserved. A separate maximum-layout capacity check used four examples each with
39 codes and512 fully supervised queries, reserving two Adam moments plus update
scratch. It passed at1.340GiB allocated/1.547GiB reserved, all gradients finite,
unchanged weights and sources, zero optimizer updates. The separate check is not
a universal allocator bound; validation and optimizer workspace can vary.

Reports: `dump/growing-query-reader-idle-micro4-20261016.json` and
`dump/growing-query-reader-micro4-capacity-20261016.json`. Capacity tool:
`tools/check_growing_query_reader_capacity.py`. The model implementation did not
change during these measurements. No new training directory was created.

## Training

The full command in the model README starts fresh47 with microbatch4, effective32,
and the original48,762-update two-epoch LR schedule. Add `--stop-after 4000` only
for the bounded pilot. Matching the recipe does not promise matching quality. The
worker lock remains enforced. Old checkpoint format `growing-token-memory-full-v1`
is rejected; new format is `growing-token-memory-query-reader-v1`. New run outputs
must use a new directory. The original growing sources are also archived in
`dump/growing-query-reader-original-sources-20261016.zip`.
