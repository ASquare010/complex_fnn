# Integrated selective LM: zero-update verification

The new experimental package is
`src/models/token_memory_64_selective_lm`, separate from native256 and the frozen
selective component package. Six CPU tests, full-size GPU preflight and independent
CPU verification passed. **No optimizer updates or trained QA result at this gate.**
Run: `dump/native256-selective-lm-preflight-20261017`.

The reader has exactly23,209,216 stored parameters, including3,981,312 frozen
decoder parameters and36,864 trainable controller parameters. There are19,227,904
trainable parameters. The backbone has21 layers, width256 and24619 total FFN rows.
The encoder is separate9,426,432 parameters, consistent with the user's confirmed
baseline budget convention. This is fresh seed47 core/controller initialization;
the decoder is pretrained. Budget equality does not establish equal capacity.

The last layer's existing memory value slot contains the original FP32 code.
Incremental decoding reads that same allocation; it retains no extra code bank,
decoded history or expanded source-position features. Ordinary code attention
still sees one position per64 old tokens. The controller additionally computes64
temporary scalar address scores per code and NULL, then decodes one selected word.
This adds computation, although it does not add global attention or KV positions.

The output mixes the ordinary generation head with the frozen selected-word head.
The chosen address probability sets copy mass; other mass falls back to generation.
NULL uses generation. Gold addresses are absent from forward/cache inference.
Address CE can supervise question-to-source selection during training. Without
that training the selector is random; accurate oracle word decoding alone does
not establish factual reading.

## Verified behavior

Tests cover packed versus isolated-window all-parameter gradients, future and
padding isolation, fully blocked codes with finite gradients, public versus
incremental prefill/token outputs, frozen decoder isolation, hot-path equivalence,
actual cache bytes, original-code storage alias, exact serialized Adam resume and
rollover. Rollover encodes63 preceding actual tokens plus the last generated token,
excludes BOS, discards raw K/V and starts a new BOS window. It never retains all
source token history. Full-size microbatch gradient max error2.98e-7; weights
unchanged and zero optimizer updates. Independent CPU replay checks all64 saved
query outputs and incremental steps, parameter count, allocations and old hashes.
All1764 protected hashes remained unchanged.

## Synthetic FP32 timing

Medians in milliseconds, five samples after two warmups, same measurement harness:

| Batch / old codes | Selective raw64 forward | Native raw64 forward | Selective cached next-token step | Full64 rollover |
|---|---:|---:|---:|---:|
| 1 / 1 | 42.95 | 63.66 | 44.23 | 107.90 |
| 1 / 39 | 42.54 | 41.61 | 48.67 | 99.24 |
| 16 / 1 | 41.73 | 41.05 | 43.62 | 98.87 |
| 16 / 39 | 41.49 | 40.75 | 43.65 | 98.91 |

Rollover uses three samples after one warmup and includes the frozen encoder,
new compressed-code prefill and next BOS. Forward includes the complete LM/head/
controller/readout. Native has a different FFN allocation and no comparable
incremental API; its forward timing is not a matched cache speed control. The
first native timing is unusually high relative to other samples; no speed win
is claimed. These random-model synthetic measurements are not actual training
throughput or representative workload performance.

Heavy batch8/raw512/39codes with2048 labels: full loss/backward/clipping median
128.13ms over three samples. Peak3.344GiB allocated/3.631reserved includes two
Adam-moment-sized buffers. Excludes Adam update/scratch, data I/O, encoding,
validation and saving. This establishes feasibility before training, not a speed
or quality improvement. Raw samples and precise provenance are in the run report.

The next experiment is recorded separately in
[the question-to-address curriculum protocol](native256_address_curriculum_protocol.md).
Final fresh two-original-epoch quality, real all-raw reference, broad reasoning
and end-to-end efficiency remain unmet requirements.
