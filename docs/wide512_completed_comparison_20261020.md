# Completed wide512 comparison — 20 October 2026

The result is mixed: compressed64 improves familiar notebook recovery and reduces KV storage, while raw1 has better broad language accuracy and prefix-assisted transfer. Neither provides robust general fact recovery. There is no measured request-latency win on the tested workloads. No new training was started for this comparison.

## Matched conditions and verification

Both readers have exactly23,209,216 trainable parameters, identical fresh seed47 initial tensors/RNG, two original epochs,48,762 updates, and361,357,104 original assistant-label exposures each. Both use the same copy/query reader and original training-derived auxiliaries. Compressed64 additionally uses the original frozen9,426,432-parameter encoder; raw1 uses one global position per old token. Raw1 is a matched custom uncompressed control, not a vanilla Transformer or a continuous-KV serving baseline. Its raw64 rollover rebuilds64 old-token states. These latency comparisons apply to the two implementations as measured.

Both full training audits passed: all orders, targets, LR/cursors, within-update microbatch layouts, auxiliary denominators, Adam steps/moment shapes/finite values, frozen source182 and2770 protected-file hashes were checked. Each arm has independent aggregate/source checks plus deterministic CPU public-forward generation replay. The closed literal suite was scored only after both full-exposure gates passed. Final quality comparison and request-efficiency verification both passed.

Primary artifacts: [paired quality](../dump/token-memory-64-wide512-unified-v1/comparison-48762/report.json), [request samples and medians](../dump/token-memory-64-wide512-unified-v1/request-efficiency/report.json), [timing verification](../dump/token-memory-64-wide512-unified-v1/request-efficiency/independent-verification.json).

## Language and factual recovery

Exact factual scores require the complete normalized expected answer plus EOS. Prefix-assisted answers are listed separately. Teacher-forced token accuracy is not factual recall.

| Measure | compressed64 | raw1 |
|---|---:|---:|
| Full-dev token accuracy | 62.322818% | 63.709511% |
| Full-dev CE | 1.726689593 | 1.652498787 |
| Original answerable facts | 67/285 (23.51%) | 27/285 (9.47%) |
| Changed-value answers | 12/64 (18.75%) | 6/64 (9.38%) |
| Changed-question answers | 19/64 (29.69%) | 8/64 (12.50%) |
| Original + changed value both correct | 11/64 (17.19%) | 6/64 (9.38%) |
| Original + changed question both correct | 11/64 (17.19%) | 1/64 (1.56%) |
| Correct missing-information answer | 60/63 (95.24%) | 63/63 (100.00%) |
| Exact answer with fact blocks hidden | 0/64 (0.00%) | 0/64 (0.00%) |
| Notebook full replies | 66/256 | 39/256 |
| Notebook full replies: all four counterfactuals correct | 7/64 | 0/64 |
| Notebook prefix-assisted replies | 158/256 | 130/256 |
| Notebook prefix-assisted replies: all four counterfactuals correct | 18/64 | 6/64 |
| Transfer full replies | 0/224 | 0/224 |
| Transfer prefix-assisted replies | 0/224 | 56/224 |
| Literal-suite known-value controls (all included forms) | 0/512 | 0/512 |
| Literal-suite new values (all included forms) | 0/512 | 0/512 |
| Literal-suite new names + values (all included forms) | 0/512 | 0/512 |
| Literal-suite missing-information cases (all included forms) | 0/128 | 0/128 |

Compressed-minus-raw language difference: -1.386693pp; document95% interval [-1.4381128573136306, -1.3365149242087873]. Original-fact difference: +14.035088pp; interval [9.824561403508772, 18.24561403508772]. These are paired fixed-checkpoint intervals, not uncertainty across training seeds.

Raw1 incorrectly returns the exact unknown template on258/285 answerable original cases; compressed64 does so on195/285. Compressed64 also has82/285 entity-bound value-string matches versus27/285 for raw1. This latter string test is weaker than exact answer recovery. Its recall gain does not establish a broad precision or generalization win.

Both models score zero exact answers in all seven512-case legacy value-only groups. Both score0/103 historical notebook facts and0/32 changed historical facts. The larger literal suite also fails its known-value controls, so its zero new-value score cannot isolate novelty as the cause. Some compressed outputs begin correctly and then repeat/break inside a value. The small53-case last-number probe is reference-number agreement, with incidental numbers and generation-cap confounds; it is not independently certified mathematical reasoning accuracy.

Canonical re-evaluation differs from training validation by one argmax token in each arm: comp1,158,823 rather than1,158,824; raw1,184,607 rather than1,184,606. CE changes are around1e-9. Use the canonical evaluator documents/report for the paired comparison. Checkpoint weights did not change.

## Request speed and memory

RTX5070, FP32, TF32 disabled. Twelve original-dev workloads: old history128/512/2048, raw phase0/63, batch1/4,128 forced token advances per row. One warmup and six measured rounds, alternating arm order. Includes input transfer, original encoder, packing, KV allocation, global/raw prefill, token advances and rollover encoding. Excludes tokenizer, disk/checkpoint loads, model-device transfers, greedy token selection and text decoding. This is fixed-token model-request work, not full chat-serving latency.

| Old tokens / phase / batch | comp request s | raw request s | comp KV MiB | raw KV MiB | comp peak allocated MiB | raw peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| old128-phase0-batch1 | 4.3670 | 4.3296 | 2.157 | 12.001 | 168.62 | 142.66 |
| old128-phase0-batch4 | 4.3739 | 4.3076 | 8.627 | 48.003 | 184.79 | 188.95 |
| old128-phase63-batch1 | 4.3667 | 4.3170 | 2.157 | 12.001 | 169.73 | 143.38 |
| old128-phase63-batch4 | 4.3618 | 4.2917 | 8.627 | 48.003 | 189.74 | 195.77 |
| old512-phase0-batch1 | 4.3524 | 4.2897 | 2.344 | 24.001 | 184.79 | 166.11 |
| old512-phase0-batch4 | 4.3260 | 4.2700 | 9.377 | 96.005 | 265.96 | 299.99 |
| old512-phase63-batch1 | 4.3261 | 4.2820 | 2.344 | 24.001 | 184.79 | 166.11 |
| old512-phase63-batch4 | 4.3530 | 4.2830 | 9.377 | 96.005 | 265.98 | 300.00 |
| old2048-phase0-batch1 | 4.3959 | 4.2398 | 3.094 | 72.003 | 265.96 | 280.55 |
| old2048-phase0-batch4 | 4.3967 | 4.3923 | 12.377 | 288.011 | 590.62 | 744.96 |
| old2048-phase63-batch1 | 4.3480 | 4.2575 | 3.094 | 72.003 | 265.96 | 280.55 |
| old2048-phase63-batch4 | 4.3235 | 4.2910 | 12.377 | 288.011 | 590.69 | 744.96 |

Compressed request medians are 0.10% to 3.68% slower across these workloads. The smallest differences can be noise; there is no demonstrated latency advantage. Near-flat latency across128 to2048 old tokens suggests fixed decoder/reader/runtime work matters in this range; an operator profile is needed for attribution.

KV reduction is5.56x/10.24x/23.27x at128/512/2048 old tokens, including raw64 and the identical conservative generation-reserve policy. The old-token global KV position count alone remains64:1. At2048/batch4, peak PyTorch allocation is590.62MiB versus744.96MiB (about20.7% lower), much less than a23x total-memory saving. At short batch1 contexts the encoder/temporaries make total allocated peak higher for compressed64 despite its smaller KV. Batching4 streams processes about4x the advances in roughly the same wall time as1 stream; it improves aggregate throughput and uses more total memory, not single-stream latency.

The earlier no-reserve illustration was3MiB compressed versus66MiB raw for2048 old tokens plus64 live slots. The measured128-advance request reserves additional capacity and therefore gives3.094MiB versus72.003MiB at batch1. The direction is compressed smaller, raw larger.

## Projection and position ideas

The explicit256-to512 matrix is applied at old-memory prefill and new-code append. Cached next-token decoding reuses the resulting K/V; it does not reproject all old vectors each step. Matrix arithmetic is131,072 multiply-accumulates per new64-token code, or2,048 amortized per old token, excluding norm/gain/type work. Removing it is not a demonstrated route to the current latency bottleneck.

User preference: a final design with no separate width adapter. A native256 core should create256-wide raw embeddings directly, rather than first create512-wide embeddings and project each token down. At a fixed parameter budget, depth/FFN allocation must be considered: smaller width alone does not guarantee less KV or latency. The previous21-layer256-wide layout has31.25% more layer-width product than this8-layer512-wide layout. Matching dimensions also does not automatically align encoder and token feature bases.

Keep the existing encoder output256 for the first native-width experiment; do not shrink it simply to reserve position channels. A naturally512-wide encoder could match a512-wide reader, but changes encoder cost, code storage and training. An output500 plus12 position channels is not a free optimization. Encoder input block length and output feature width are independent. No new version or training was implemented here.

First low-risk speed candidate: precompute the fixed query-selector table at inference. The current code projects4096 normalized512-wide embeddings to8 scores each call:16,777,216 multiply-accumulates. The resulting8-by4096 FP32 table is128KiB, independent of context length; it is model-wide derived data, not extra per-context KV. Verify logits/outputs and measure timing before adoption. Fusing repeated reader operations is another profiling target.

Targeted quality hypothesis: `token_memory_64_lexical/reader.py:read` pools source probabilities by token ID before the embedding readout. Two occurrences of the same token therefore have identical value embeddings. The model still has context/recency and global RoPE; this is not proof all positions are absent. A small weighted source-position signal in the same readout/state might help it select and follow a source span coherently. Token offsets already exist in the payload order; no wider encoder, per-word global KV entries or persistent word-feature bank is needed. This adds local arithmetic and needs a controlled training/generalization test. It is not a verified fix.

## Exact bit-packing example

This is an illustrative vector, not actual trained encoder output. TokenID291 is hex0x0123. Its four nibbles3,2,1,0 replace the lowest four fraction bits of four consecutive FP32 numbers:

| Component (one based) | Original float | Packed float | low4 bits |
|---|---:|---:|---|
| 1 | 1.0 | 1.0000003576 | 0011 |
| 2 | 2.0 | 2.0000004768 | 0010 |
| 3 | 3.0 | 3.0000002384 | 0001 |
| 4 | 4.0 | 4.0000000000 | 0000 |

Recover291 =3 +2*16 +1*256 +0*4096. Token0 uses components1–4, token1 uses5–8, and token63 uses253–256. The grouping supplies within-block offset; block j and offset i give token coordinate64*j+i. No token is dropped to insert positions. Four low semantic precision bits per float are replaced. The neural encoder checkpoint is unchanged; literal packing is additional postprocessing and an exact-information path.

The current format uses256 FP32 carrier numbers for64 sixteen-bit payload slots. The current vocabulary IDs use12 bits. It is global attention/KV compression, not a claim of learned lossless semantic compression or byte compression versus token IDs: one input code is1024bytes, while64 uint16 IDs are128bytes. Direct FP16/BF16 conversion of payload-bearing values destroys this format. A100M parameter count does not change that requirement; retaining FP32 is a design choice with bandwidth/memory costs.

Only the live raw window is64. With2048 old tokens, the model stores32 global code positions but the local reader still touches2048 token candidates. It recovers IDs temporarily without allocating2048 persistent contextual K/V entries. The global quadratic prefill reduction therefore coexists with query-by-history local-reader work; it is not64x less total work.

## Decision and remaining scope

Retain the completed checkpoints as immutable references. First profile/remove redundant inference work with output-parity checks; then investigate source selection versus copy continuation and test position-aware readout within the existing width. Evaluate a no-projection/native-width candidate separately instead of changing encoder width, positions and core allocation together. New model selection needs fresh held-out cases because the closed literal suite is now exposed. No quality/adoption claim or100M-versus10B claim follows from this study. The broad compression-quality objective remains unmet.

[Recorded user preferences and hypotheses](wide512_final_architecture_preferences_20261020.md).
