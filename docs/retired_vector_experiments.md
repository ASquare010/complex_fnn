# Retired vector experiments — 16 October 2026

The user ended the growing/vector-generation direction and requested removal of
its variants, source code, checkpoints and caches. Work now focuses on
`src/models/token_memory_64`. This is a retirement record, not a recommendation
to resume any of the experiments below.

## What is retained

- The frozen 64-token codec in `position_compressor`, its original checkpoint and
  tokenizer, and its `branch_sigmoid` dependencies.
- The original 512-token `token_memory` baseline and completed plain
  `token_memory_64` run, including their checkpoints and source provenance.
- The completed 64-token cross-attention implementation/run and the stopped
  attention-residual checkpoint/results for comparison. The attention-residual
  source was already absent; its orphaned evaluator/test are retired here.
- Raw-transformer reference code, chat datasets, data preparation, evaluation
  tools and the completed-block cache `dump/token-memory-64-cache-v4`.
- Existing research Markdown reports. References in those historical reports to
  retired code or binary artifacts no longer describe runnable experiments.

## Growing token experiments

These models predicted ordinary next-token logits. Their inputs contained
completed compressed blocks plus an active code re-encoded from the strictly
past generated/teacher-forced prefix. At 64 tokens the code was committed and
the active prefix cleared. The encoder was frozen. Prefix caches were reused
across training epochs; learned readers were recomputed during training.

The full-corpus recipe used fresh seed47, 23,209,216 trainable parameters,
780,186 original segments per epoch, effective batch32, FP32, assistant-token
cross-entropy and the original AdamW schedule. Two planned epochs meant 48,762
updates and 361,357,104 assistant-label exposures. None of the three growing
full-corpus runs below completed that schedule.

| Experiment | Reader / attention positions per code | Last durable update | Latest full-dev update | Dev CE | Dev token accuracy |
|---|---|---:|---:|---:|---:|
| Original growing token | Normalize/project one code to one feature; 1 position | 35,472 | 35,000 | 3.396384 | 35.4847% |
| Query-conditioned reader | Low-rank query-dependent reading at layers 4 and 8; 1 position | 6,834 | 6,000 | 5.289426 | 16.0129% |
| Four learned readouts | `256 -> 4 x 512`, SiLU; 4 positions | 2,000 | 2,000 | 5.918671 | 9.8103% |

Every full-dev row above scores 1,859,388 targets. Last durable update is read
from the actual `last.pt`, with its SHA-256 checked before deletion. Four-readout
training logged through update 2,140, while its stale status file reported
2,125 and an inactive PID. Its last saved weights were update 2,000. No trainer
was alive at the cleanup process check; no unrelated process was terminated.

The query reader used head rank16 and narrowed two FFNs from1534 to1518 to keep
the parameter count fixed. At matched update6,000, original growing dev CE was
5.275467 and accuracy16.1385%, versus5.289426 and16.0129% for the query reader.
This showed no early improvement; it is not a final two-epoch comparison.

Four readouts narrowed all eight FFNs from1534 to1486. The four features were
learned readouts, not four predetermined 16-token chunks. They multiplied the
memory attention positions and KV bytes by four. A prior zero-update benchmark
measured median effective-batch forward/backward time0.87791s versus0.44572s
for original growing at microbatch4. Those timings excluded disk access,
encoding, optimizer work, validation and saving. The user's training output was
about1 update/s; no completed quality or fact-recovery result exists.

The earlier full decoder-unpacking pilot was already rejected and removed.
Expanding each code into64 position features lost the desired sequence-length
advantage. It reached two optimizer updates, failed at its own memory cap on the
next update, and had no useful quality result. It is not revived by this cleanup.

The earlier, separately fine-tuned growing token pilot used1,024 matched updates
per arm from the trained baseline. Synthetic answer recovery was16/204 baseline
versus6/204 growing; real SQuAD recovery was1/209 versus0/209. Both were0/204 on
paired original/changed-fact correctness. See the preserved
[pilot report](growing_token_memory_pilot.md).

## Why retain the token-memory baselines

| Completed model | Updates | Test CE | Test token accuracy | Exact old-notebook facts |
|---|---:|---:|---:|---:|
| Original raw512 + memory | 48,762 | 1.806954 | 60.9990% | 0/103 |
| Plain raw64 + memory | 48,762 | 2.216948 | 53.5088% | 0/103 |
| Raw64 + gated memory cross-attention | 48,762 | 2.249793 | 53.0127% | 0/103 |

These three completed runs had matched batches, labels, learning rates and
parameter counts. Their language metrics are comparable; they did not establish
reliable old-fact recovery. The attention-residual64 run was intentionally
stopped at37,045 updates and therefore has a different training budget.

Preserved reports: [plain64](token_memory_64_results.md),
[cross-attention64](token_memory_64_crossattn_results.md), and
[attention residuals64](token_memory_64_attnres_results.md).

## Earlier retired families and written evidence

The cleanup also retires obsolete latent-output, growing-vector, flow,
constructor, planner, grounded-codec and hierarchical-memory artifacts. These
were separate experiments, not equivalent training schedules or a controlled
single-variable sweep. Their conclusions must be read with their own protocols.

- [Latent prediction diagnostic](latent_prediction_diagnostic.md) and
  [frozen/growing-vector diagnostic](frozen_growing_vector_diagnostic.md).
- [Frozen-codec memory investigation](frozen_codec_memory_investigation.md),
  [probabilistic generation](probabilistic_vector_results.md), and
  [attention/composition curriculum](vector_curriculum_results.md).
- [Shared planner](shared_vector_planner_results.md),
  [feature planner](feature_vector_planner_results.md),
  [slot constructor](slot_vector_constructor_results.md), and
  [position constructor](position_vector_constructor_results.md).
- [Preconditioned flow](preconditioned_flow_results.md),
  [full-answer flow](full_answer_flow_results.md),
  [direct clean-code prediction](data_prediction_flow_results.md),
  [conditional flow head](conditional_head_flow_results.md), and
  [decoder-feature reader](decoder_reader_flow_results.md).
- [Grounded reading comparison](grounded_codec_results.md): held-out answerable
  exact scores raw1/410, deterministic vector0/410, flow vector0/410. The weak
  raw control means this does not isolate compression as the cause.
- [Query-reader design and setup](growing_query_reader_setup.md).

No retired family demonstrated a replacement for the original model with both
useful factual reading and preserved broad language quality.

## Position handling in the retained raw64 model

The current implementation assigns memory codes positions `j` and raw queries
positions `completed_code_count + within_block_offset`. A completed64-token
block therefore advances the coordinate by1, while a raw token advances it by1.

A future controlled experiment can instead use original-token coordinates:
anchor completed block `j` at `64*j + 63` and put the next block's query positions
at `64*completed_code_count + within_block_offset`. With the BOS/shifted input
convention, raw positions refer to the next-token prediction positions. Within
each block the token offsets still advance by1. This changes coordinate values,
not the number of vectors or the attention mask.

It would preserve parameter count, attention matrix dimensions and KV tensor
shapes. The same RoPE rotations are performed; coordinates can be precomputed.
An exact wall-clock speed claim still requires measurement. Outer block
coordinates do not expose64 separate internal token positions or guarantee
better recovery from a compressed vector. Existing weights must keep their old
position convention for reproducible comparisons. No model architecture or
training run is changed by this cleanup.

## Audit and records

The [retirement record directory](experiment_records/vector_retirement_20261016/)
contains an exact deletion manifest, compact per-run metadata, verified last
checkpoint identities for the three full growing runs, and protected baseline
hashes. `evidence.zip` contains text records only: recipes, metrics, logs,
verification reports and historical Markdown. It contains no model source,
compiled code, checkpoint tensors or training cache. Original prose from the
repository README and AGENTS file is also retained there.

Cleanup completion and validation results are recorded at the end of this document.

## Inventory of retired run directories

This lists saved status fields before removal. `n/a` means no status file supplied that field; it does not mean zero updates. Tiny synthetic test fixtures are listed separately in the machine-readable deletion manifest.

| Run directory | Recorded state | Recorded step | Size (GiB) |
|---|---|---:|---:|
| `ambiguous-vector-generation-20261013` | n/a | n/a | 0.2194 |
| `block-vector-ce-mse-v4` | paused | 7096 | 0.0036 |
| `block-vector-pilot-300` | n/a | n/a | 0.0008 |
| `block-vector-pilot-300-v2` | n/a | n/a | 0.0008 |
| `conditional-head-flow-20261013` | complete | n/a | 0.2215 |
| `data-prediction-flow-20261013` | complete | n/a | 0.2215 |
| `decoder-reader-flow-20261013` | complete | n/a | 1.2278 |
| `encoder-latent-ce-21m-fullcontext` | preparing | n/a | 0.0001 |
| `encoder-latent-ce-21m-fullcontext-v2` | training | 5900 | 5.5182 |
| `encoder-latent-ce-context64-320k` | paused | 6708 | 0.5626 |
| `encoder-latent-ce-trainable-decoder-320k` | failed | n/a | 0.0001 |
| `encoder-latent-ce-trainable-decoder-320k-v2` | paused | 8239 | 0.5628 |
| `encoder-latent-crossentropy-batch32` | complete | 1000 | 0.0154 |
| `encoder-latent-mse-320k` | preparing | n/a | 0.0001 |
| `encoder-latent-mse-320k-online` | training | 19100 | 0.5640 |
| `encoder-latent-mse-batch32` | complete | 1000 | 0.0154 |
| `encoder-latent-mse-batch32-10k` | complete | 10000 | 0.0162 |
| `feature-vector-planner-20261013` | n/a | n/a | 0.9583 |
| `flow-vector-real-20261013` | n/a | n/a | 1.3155 |
| `frozen-loss-diagnosis-20261013` | n/a | n/a | 0.3063 |
| `frozen-memory-generation-20261013` | n/a | n/a | 0.2104 |
| `frozen-memory-generation-oracle-20261013` | n/a | n/a | 0.2103 |
| `frozen-memory-reader-20261013` | n/a | n/a | 0.1028 |
| `frozen-memory-router-20261013` | n/a | n/a | 0.1024 |
| `frozen-memory-router-expanded-20261013` | n/a | n/a | 0.1028 |
| `frozen-vector-real-300` | n/a | n/a | 0.0024 |
| `full-answer-flow-data-20261013` | n/a | n/a | 0.4961 |
| `full-answer-flow-data-audit-20261013` | n/a | n/a | 0.0011 |
| `full-answer-flow-data-audit-initial-exclusions-20261013` | n/a | n/a | 0.0011 |
| `full-answer-flow-diagnosis-20261013` | n/a | n/a | 0.0002 |
| `full-answer-flow-preflight-20261013` | n/a | n/a | 0.0019 |
| `full-answer-generation-check-20261013` | n/a | n/a | 0.0001 |
| `full-answer-lifecycle-tests-20261013` | n/a | n/a | 0.0000 |
| `full-answer-literal-end-20261013` | n/a | n/a | 0.4997 |
| `full-answer-literal-flow-20261013` | complete | n/a | 0.3117 |
| `grounded-codec-20261013` | n/a | n/a | 1.1057 |
| `grounded-codec-data-20261013` | n/a | n/a | 0.1510 |
| `growing-token-pilot-20261013` | n/a | n/a | 0.7489 |
| `growing-token-query-reader-v1` | paused | 6834 | 0.7815 |
| `growing-token-readout4-v1` | training | 2125 | 0.7794 |
| `growing-token-unified-v4` | paused | 35472 | 0.7946 |
| `growing-vector-23m-8to64-v4` | paused | 4595 | 0.8277 |
| `growing-vector-packing-speed` | n/a | n/a | 0.0000 |
| `growing-vector-real-300` | n/a | n/a | 0.0000 |
| `hierarchical-memory-64-20k` | training | 5075 | 0.0006 |
| `hierarchical-micro16-20261012-091238` | complete | 20000 | 0.0016 |
| `hierarchical-resume-20261012-090022` | paused | 6186 | 0.0002 |
| `latent-baseline-audit-v1` | n/a | n/a | 0.0000 |
| `latent-ce-20261009-170025` | complete | n/a | 0.0153 |
| `latent-diagnostic-v1` | stopped_before_training | n/a | 0.0001 |
| `latent-diagnostic-v2` | complete | n/a | 0.0486 |
| `latent-direct-ce-v1` | complete | n/a | 0.0003 |
| `latent-mse-only-v1` | complete | n/a | 0.0487 |
| `latent-pilot-v1` | complete | n/a | 0.0153 |
| `position-memory-joint-20261013` | n/a | n/a | 0.4743 |
| `position-memory-joint-v4` | paused | 1001 | 0.8258 |
| `position-memory-speed` | n/a | n/a | 0.0001 |
| `position-memory-v4-verification` | n/a | n/a | 0.0002 |
| `position-vector-constructor-20261013` | n/a | n/a | 2.0995 |
| `preconditioned-flow-20261013` | n/a | n/a | 1.1456 |
| `probabilistic-vector-20261013` | n/a | n/a | 0.2813 |
| `sampled-plan-vector-20261013` | n/a | n/a | 0.1156 |
| `shared-vector-planner-20261013` | n/a | n/a | 0.4431 |
| `slot-vector-constructor-20261013` | n/a | n/a | 1.1611 |
| `two-pass-vector-memory-20261013` | n/a | n/a | 0.1253 |
| `vector-attention-curriculum-20261013` | n/a | n/a | 0.1114 |
| `vector-composition-curriculum-20261013` | n/a | n/a | 0.4587 |
| `vector-encoding-speed` | n/a | n/a | 0.0000 |
| `vector-memory-unified-v4` | training | 2075 | 0.0011 |
| `vector-memory-verification` | n/a | n/a | 0.0000 |

## Cleanup completed

- Removed 316 local targets, including obsolete model/tool/test
  source, runs, source archives, and48 additional synthetic fixture directories
  identified by their embedded growing-model sources. Removed the separate
  `I:\growing-token-cache-v4` prefix cache as well.
- Removed 218,008,047,606 bytes (203.036 GiB gross). The retained
  text evidence and audit records occupy about87 MiB; net recovery is about203 GiB.
- All124 protected files were SHA-256 identical, including retained model Python
  sources, codec and comparison checkpoints, unified-v4 files and raw64 cache.
- 23 CPU checks passed and2 historical-fixture checks were skipped. One codec
  check initially hit a Windows temporary-directory permission error; the
  unchanged test passed outside the sandbox. The first collection invocation
  also needed `src` added to the Python import path. No model/test source was
  changed to obtain these results.
- All54 remaining tests collected without import errors. Remaining Python imports
  were checked for dependencies on retired modules;
  retained source archives contain no growing-model code. No corpus training or
  GPU experiment was launched; only synthetic CPU optimizer checks were run.

See [verification.json](experiment_records/vector_retirement_20261016/verification.json)
for the verification receipt and the adjacent manifests for exact removed paths.

### Rechecked after the repeated cleanup request

On 16 October 2026 at 11:28 UTC, all 316 retired local targets and the external
growing-prefix cache were still absent. No growing/vector source filenames
remained in `src`, `tools` or `tests`. All 124 protected files matched their
original SHA-256 hashes. The retained evidence archive contains text records,
with no Python sources or model/cache binaries. No further deletion, architecture
change or training was needed for this request. See the
[recheck receipt](experiment_records/vector_retirement_20261016/cleanup-request-recheck-20261016.json).
