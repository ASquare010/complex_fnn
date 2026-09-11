# Memory- and parameter-efficient FFN research

Latest: [H156 fresh batch16 long training](research/batch_scale_long_results.md):
**PASS fresh batch16 long-training gate.** 12 continuous runs, three seeds and two corpora; independent audit True.
Memory optimization only; the broad FFN research goal remains open.

Earlier: [H155 augmentation dimension proof](research/augmentation_barrier_results.md):
exact even output with full affine span requires at least as many extra state
coordinates as output dimensions, under the stated reversible/linear-readout assumptions.
CPU witnesses verified; no training or VRAM-performance claim.

Earlier: [H154 narrow coupling](research/narrow_coupling_results.md):
NO PROMOTION from narrow coupling. 120 audited fits; 74% fewer parameters does not establish quality.

Earlier: [H153 even-direction diagnostic](research/even_tangent_results.md):
NO PROMOTION from even-direction diagnostic. Tangent-space gains do not establish nonlinear learning.

Earlier: [H152 readout learning pilot](research/readout_learning_results.md):
NO PROMOTION from the readout learning pilot. Mandatory activations and fixed/affine controls included.

Earlier: [H151 fixed-tail capacity proof](research/fixed_tail_capacity_results.md)
identifies a representational limit and a concrete readout extension to test.

Earlier: [H150 mirrored-order confirmation](research/fused_timing_confirmation_results.md):
FAIL mirrored-order resource confirmation. Learning quality remains unproven.

Earlier: [H149 fused reconstruction](research/fused_reconstruction_results.md):
REJECT tested fused-reconstruction resource recipe. Resource evidence is separate from learning quality.

Earlier: [H148 reconstruction training screen](research/reversible_training_results.md):
REJECT tested reconstruction resource recipe. Fixed buffers counted; learning quality remains unproven.

Earlier: [H147 reversible scalar preflight](research/reversible_softsign_results.md)
passes reconstruction and gradient checks through128 layers. Resource savings
and learning quality remain unproven; research continues.

Earlier: [H146 depth/recomputation screen](research/modulation_depth_results.md):
REJECT tested depth/recomputation modulation recipes. Audited resource evidence; research goal remains open.

Earlier: [H145 resource screen](research/split_modulation_results.md)
rejects shared/split output modulation: fewer parameters did not yield enough
VRAM savings. Audited results; no model/default changes. Research goal remains open.

Earlier: [H144 capacity proof](research/modulated_capacity_results.md)
identifies a shared-latent output-modulation limit before GPU fitting. No active
model/default changes; the broad research goal remains open.

Earlier: [H143 structural FFN screen](research/conjugated_ffn_results.md): **REJECT fixed shared-permuted recipe.** All recipes and failed gates are retained; no default changes.

Earlier: [H142 batch-16 qualification](research/batch_scale_training_results.md): **PASS scoped batch-16 qualification gate.** H138 remains failed; no maintained default changes.

Earlier: [H141 opt-in memory helper](research/training_memory_integration_results.md)
passed source-equivalence, numerical and lifecycle checks. [Usage](research/training_memory_usage.md).
Ordinary defaults are unchanged; broader performance/generalization remains open.

Earlier: [H140 interleaved timing](research/interleaved_training_results.md): **PASS scoped interleaved timing gate.** H138 remains failed; no maintained default changes.

Earlier: [H139 timing-drift audit](research/timing_drift_audit_results.md) identifies
a P0-to-P4 transition in H138's unstable control. Zero further GPU training;
H138 remains failed. See the report for the next controlled timing experiment.

Earlier: [four-block fresh validation](research/partial_offload_long_results.md).
**FAIL four-block fresh-training gate.** Three seeds, two corpora and independent checkpoint audit.
The broader FFN research goal remains open.

Earlier: [partial checkpoint offload](research/partial_offload_training_results.md).
**PASS short qualification: buffer4.** Explicit memory/transfer tradeoff with matched controls.
Fresh validation and the broader research goal remain open.

Earlier: [fresh three-seed memory validation](research/ordinary_long_training_results.md).
**FAIL three-seed long-training gate.** 18 fresh runs with native controls and independent checkpoint audit.
The broader FFN research goal remains open.

Earlier: [ordinary-policy complete training](research/ordinary_complete_training_results.md).
**PASS ordinary short-training gate.** Matched memory/runtime controls and calibrated numerical audit.
Full training and the broader research goal remain open.

Earlier: [paired complete training](research/paired_complete_training_results.md).
**FAIL complete short-training gate.** Full update/peak-memory comparison; 360 updates and independent audit.
The broader research goal remains open.

Earlier: [paired workspace timing](research/paired_workspace_timing_results.md).
**PASS paired timing gate.** Balanced probes and passive GPU telemetry; exact native gradients.
This is timing evidence only; the full VRAM/quality research goal remains open.

Earlier: [intermediate-workspace comparison](research/blas_workspace_mid_results.md).
**FAIL combined diagnostic gate.** Verified allocation and gradient evidence; runtime limitations
and next experiment are documented. The full research goal remains open.

Earlier workspace study: [deterministic workspace isolation](research/blas_workspace_results.md).
**FAIL combined diagnostic gate.** Direct allocation and gradient evidence; see measured runtime
tradeoffs and the next test. The full research goal remains open.

Earlier complete-training study: [complete native-buffer training](research/native_buffer_training_results.md).
**FAIL combined short-training gate.** Eight matched short continuations; see the report for exact
resource/quality comparisons and limitations. The broader goal remains open.

Earlier reproducibility study: [reference reproducibility](research/attention_reproducibility_results.md).
**PASS scoped reproducibility: deterministic_default, deterministic_math.** Fixed-state evidence only; training and resource validation remain.

Earlier diagnostic: [native-buffer classifier](research/native_buffer_layout_results.md)
saves 14.5–14.8% GPU allocation; full-model bitwise replay remains unresolved
in both candidate and native controls. No training-quality or breakthrough claim.

Earlier long-training study: [long-training validation](research/compact_long_training_results.md).
**FAIL: the candidate does not pass every fixed long-run gate.** Six 800-update runs and independent audits are complete.
See [current research state](research/CURRENT_STATE.md) for the next decision.

Earlier short-run evidence: [short complete-training validation](research/compact_training_results.md)
measures 13.82–14.42% lower whole-job VRAM with passing numerical and validation
checks. Longer runs and multiple seeds remain required. See
[current research state](research/CURRENT_STATE.md) for the next experiment.

The earlier [checkpoint-input offload diagnostic](research/checkpoint_input_offload_results.md)
saves **48 MiB (about 12%)** with native classifier loss and about **15 MiB
(5%)** with chunked loss. All gradient and timing checks pass, but the chunked
cases fail the preset 10% memory gate. These are 262 fixed-state backward
checks with zero optimizer updates, not a training-quality or parameter-saving
result. No default changed; the broader research goal remains open.

The preceding [optimizer memory profile](research/optimizer_memory_results.md)
finds that fused AdamW saves **35–36 MiB during the optimizer step**, but saves
**no complete-job VRAM**: backward sets every peak. Per-tensor AdamW also fails
the memory gate and is slower. All 12 matched continuations, 360 updates and
independent numerical/data/scoring audits complete. Both alternatives are
eliminated for this memory objective; model defaults remain unchanged.

The preceding [Adam first-update diagnostic](research/adam_update_sensitivity_results.md)
measures **61–64x amplification** of tiny saved gradient differences, while the
resulting update-direction difference remains about 0.0017%. Larger epsilon
reduces it but changes the direction by 10.8–32.8%, failing the fixed distortion
criterion. All GPU numerical checks pass; the combined audit fails CPU clipping.
Twelve disposable steps use **zero new language training**. This scoped mechanism
does not repair the old quality failure or establish new VRAM savings.

The preceding [repeat-variability diagnosis](research/training_variability_results.md)
adds four matched 800-update runs on the failed WikiText seed. Chunking remains
worse in both new pairs (+1.143% / +0.940% NLL), while unchanged native runs also
vary. The fixed diagnostic is **INCONCLUSIVE**; the prior two-corpus gate stays
failed. Initial gradient agreement does not establish training-trajectory
agreement. All checkpoint, native-score, replay and batch checks pass. No model
default changes and no new FFN breakthrough is established.

The preceding [fresh three-seed replication](research/fp32_training_replication_results.md)
completes **18 runs / 14,400 updates**. FP32 decoder/classifier chunking saves
**26.98–27.75% full-job tensor allocation** versus BF16 native. wikitext2 fails the fixed gates; tinystories qualifies in this scope.
The fixed two-corpus claim fails.
Every seed uses 800 updates from matched random initialization; a passing mean
cannot override a failed seed. The independent audit passes:
six regenerated initial states, 54 trained checkpoints, 60 native scores,
12 FP32 gradient replays and all 14,400 sampled batches. No new architecture,
parameter reduction or default change is established. The broad goal stays open.

The preceding [full-job FP32 resource screen](research/fp32_decoder_resource_results.md)
finds a qualified narrow-model execution recipe: **26.97-27.23% lower allocation**,
with update time approximately unchanged versus BF16 native training. Default
attention with FP32 decoder/classifier chunking passes numerical, memory and
short-continuation quality gates on both narrow corpus fixtures. Thirty matched
continuations / 1,500 updates and independent gradient/state/scoring checks
complete. These are correlated single-seed fixtures and 50-update screens;
broader replication and duration are still required. Full GELU/SwiGLU and math
attention fail the memory target. There is no new layer or parameter reduction.

The previous [decoder diagnosis](research/decoder_gradient_transport_results.md)
separates repeat variability from gradient transport error. Thirty fixed-state
conditions and 320 backward passes, with **zero optimizer updates**, show that
BF16 can turn classifier-gradient differences around 1e-6 into decoder-gradient
differences around 0.003. Removing checkpointing does not solve this. Both
FP32 decoder modes pass the unchanged numerical gates and earn a separate
whole-model memory/runtime screen. No new language training is earned yet.


The previous [complete-model precision screen](research/fp32_classifier_profile_results.md)
finishes 56 matched checkpoint continuations / 2,800 updates. FP32 classifier
chunks save **32-33% whole-job allocation** on narrow context-512 models, with
about **13% slower updates**, but fail the original global-gradient fidelity
gate. Full context-128 GELU/SwiGLU save only 0.53-1.08%. Two gradient replay
audits also fail. Native scores and all batches verify; the fixed qualification
claim is closed and no fresh language allocation is earned. These are scoped
memory observations, with no parameter reduction or breakthrough claim.

**The architectural research goal remains unmet.** The current primary objective
is lower measured VRAM with preserved quality and practical runtime. Parameter
reduction is one possible means. We also test whether a smaller FFN can
match full GELU and SwiGLU with at least 70% fewer FFN weights, while beating
calibrated narrow controls at practical memory and runtime costs.

Plain BlockShuffle achieves **70.31% fewer FFN weights** and **42.17% fewer total
model weights**. In the latest one-seed 3,200-update WikiText-2 comparison its
validation loss is 4.1443 versus 4.1054 for full SwiGLU. Earlier three-seed evidence
misses the 1% loss allowance, and native updates are slower. These are promising
compression measurements, not a completed breakthrough.

The latest complete [training comparison](research/whole_job_memory_results.md)
completes **12 fresh trials / 9,600 updates**. Chunked training with the same
streamed evaluator for both arms reduces measured job tensor allocation:

| Corpus, three seeds | Reference → candidate peak | Quality and runtime | Decision |
|---|---:|---|---|
| TinyStories | 402.8 → 269.6 MiB (**33.06% saved**) | NLL changes -0.51% to +0.32%; updates 13.8-14.5% slower | Scoped component qualifies |
| WikiText-2 | 408.6 → 276.1 MiB (**32.43% saved**) | Two seeds exceed the 1% NLL allowance | Fixed recipe fails |

Independent audit verifies **72 native scores, 48 checkpoints and every sampled
batch**. These are PyTorch tensor allocations including training, diagnostics,
validation and corpus caches; driver/context memory is excluded. All trials use
the same narrow GELU architecture and parameter count. The two-corpus claim is
rejected, while the TinyStories memory component is retained. This is established
execution technology, with no new activation or general superiority claim.

The follow-up [precision diagnostic](research/classifier_precision_results.md)
identifies an additional error source: shared BF16 weight casts accumulate
chunk gradients before converting them to FP32. All 24 saved states reproduce
the effect with unchanged losses and hidden gradients. FP32 classifier chunks
reduce local gradient error to about 4e-7 and save **72% of classifier allocation**.
That is a local result with zero optimizer updates. Timing varies widely, so the
candidate still needs complete-model memory and cost qualification. All 144
gradient reruns and 24 independent closed-form references pass.

The preceding [evaluation study](research/streamed_evaluation_results.md) qualifies
classifier streaming on all 24 saved states: 16-31% less evaluation allocation,
approximately 1% median timing overhead and negligible score drift. Processing
fewer complete sequences costs 3.7-7.1 times as much and is rejected at these shapes.
Native runtime failures and bounded recoveries are preserved; their root cause
remains unresolved. Maintained code/defaults stay unchanged from the prior
116-test pass. Both studies use the existing GPU and UV-managed environment.

Other recent results remain fully documented:

- [800-update execution study](research/token_memory_duration_results.md):
  training savings pass at short context, but native evaluation limits job savings.
- [Derivative initialization](research/sobolev_learning_results.md): its initial
  advantage largely disappears under value-only learning; fixed recipes closed.
- [Derivative-aware calibration](research/sobolev_selection_results.md): useful
  local component, while complete compressed-model quality gates fail.
- [Affine residuals](research/affine_residual_fit_results.md),
  [projection/rank limits](research/real_subspace_results.md) and
  [feature discovery](research/spectral_fitting_results.md): audited negative
  results that constrain the next architecture.

## Start here

- [Progress overview](research/PROGRESS_OVERVIEW.md): results, data, batches,
  duration and activation experiments.
- [Current state](research/CURRENT_STATE.md): decisions and outstanding evidence.
- [New research direction](research/research_direction_2026_09_10.md): coupled
  feature-pair geometry, falsifiable comparisons and prior work.
- [Candidate ledger](research/idea_bank.md): hypotheses and eliminated branches.
- [Goal](doc/RESEARCH_GOAL.md) and [rules](doc/RESEARCH_RULES.md).

## Maintained code

Only **two model folders, five variants and eight recipes** remain active.

| Folder | Role |
|---|---|
| [dense_ffn](src/dense_ffn/README.md) | Full and narrow GELU/SwiGLU controls |
| [blockshuffle_ffn](src/blockshuffle_ffn/README.md) | Compressed comparison operator; its limitations remain documented |

Shared data, training, evaluation and diagnostics live in src/core. Rejected
architectures, including rational BlockShuffle after repeated memory failures,
are [archived](research/archive/README.md). Isolated prototypes stay with their
experimental source until evidence earns integration.

## Run

Use Python 3.12+, UV and a CUDA-capable PyTorch installation. Existing measurements
use one RTX4070 Laptop GPU with 8 GiB VRAM. Keep GPU experiments sequential.

~~~powershell
uv sync --extra compile --extra data
uv run --extra compile --extra data python -m src.core.cli hardware
uv run --extra compile --extra data python -m src.core.cli counts
uv run --extra compile --extra data python -X faulthandler -m pytest -p no:anyio -q
uv run ruff check src tests scripts main.py
~~~

With the local WikiText-2 cache prepared, a new reference run can be started with:

~~~powershell
uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_blockshuffle_stronger_800.json --cache data/wikitext2_v1 --seed 17
~~~

Read the [recipe notes](configs/README.md) before comparing selected learning rates.
Training creates new artifacts and refuses to overwrite completed runs.

## Evidence without large Git history

This repository keeps readable research, source, compact results and essential
source snapshots. Datasets, checkpoints, raw gradient tensors, caches and repeated
run archives remain local and are ignored. They have not been deleted.
See [artifact policy and inventory](research/ARTIFACTS.md) for exact scope and the
limits of reproducing historical measurements from a small clone.

Local mathematical proofs, controlled fitting, language quality and GPU speed
are separate claims. Neither a better synthetic score nor a passed gradient test
establishes general superiority, novelty, convergence or a breakthrough.
