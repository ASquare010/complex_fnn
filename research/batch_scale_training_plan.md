# H142: doubled-batch qualification of the maintained memory helper

Previous turn: progress. H141 added an opt-in helper and verified its arithmetic,
state updates and memory against frozen code. H140 passed short paired timing;
H138's long-run failure remains unchanged. No existing source/default changes.

Double only training batch size from 8 to 16 at context 512. Keep the six H140
source states, model size (9,099,648 parameters), optimizer/LR/clipping, FP32,
TF32 off, ordinary attention, four CPU threads and 8.125 MiB workspace fixed.
Validation remains at batch 8 with unchanged complete target coverage. This is
activation-workload scaling, not a larger model or another task domain.

Use maintained buffer_model_loss and offload_checkpoint_inputs(last_n_blocks=4)
against ordinary native loss/no offload. Repeat H140's 24 segments: all six
corpus/seed fixtures in alternating ABBA/BAAB order; each of four 30-update
segments resets to the same step-800 model/Adam/sampler state. Ten warmups and
20 timed updates per segment. One GPU model resident; clean boundaries between
segments; passive 200 ms telemetry; no hardware adjustments.

Unchanged per-fixture gates: combined 40-update median CUDA and wall ratios
<=1.15, max whole-job GPU allocated peak ratio <=0.90, both repeat NLL ratios
<=1.01, candidate pinned host peak <=128 MiB; each segment's half-median timing
stability and each arm's between-repeat median ratio <=1.15. Require all six
fixtures, every repeat, matching source/data hashes, finite states, correct
wrapped suffix, telemetry coverage and complete independent audit. Report
neighboring pairs, means/medians/sample variance, setup and segment wall times.
No best-repeat selection, clock normalization or retrospective threshold changes.

Derive the training loop from H140 by changing only batch_size=16, batch(16,512)
and the reported targets per update to8192. Derive the independent H120 native
audit by changing only its sampled batch size to16; numerical Adam/clipping and
native validation code remain unchanged. Verify these changes with AST checks.
Preserve existing gradient-noise caps and all original numerical tolerances.

Budget: 720 updates/backwards, 5,898,240 training targets, 48 study scores and
24 independent native scores, 72 tensor artifacts, 1,608 memory-phase records,
25 training plus25 audit zero GPU allocator boundaries. Audit adds no backwards.
About47GB free before preparation; expected artifacts below8GB. Stop on process
failure, preserve completed work and inspect live state before any recovery.

A pass extends only the measured batch-size scope of the opt-in helper. It does
not establish sustained throughput, fresh convergence at batch16, cross-domain
quality, parameter reduction or new activation geometry. Preserve H138. After
this bounded scaling check, return to the unresolved structural FFN/parameter
question rather than indefinitely accumulating memory-only microbenchmarks.
