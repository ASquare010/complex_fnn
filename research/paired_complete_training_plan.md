# H134: complete training with 8-MiB workspaces and mirrored run order

Previous goal turn: progress (H133 measured/audited exact gradients and passed
paired timing). The broad low-VRAM, quality and parameter-efficiency goal is open.

Run WikiText-2 then TinyStories. Each has six fresh 30-update continuations from
its unchanged step-800 H120 fixture, ordered reuse, ordinary, native, native,
ordinary, reuse. Pair first reuse with first ordinary/native, and last reuse
with last ordinary/native. These are two mirrored-order measurement repeats,
not additional independent training seeds. All runs see identical batches.

Ordinary: native classifier, no input offload, ordinary nondeterministic default
attention, no cuBLAS environment override or capacity setter. Record actual
capacity (expected Ada default 8.125 MiB). Native: strict deterministic default
attention, environment :4096:8, explicit external workspace 8 MiB, native loss,
checkpoint-input CPU offload. Reuse: identical deterministic/offload policy,
H128 buffer classifier. All retain FP32, TF32 disabled, four CPU threads,
whole-block checkpoints, ordinary RMSNorm, resident gradients, original AdamW,
clipping and learning rate. Never adjust power/clock settings.

Reuse H120's full training source with a recorded textual derivation that only
changes instrumentation: remove forward/backward/clipping inventory barriers;
synchronize final optimizer event and stop timers before the post-step memory
mark; add whole-update CUDA elapsed time and wall timestamps. Keep per-phase
CUDA sums for description. The full timed interval covers loss, backward,
clipping, optimizer and offload/recompute, but excludes batch generation,
gradient clearing, validation, serialization and inventory checks. Whole-job
peak allocation includes all those phases. First ten updates are warmup.

Collect passive nvidia-smi every 200 ms per run; stop monitor in finally.
Require at least one telemetry sample in the timed steps 11-30 interval for
all cases. Retain all samples, unavailable values, timing means/medians/variance,
and first-half/second-half stability. No causal inference from clock/power.

Budget: 12x30=360 backwards and optimizer updates; 1,474,560 training targets;
24 full study validation scores plus 12 independent native audit scores;
36 tensor artifacts (raw/clipped first gradients, step801 and step830 states).
24 worker and 13 audit allocator boundaries must reach zero. Frozen maintained
files, fixture/data and source hashes must verify before and after work.

Independent inherited NumPy audit checks clipping, first Adam update/moments,
checkpoint state/hash/step and sampler replay. Native GPU audit scores every
final checkpoint. All deterministic native/reuse runs must match first/final
model/optimizer/sampler hashes, first raw/clipped gradients, all 30 loss/norm
pairs and before/after NLL bitwise. Ordinary may differ within original audit
and NLL tolerances; do not demand nondeterministic bytewise reproducibility.

For each corpus and control, report both candidate/control ratios. Acceptance
requires BOTH repeats: whole-job allocated memory ratio <=0.90, complete-update
CUDA and wall ratio <=1.15, validation NLL ratio <=1.01, candidate pinned host
peak <=128 MiB, and each case's halves timing ratio <=1.15. Require numerical
and telemetry checks. No dropping outliers, averaging away a failed repeat,
changing thresholds or retroactively rescuing H130/H132. No parameter or
breakthrough claim from this fixed-seed short continuation.

A complete pass earns a fresh multi-seed longer-training test with ordinary
controls; failure determines one narrower follow-up from measured evidence.
No chart is required; numerical tables and sealed raw results are primary.
