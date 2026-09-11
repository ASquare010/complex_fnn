# H138: fresh validation of four-block offload

Previous turn: progress. H137 selected buffer4 after all short gates passed:
18.7-19.0% lower GPU allocation with5.2-5.5% runtime overhead. H136's full-offload
long-run failure remains unchanged. This test must establish fresh convergence.

Run H117's six step-zero fixtures (WikiText-2/TinyStories, seeds101,113,127), each
for800 updates under ordinary native loss/no offload and H128 buffer loss with
only blocks4-7 offloaded by H137's unchanged adapter. Reuse H136's exact long
training loop and independent audit. There are no other architectural or
numerical changes. Parameters remain9,099,648. FP32, TF32off, ordinary attention,
default8.125-MiB workspace, four CPU threads, PYTHONMALLOC=pymalloc, UV-managed
Python, original AdamW/LR0.0006/clipping1, batch8/context512 and whole-block
checkpointing remain fixed. No clock or power adjustments.

For fixtures in existing corpus/seed order, alternate ordinary/buffer4 then
buffer4/ordinary. Three pairs have each ordering, balancing each seed's position
across the two corpora. Freeze this order before execution; never select a faster
repeat or rerun completed work. Retain all seeds even if one gate fails.

Budget12x800=9,600 updates,39,321,600 training targets. Twelve initial gradient
probes and twelve independent native replays give9,624 total backwards.
48 study scores plus42 independent native scores (six initial +36 saved states).
48 tensor artifacts;24 worker +19 audit zero allocator boundaries. Free disk
checked before launch. All model/optimizer/sampler checkpoints and histories
remain available; compact atomic JSON plus raw-case aggregation preserve values.

EVERY corpus/seed must pass versus its ordinary control: whole-job allocation
ratio<=0.90, complete CUDA/wall median ratio<=1.15, final validation NLL ratio
<=1.01, candidate pinned-host peak<=128MiB. Each run's three260-update wall-mean
blocks after20 warmups must have max/min<=1.15. Require finite model/gradients/
moments/layer diagnostics, matching initial states/batch order, selected block
indices, passive telemetry coverage and the complete independent audit. Report
mean/median/sample variance across all three seeds; aggregates cannot rescue a
failed individual seed. No threshold changes from H136/H137.

Native audit regenerates six empty-Adam initial states exactly, recomputes first
gradients without custom loss/offload, replays all batches, checks saved state/
sampler hashes at200/400/800 and scores all endpoints. Existing numerical caps:
global gradient1e-5, tensor1e-4, loss/score relative error1e-6. Ordinary attention
is nondeterministic; no bitwise trajectory claim. Initial native operator proof
and H137's noise calibration remain supporting evidence, not audit substitutes.

Timing includes forward/backward/clipping/Adam/recomputation/transfers, excluding
sampling, gradient clearing, validation, serialization and inventory. Whole-job
CUDA peak includes those phases but excludes driver/context/other processes.
Pinned memory is reported separately and is not total CPU RSS. Read-only
nvidia-smi telemetry every200ms, stopped in finally; missing sensors stay missing.
Clock/power correlations are descriptive. All source/input/library hashes verify.

Stop the driver on any process failure. Before any bounded recovery, inspect
existing artifacts and live handles; do not restart completed or still-running
work. A full pass earns opt-in maintained integration and verification at another
scale/workload. It does not prove novel activation geometry, parameter savings,
SOTA superiority or completion of the broader FFN research goal.
