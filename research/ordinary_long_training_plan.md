# H136: fresh three-seed ordinary-policy validation

Previous turn: progress, H135 passed short-training gates. Its 24.6-25.0% memory
saving does not establish fresh convergence. Keep every historical failure.

Use all six original step-zero fixtures from H117: WikiText-2/TinyStories,
seeds101,113,127, independently regenerated during audit. Train each for800
updates under three arms: ordinary(native loss, GPU checkpoint inputs),
native(native loss, CPU input offload), reuse(H128 buffer loss, CPU input
offload). All use ordinary/default attention, deterministic=False, cudnn
benchmark/deterministic=False, default8.125-MiB workspace, no workspace override,
FP32, TF32off, four CPU threads, UV-managed Python, PYTHONMALLOC=pymalloc.
Original AdamW, LR0.0006, clipping1, batch8, context512 and model9,099,648
parameters remain identical. No GPU power/clock settings change.

Balance arm order with a Latin square across seeds: ordinary/native/reuse,
native/reuse/ordinary, reuse/ordinary/native. Reverse each order for TinyStories.
Reusing step-zero files means fresh initialization, not continuation of learned
weights. Audit must regenerate all six states exactly.

Reuse H117's long-training loop with only wall timestamp fields added to each
update for telemetry association. Complete CUDA and synchronized wall timers
already span forward/backward/clipping/Adam and exclude phase inventory.
Use H135's qualified loss and checkpoint-input offload; native RMSNorm and
resident gradients. Compact JSON serialization is an operational precaution
against previously observed Python encoder crashes; numerical content unchanged.
Store per-step losses/grad norms/timing/memory, layer diagnostics, full validation
at0/200/400/800 and model/optimizer/sampler checkpoints at200/400/800.

Budget18x800=14,400 updates,58,982,400 training targets.18 initial gradient
probes give14,418 study backwards. Independent native replay adds18:14,436 total.
72 study scores +60 independent native scores (six initial and54 checkpoints).
72 tensor artifacts,36 worker +25 audit zero allocator boundaries. About7GB
new artifacts; free disk checked before launch. No early termination for a
single unfavorable result; report all seeds under the frozen budget.

For EVERY corpus/seed, reuse must pass against BOTH controls: whole-job allocated
memory ratio<=0.90; complete-update CUDA/wall median ratio<=1.15; final validation
NLL ratio<=1.01; pinned host peak<=128MiB. Each run's three260-update measured
blocks (after20warmups) must have max/min mean wall time<=1.15. Also require
telemetry samples during the measured-update window, finite parameters/gradients/
moments/layer diagnostics, matching batch and initial-state hashes, source hashes,
and independent audit. Aggregate mean/median/sample variance per corpus/arm and
paired ratios across all three seeds, but do not average away a failed seed.

Independent H117 audit regenerates initial state, replays every batch and checks
model/optimizer/sampler hashes at200/400/800, scores all endpoints through native
code, and recomputes initial gradients without custom loss/offload. Gradient
relative L2 caps1e-5 global/1e-4 tensor; loss/score relative error<=1e-6. This is
ordinary nondeterministic attention: do not claim bitwise trajectory equality.
H135's noise calibration and H128's exact operator evidence remain precedents,
not replacements for fresh replay. No new gradient tolerance is selected here.

Sample passive nvidia-smi at200ms per worker, timestamps/timezone recorded; stop
monitor in finally. Collect clocks/power/temperature; missing values stay missing.
Samples are descriptive, not a causal hardware diagnosis. Runtime limits include
recomputation/transfers; whole-job CUDA peak includes evaluation/serialization,
but excludes driver/context/other processes. Pinned memory is not total host RSS.

Run sequentially; stop driver if a process fails. Never restart a completed run.
Any recovery must first inspect authoritative artifacts and record a bounded
prospective recovery protocol. An import failure and a mid-training failure
require different handling; absence of an observer response is not a failure.

A full pass supports an ordinary-policy memory implementation on these two
corpora, with explicit runtime/host tradeoffs. It does not establish a novel
activation, parameter reduction, SOTA superiority or the broad research goal.
Afterward evaluate maintained integration and another workload scale; preserve
separate structural FFN/activation research requirements.
