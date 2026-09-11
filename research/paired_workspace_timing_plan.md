# H133: paired workspace timing with passive GPU telemetry

Question: does the 8-MiB buffer classifier have a persistent runtime penalty
under balanced nearby comparisons, or were sequential measurements unstable?
This tests timing only; it cannot rescue H132's failed combined resource gate.

Use unchanged H121 step-800 native fixtures, WikiText-2 then TinyStories,
FP32, strict deterministic default SDPA, TF32 off, four CPU threads,
CUBLAS_WORKSPACE_CONFIG=:4096:8. Explicit external workspace: 8 or 32 MiB.
All arms use H121 checkpoint-input CPU offload, resident gradients and native
RMSNorm. Candidate A is H128 buffer reuse at 8 MiB. Compare separately against
B=native at 8 MiB, then B=buffer reuse at 32 MiB. No optimizer updates.

Per comparison, execute ABBA then BAAB. Each of 32 bursts clears gradients
and old cuBLAS workspaces, sets capacity, warms one BF16 validation batch,
then performs three warmup and five timed forward/backward passes. Use fixed
8x512 tokens throughout. Reset gradients outside timing; no allocation
inventory, hashing or phase barriers inside timers. CUDA events and synchronized
wall time cover forward+backward including checkpoint input transfers. Also
record host process CPU time. Summaries exclude three warmups.

Four fresh native references (two corpora x two capacities), each with its own
model construction, one evaluation and one backward, precede corpus probes.
Compare every burst's final loss, evaluation metadata and every gradient hash
bitwise with its workspace-specific reference. Check model/optimizer/sampler
unchanged. Save four reference gradients and all raw timings. Budget exactly
260 backwards, zero updates, 36 one-batch evaluations (147456 targets),
1064960 diagnostic targets. Seven allocator boundaries must reach zero.

Run read-only nvidia-smi telemetry every 200 ms in a hidden subprocess; capture
local timestamp, UUID, pstate, SM/memory clocks, temperature, power draw/limit
and utilization. Stop the subprocess in finally. Record wall-clock/timezone
anchors. N/A remains missing. No clock/power setting changes. Associate samples
with each burst's measured interval; at least one sample per burst is required
for telemetry validity. Telemetry is descriptive, not evidence of causation.

Adjacent burst pairs (0,1) and (2,3) in each cycle give four oriented A/B ratios
per comparison. Acceptance requires exactness, complete budget, frozen hashes,
zero boundaries, telemetry coverage, and median paired CUDA-event and wall
ratios <=1.15 for every comparison. Report all ratios plus mean, median,
variance, extrema; never discard slow pairs or change this gate. Clock/power
correlations do not establish a cause. This is one seed/fixed batch per corpus,
not an independent-seed estimate or complete training test.

Passing earns a separately frozen complete-update experiment with ordinary
and matched deterministic controls. Failure keeps long training paused and
identifies whether persistent penalty or unresolved variability remains.
Prior failures are retained. No activation or algorithm novelty is claimed.
