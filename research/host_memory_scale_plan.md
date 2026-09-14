# H162: total host-memory qualification at larger model scale

H161's three seeds passed its numeric/resource smoke gates, but process RSS was
not recorded. H162 repeats its six fresh-process30-update cases for a prospective
host-memory measurement, not to replace any H161 outcome. Budget180 updates and
six initial backwards. No new datasets, architecture changes or long-quality claim.

Reuse the unchanged H161 training/evaluation loop and audit, with its verified
post-measurement cuBLAS cleanup. Same three seeds401/409/419, arm orderAB/BA/AB,
22,163,968 parameters, FP32/TF32off, batch16/context512, 12 blocks, four-block exact
CPU offload plus buffered classifier loss versus native. Each process contains
only one model. Preserve all checkpoints, gradients, histories and telemetry.

Call Windows GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX) before importing
torch, after importing torch/cleanup adapter, and after the worker completes.
Record current and lifetime peak working set, private commit/current and lifetime
peak commit. Peaks include imports, initial/final validation, CPU serialization,
training and cleanup through the last sample. They are not reset or baseline
subtracted; startup can dominate. No sampling thread is required for OS high-water
counters. These are per-worker process measurements, not system RAM consumption;
shared pages, kernel/driver allocations and other processes limit interpretation.

Working set is resident memory; private commit is a distinct memory-manager
reservation/accounting measure, not physical RAM or bytes written to a pagefile.
[Microsoft counter definitions](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex).
Validate the native struct/API first with a touched64MiB CPU allocation, requiring
at least32MiB rises in current working set/private commit and nondecreasing peaks.
This check has zero GPU work. Freeze source, API check and prior evidence hashes
before any model execution.

Every seed must pass all unchanged H161 numerical/GPU/timing/short-NLL gates.
Additional host gates: helper peak working set minus paired native<=256MiB AND
helper peak private commit minus native<=256MiB. Both arm peaks must be<=8GiB.
Pinned allocation remains<=128MiB. These are deliberately explicit host budgets,
not claims of host-memory savings. Record absolute bytes and differences. Do not
clamp negative differences, select a favorable seed or subtract startup afterward.

Audit saved initial gradients using NumPy FP64 and reload every final model for
native full-classifier scoring, using the unchanged tolerances. Verify API fields,
monotonic lifetime peaks across snapshots, counters and artifact hashes. Report
mean/median/sample variance over all three seeds for both host peaks. No retry
may replace a completed case; preserve execution failures and stop for diagnosis.
A pass closes this measurement gap only for the short workload. It earns planning
sustained quality/throughput testing; it does not retroactively fill H161's missing
RSS measurements, establish architecture novelty, or complete the broad goal.
