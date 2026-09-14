# H162: process host memory at the larger scale

**PASS short-workload host qualification.** Six fresh worker processes, three seeds,180 updates/186 backwards.
This supplies a new host-memory measurement; H161's absent RSS remains absent.
The broad research goal and sustained quality/throughput qualification stay open.

Same22,163,968-parameter FP32 Transformer and original H161 loop: width512,
hidden608,12 layers, batch16/context512, WikiText2. Compare native loss with
unchanged buffered loss plus last-four-block exact input offload. Every seed uses
identical initialization and batches; orderAB/BA/AB. No parameter savings or
activation novelty is claimed. All six gradients and final model scores were
audited with the original numerical tolerances and native final-state replay.

## Total worker host memory

All memory entries below are MiB. Differences are helper minus ordinary.

| Seed | Native peak WS | Helper peak WS | Difference | Native peak commit | Helper peak commit | Difference | Failed qualification gates |
|---|---:|---:|---:|---:|---:|---:|---|
| 401 | 1571.81 | 1637.18 | +65.37 | 4097.57 | 3984.84 | -112.73 | none |
| 409 | 1572.89 | 1638.03 | +65.14 | 4091.68 | 3981.54 | -110.15 | none |
| 419 | 1572.77 | 1637.98 | +65.20 | 4096.88 | 3985.15 | -111.73 | none |

Windows working set measures resident process memory; private commit is a
separate memory-manager accounting measure, not physical RAM or pagefile writes.
Both peaks are OS lifetime high-water counters read after the worker returns;
they include imports, validation, training, CPU checkpoint serialization and
cleanup through that sample. Startup is not subtracted and can dominate.
[Microsoft counter definitions](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex).

These counters describe the worker, not whole-system RAM. Shared pages, memory
compression/trimming, kernel/driver allocations and other processes prevent
summing this table into exact system memory use. The final host JSON write and
interpreter exit occur after the final sample. Pinned allocator statistics remain
separate in each case result. No sampling thread perturbs the training loop.

| Arm | Counter | Mean MiB | Median MiB | Sample variance MiB^2 |
|---|---|---:|---:|---:|
| ordinary | working_set | 1572.49 | 1572.77 | 0.354 |
| ordinary | private_commit | 4095.38 | 4096.88 | 10.354 |
| helper | working_set | 1637.73 | 1637.98 | 0.229 |
| helper | private_commit | 3983.84 | 3984.84 | 4.013 |

## GPU, timing and short loss check

| Seed | Allocated GPU saved | CUDA ratio | Wall ratio | Final NLL ratio | Failed original gates |
|---|---:|---:|---:|---:|---|
| 401 | 18.10% | 1.0607 | 1.0607 | 1.0000001 | none |
| 409 | 18.10% | 1.0172 | 1.0172 | 1.0000055 | none |
| 419 | 18.10% | 1.0522 | 1.0522 | 1.0000000 | none |

Every seed must satisfy the original H161 checks plus new host budgets: helper
peak working-set and private-commit increases each<=256MiB, and both arms' peaks
<=8GiB. Pinned allocation stays<=128MiB. Negative differences are retained;
no selected seed, average or baseline subtraction rescues a failing gate.
GPU peaks include initial/final full validation and complete optimizer updates.
Full per-case histories and timing statistics are retained, including all failures.

## Verification and decision

Before GPU work, a separate64MiB touched CPU allocation increased current working
set and private commit by at least32MiB, and verified monotonic peaks. The native
80-byte Windows struct and API result are checked. Protocol/code/data and prior
receipt hashes were frozen; initial and final model hashes, batch sequences,
finite states, saved gradient hashes and native final scores were audited.
Every worker and audit model released GPU allocations/reservations to zero using
the already-qualified cuBLAS cleanup. This study had no execution retry.

This closes the worker host-memory measurement gap for this short, fixed workload. It supports a bounded sustained qualification at this scale, including host counters and an uninterrupted session. It does not establish long-run host peaks or quality.

This established offload technique is an engineering baseline for the wider
architectural research. A breakthrough still requires stronger quality/resource
results across unrelated tasks; the present30-update comparison cannot supply them.

[Prospective plan](host_memory_scale_plan.md),
[host qualification](../results/host_memory_scale_v1/qualification.json),
[native and gradient audit](../results/host_memory_scale_v1/summary.json),
[CPU API check](../results/host_memory_scale_v1/api_check.json).
