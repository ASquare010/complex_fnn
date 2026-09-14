"""Publish H162's prospective host budgets with explicit measurement semantics."""

from pathlib import Path

from results.exact_offload_scale_v1.study import read, sha, write

ROOT = Path("results/host_memory_scale_v1")
for stage in ("run", "audit"):
    assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
p, q, s = [read(ROOT / name) for name in ("protocol.json", "qualification.json", "summary.json")]
for path, digest in p["hashes"].items():
    assert sha(path) == digest
rows, stats, gpu = [], [], []
for pair in q["pairs"]:
    seed = pair["seed"]
    a = next(r for r in q["peaks"] if r["seed"] == seed and r["arm"] == "ordinary")
    b = next(r for r in q["peaks"] if r["seed"] == seed and r["arm"] == "helper")
    d = pair["host_delta_bytes"]
    failed = ", ".join(k for k, v in pair["gates"].items() if not v) or "none"
    rows.append(
        f"| {seed} | {a['working_set'] / 2**20:.2f} | {b['working_set'] / 2**20:.2f} | {d['working_set'] / 2**20:+.2f} | {a['private_commit'] / 2**20:.2f} | {b['private_commit'] / 2**20:.2f} | {d['private_commit'] / 2**20:+.2f} | {failed} |"
    )
for arm, metrics in q["statistics"].items():
    for key, v in metrics.items():
        stats.append(
            f"| {arm} | {key} | {v['mean'] / 2**20:.2f} | {v['median'] / 2**20:.2f} | {v['sample_variance'] / 2**40:.3f} |"
        )
for pair in s["pairs"]:
    r = pair["ratios"]
    failed = ", ".join(k for k, v in pair["gates"].items() if not v) or "none"
    gpu.append(
        f"| {pair['seed']} | {100 * (1 - r['memory']):.2f}% | {r['cuda']:.4f} | {r['wall']:.4f} | {r['nll']:.7f} | {failed} |"
    )
status = (
    "PASS short-workload host qualification"
    if q["passed"]
    else "FAIL short-workload host qualification"
)
report = f"""# H162: process host memory at the larger scale

**{status}.** Six fresh worker processes, three seeds,180 updates/186 backwards.
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
{chr(10).join(rows)}

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
{chr(10).join(stats)}

## GPU, timing and short loss check

| Seed | Allocated GPU saved | CUDA ratio | Wall ratio | Final NLL ratio | Failed original gates |
|---|---:|---:|---:|---:|---|
{chr(10).join(gpu)}

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

{"This closes the worker host-memory measurement gap for this short, fixed workload. It supports a bounded sustained qualification at this scale, including host counters and an uninterrupted session. It does not establish long-run host peaks or quality." if q["passed"] else "Do not promote this workload beyond the measured passing subchecks. Retain the failed host or original gates and diagnose them before a longer study."}

This established offload technique is an engineering baseline for the wider
architectural research. A breakthrough still requires stronger quality/resource
results across unrelated tasks; the present30-update comparison cannot supply them.

[Prospective plan](host_memory_scale_plan.md),
[host qualification](../results/host_memory_scale_v1/qualification.json),
[native and gradient audit](../results/host_memory_scale_v1/summary.json),
[CPU API check](../results/host_memory_scale_v1/api_check.json).
"""
path = Path("research/host_memory_scale_results.md")
assert not path.exists()
path.write_text(report, encoding="utf-8")
for target, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = target.read_text(encoding="utf-8")
    before = ROOT / (
        "README.before.md" if target.name == "README.md" else "CURRENT_STATE.before.md"
    )
    with before.open("x", encoding="utf-8") as f:
        f.write(old)
    assert old.startswith(title)
    rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    link = (
        "research/host_memory_scale_results.md"
        if target.name == "README.md"
        else "host_memory_scale_results.md"
    )
    target.write_text(
        title
        + f"\n\nLatest: [H162 worker host-memory test]({link}): **{status}.**\nSix fresh processes/180 updates with Windows peak working-set and private-commit counters.\nSustained quality and the broad research goal remain open.\n\n"
        + rest,
        encoding="utf-8",
    )
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "publish.log", "publish_exit.txt")
]
files += [
    path,
    Path("research/host_memory_scale_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write(
    ROOT / "receipt.json",
    dict(
        status=status,
        passed=q["passed"],
        optimizer_updates=180,
        backwards=186,
        goal_achieved=False,
        hashes={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
