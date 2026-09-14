"""Publish H161 without widening the scope of its short measurements."""

import statistics as st
from pathlib import Path

from results.exact_offload_scale_v1.study import read, sha, write

ROOT = Path("results/exact_offload_scale_v1")
R = ROOT / "recovery"
s = read(R / "summary.json")
assert (R / "audit_exit.txt").read_text().strip() == "0"
assert (R / "run_exit.txt").read_text().strip() == "0"
for p in (read(ROOT / "protocol.json"), read(ROOT / "recovery_protocol.json")):
    for path, digest in p["hashes"].items():
        assert sha(path) == digest
rows = []
for pair in s["pairs"]:
    ratio = pair["ratios"]
    failed = ", ".join(k for k, ok in pair["gates"].items() if not ok) or "none"
    rows.append(
        f"| {pair['seed']} | {100 * (1 - ratio['memory']):.2f}% | {ratio['cuda']:.4f} | {ratio['wall']:.4f} | {ratio['nll']:.6f} | {failed} |"
    )
statistics = {}
stat_rows = []
for arm in ("ordinary", "helper"):
    values = [r["final_score"]["nll"] for r in s["runs"] if r["arm"] == arm]
    statistics[arm] = dict(
        mean=st.mean(values), median=st.median(values), sample_variance=st.variance(values), seeds=3
    )
    stat_rows.append(
        f"| {arm} | {st.mean(values):.8f} | {st.median(values):.8f} | {st.variance(values):.8g} |"
    )
write(ROOT / "seed_statistics.json", statistics)
raw = []
for run in s["runs"]:
    raw.append(
        f"| {run['seed']} | {run['arm']} | {run['memory']['allocated'] / 2**20:.2f} | {run['memory']['reserved'] / 2**20:.2f} | {run['memory']['host']['allocated_bytes.peak'] / 2**20:.2f} | {run['timing']['cuda_ms']['median']:.2f} | {run['final_score']['nll']:.6f} |"
    )
status = "PASS numeric/resource smoke gates" if s["passed"] else "FAIL numeric/resource smoke gates"
report = f"""# H161: larger-model exact memory-helper probe

**{status}.** Process RSS promised by the plan was not collected; full measurement
coverage is therefore incomplete. Pinned allocator counters do not measure total
process RAM. No long-training quality or breakthrough qualification is claimed.

The same 22,163,968-parameter FP32 Transformer is used in both arms (2.44x H156's
9,099,648 parameters). Width512, hidden608, 12 layers, eight heads, context512,
batch16, WikiText2 only. Compare native classifier loss with unchanged buffered
loss and last-four-block CPU checkpoint-input offload; whole-block recomputation
is enabled in both. No learned activation or parameter reduction is introduced.

| Seed | Allocated GPU saved | CUDA update ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

| Seed | Arm | Peak allocated MiB | Peak reserved MiB | Pinned peak MiB | Median CUDA ms | Final NLL |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(raw)}

Across-seed final validation NLL (three seeds; descriptive statistics):

| Arm | Mean | Median | Sample variance |
|---|---:|---:|---:|
{chr(10).join(stat_rows)}

## Evidence and interpretation

Six fresh30-update runs, three seeds, 180 accepted updates/186 backwards. Every
initial model hash and paired batch sequence matches. All final states are
reloaded and independently scored using native full-classifier cross entropy;
the measured loop uses FP32 chunked classification for full validation. Initial
full gradients are compared in NumPy FP64 under the original exact tolerances.
Memory includes model/data construction, initial probe, both full validation
passes and all optimizer updates. Per-update forward/backward/optimizer timing,
loss, gradient norm, and mean/median/sample variance are in the raw summary.
All six retained model checkpoints have unchanged parameter count.

The fixed gates are >=10% allocated-memory saving, <=15% median CUDA/wall
slowdown, <=1% final-NLL increase, <=128MiB pinned allocation, and the planned
within-run stability/interruption limits, plus numerical audits. Thirty updates
only screen execution and resource behavior; they cannot establish convergence,
robust speed across sessions, generalization quality, or unrelated-task benefit.
No selected seed is hidden and no aggregate rescues a failing seed.

## Failure and repair accounting

The original case00 completed30 updates but failed post-run cleanup before its
resource summary was written. Its gradients, final state, history and logs remain
untouched. A separate one-matrix, zero-training diagnostic reproduced8.125MiB of
retained cuBLAS workspace after empty_cache; clearing the workspace released it.
Recovery changes only post-measurement cleanup through a scoped process adapter,
as in the repository's existing boundary helper. All original source hashes,
measurements and gates remain intact. Recovery uses a separate output folder.
Total spent:210 updates/217 backwards, plus that one diagnostic matrix product.
The missing first-attempt resource summary cannot be reconstructed and is not used.

## Mechanism and next decision

At FP32, weights, gradients and two Adam moment arrays alone require16P bytes
when resident. Here that is338.20MiB, versus138.85MiB at H156 scale. Four saved
block inputs contain4*B*T*d FP32 values:64MiB here versus48MiB before. Thus fixed
checkpoint-input offload does not scale quadratically with the parameter state;
classifier-buffer savings and the actual peak phase also determine the result.
This accounting explains why width/parameter counts cannot substitute for measured
whole-job memory. It is not a universal lower bound for other optimizer/storage
schemes, and component savings do not necessarily add at the measured peak.

{"The recorded numeric/resource gates support a later sustained test at this scale. Repair missing RSS instrumentation before extending claims; do not promote short NLL as quality evidence." if s["passed"] else "Do not extend qualification to this scale. Diagnose the specific failed gates before spending on another long run; retain H156 only within its previously qualified scope."}
The broader architectural and complex-pattern research goal remains open.

[Prospective plan](exact_offload_scale_plan.md),
[cleanup recovery](exact_offload_scale_recovery_plan.md),
[machine-readable evidence](../results/exact_offload_scale_v1/recovery/summary.json).
"""
report_path = Path("research/exact_offload_scale_results.md")
assert not report_path.exists()
report_path.write_text(report, encoding="utf-8")
for path, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = path.read_text(encoding="utf-8")
    before = ROOT / ("README.before.md" if path.name == "README.md" else "CURRENT_STATE.before.md")
    with before.open("x", encoding="utf-8") as f:
        f.write(old)
    assert old.startswith(title)
    rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    link = (
        "research/exact_offload_scale_results.md"
        if path.name == "README.md"
        else "exact_offload_scale_results.md"
    )
    path.write_text(
        title
        + f"\n\nLatest: [H161 larger-model probe]({link}): **{status}.**\nThree seeds, 180 accepted updates, exact-gradient and native-score checks.\nRSS coverage incomplete; sustained quality and the broad research goal remain open.\n\n"
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
    report_path,
    Path("research/exact_offload_scale_plan.md"),
    Path("research/exact_offload_scale_recovery_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write(
    ROOT / "receipt.json",
    dict(
        status=status,
        gate_passed=s["passed"],
        measurement_coverage_complete=False,
        accepted_updates=180,
        total_updates=210,
        goal_achieved=False,
        hashes={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
