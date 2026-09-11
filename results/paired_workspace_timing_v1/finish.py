"""Publish bounded evidence, retain prior failures, and seal the receipt."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_workspace_timing_v1")
p, s = read(ROOT / "protocol.json"), read(ROOT / "summary.json")
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/blas_workspace_mid_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "worker", "prepare_audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert all(
    (ROOT / (stage + "_exit.txt")).read_text().strip() == "3221225477"
    for stage in ("plot", "plot_recovery")
)
status = "PASS paired timing gate" if s["passed"] else "FAIL paired timing gate"
rows = []
for c in s["comparisons"]:
    e, w = c["ratios"]["event_ms"], c["ratios"]["wall_ms"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    rows.append(
        f"| {c['dataset']} | {c['control']} | {e['median']:.4f} | {e['minimum']:.4f}-{e['maximum']:.4f} | {w['median']:.4f} | {failed} |"
    )
pairs = [
    f"| {c['dataset']} | {c['control']} | {j + 1} | {q['ratios']['event_ms']:.4f} | {q['ratios']['wall_ms']:.4f} |"
    for c in s["comparisons"]
    for j, q in enumerate(c["pairs"])
]
bursts = []
for i, r in enumerate(s["bursts"]):
    t = r["telemetry"]

    def median(k):
        return f"{t[k]['median']:.1f}" if t[k] else "missing"

    bursts.append(
        f"| {i} | {r['dataset']} | {r['arm']} | {r['timing']['event_ms']['median']:.3f} | {r['timing']['wall_ms']['median']:.3f} | {r['timing']['cpu_ms']['median']:.3f} | {median('sm_mhz')} | {median('power_w')} | {median('temperature_c')} | {r['telemetry_samples']} |"
    )
next_step = (
    "Run a separately frozen complete-update comparison at the explicit 8-MiB capacity, using alternating candidate/control windows with telemetry. Include both ordinary execution and matched deterministic native controls, original clipping/Adam, native validation and peak GPU/host memory. Do not combine H132 memory with H133 timing to claim a complete-training win."
    if s["passed"]
    else "Do not extend long training. Inspect the paired dispersion and telemetry to specify one narrower runtime hypothesis; preserve this failed gate. If timing remains variable, a repeated balanced control-only experiment should establish measurement stability before another candidate claim."
)
report = f"""# H133: paired workspace timing with passive GPU telemetry

**{status}.** All 32 bursts match independently constructed native references:
loss, evaluation metadata and every gradient are bitwise identical at their
respective workspace capacity. Model, optimizer and sampler remain unchanged.

Candidate: buffer-reuse classifier, 8-MiB external cuBLAS workspace.
Both controls use the same CPU checkpoint-input offload and deterministic policy.
`native8` uses native loss at 8 MiB; `reuse32` uses buffer reuse at 32 MiB.
Each corpus/control comparison has ABBA then BAAB bursts; four adjacent,
order-balanced candidate/control ratios determine each median. Limit: 1.15.

| Corpus | Control | Median event ratio | Event ratio range | Median wall ratio | Failed gates |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

Median ratios span 0.9793-1.0118: no persistent >15% penalty appears in this
paired test. WikiText-2 burst medians stay near 74-76 ms. TinyStories includes
a 95-ms native burst and an 82-ms reuse32 burst, while most bursts remain
near 75-76 ms. The 95-ms native burst has a 2445-MHz median SM clock and
375-ms process CPU time; the adjacent native burst is 75 ms, 2422.5 MHz,
and 296.9-ms CPU time. Thus a simple low-SM-clock explanation is unsupported
by these samples. Host/dispatch variability is a hypothesis, not a diagnosed
cause. TinyStories/native paired ratios span 0.7862-1.0665; retain that
dispersion despite the passing median. Telemetry coverage is only 1-2 samples
per burst. The next training comparison must retain balanced ordering.

The optional figure is unavailable; complete paired ratios and burst telemetry appear below.

## Evidence and limits

260 backwards, zero optimizer updates, 36 one-batch BF16 evaluations; 1,064,960
diagnostic targets and 147,456 evaluation targets. Four independent native
reference constructions precede the two resident-model corpus experiments.
Seven allocator boundaries reach zero. All 61 maintained files are unchanged.
The CPU audit verifies every reference gradient artifact, raw JSONL equality,
frozen source/input hashes, exactness, budgets, order, timing ratios and telemetry.
Mean, median, variance and extrema are retained in the machine-readable summary.

Each burst has three warmups and five measured passes. Gradient resets occur
outside timing. CUDA-event and synchronized wall intervals include forward,
backward, recomputation and offload transfers. Allocation-inventory barriers
from H132 are absent. This is a different timing harness: cross-study absolute
times cannot isolate workspace or telemetry effects. Host process CPU time is
recorded separately; its sum across threads need not equal wall time.

Read-only nvidia-smi samples were collected every 200 ms in a hidden process
that was stopped after the experiment. Each burst must have at least one sample
in its measured interval. NVIDIA local timestamps are converted using the host
local timezone. Sensor readings have their own averaging windows and may lag.
Power-limit readings are unavailable, retained as missing. GPU clocks and power
were not changed. Correlation cannot prove throttling or identify kernel choice.
These data cannot retrospectively explain H132's unrecorded hardware state.

This is one fixed batch and one seed per corpus, with repeated measurements;
the four pairs are not independent training seeds. No complete-update timing,
quality, new memory reduction, parameter reduction or novelty claim follows.
H132's combined resource gate remains failed. Workspace selection uses the
existing [PyTorch API](https://docs.pytorch.org/docs/stable/backends).

## Every paired ratio

| Corpus | Control | Pair | Event ratio | Wall ratio |
|---|---|---:|---:|---:|
{chr(10).join(pairs)}

## Burst chronology and telemetry

CPU ms is process CPU time. Clock, power and temperature are medians of samples
whose timestamps fall between the first and last measured pass in that burst.

| Burst | Corpus | Arm | Event ms | Wall ms | CPU ms | SM MHz | W | Celsius | Samples |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(bursts)}

The first CPU figure-rendering process failed with a Windows access violation
during Matplotlib import (exit 3221225477). Its log is preserved. One bounded
CPU-only recovery also failed during imports with the same exit code. Both logs
are retained and the optional chart is omitted. No scientific probe was repeated.

## Next decision

{next_step}
The broad VRAM/quality and parameter-efficient FFN research goal remains open.

[Prospective plan](paired_workspace_timing_plan.md),
[raw and summarized evidence](../results/paired_workspace_timing_v1/summary.json),
[receipt](../results/paired_workspace_timing_v1/receipt.json).
"""
report_path = Path("research/paired_workspace_timing_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + f"\n\n## Latest: balanced paired workspace timing\n\n[H133](paired_workspace_timing_results.md): **{status}.**\nAll 32 bursts match fresh native gradient references bitwise.\n260 backwards, zero updates, passive GPU telemetry; independent CPU audit verified.\n\n{next_step}\nEarlier gates and the broad research goal remain open.\n\n"
    + rest,
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("Latest:", "Earlier:", 1)
readme.write_text(
    head
    + f"\n\nLatest: [paired workspace timing](research/paired_workspace_timing_results.md).\n**{status}.** Balanced probes and passive GPU telemetry; exact native gradients.\nThis is timing evidence only; the full VRAM/quality research goal remains open.\n\n"
    + rest,
    encoding="utf-8",
    newline="\n",
)
files = [
    f
    for f in ROOT.iterdir()
    if f.is_file() and f.name not in ("finish.log", "finish_exit.txt", "receipt.json")
]
files += [
    report_path,
    Path("research/paired_workspace_timing_plan.md"),
    current,
    readme,
]
receipt = dict(
    study="H133",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    backwards=260,
    updates=0,
    files={f.as_posix(): sha(f) for f in files},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
