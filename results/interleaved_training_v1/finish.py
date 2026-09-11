"""Publish the paired experiment without modifying prior failed conclusions."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/interleaved_training_v1")
p, s, a = [read(ROOT / name) for name in ("protocol.json", "summary.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/timing_drift_audit_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "worker", "prepare_audit", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
status = (
    "PASS scoped interleaved timing gate" if s["passed"] else "FAIL scoped interleaved timing gate"
)
rows = []
pairrows = []
for c in s["comparisons"]:
    q = c["ratios"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    rows.append(
        f"| {c['dataset']} | {c['seed']} | {100 * (1 - q['memory']):.2f}% | {q['event']:.4f} | {q['wall']:.4f} | {failed} |"
    )
    for pair in c["pairs"]:
        pairrows.append(
            f"| {c['dataset']} | {c['seed']} | {pair['repeat']} | {pair['event']:.4f} | {pair['wall']:.4f} | {pair['nll']:.6f} |"
        )
metricrows = [
    f"| {m['index']} | {m['arm']} | {m['peak_mib']:.2f} | {m['event_ms']:.2f} | {m['stability']:.4f} | {m['setup_wall_ms']:.1f} | {m['segment_wall_ms']:.1f} | {m['sensors']['clock']} | {','.join(m['pstates'])} |"
    for m in s["metrics"]
]
aggregate = []
for row in s["aggregates"]:
    for key, v in row["statistics"].items():
        aggregate.append(
            f"| {row['dataset']} | {key} | {v['mean']:.6f} | {v['median']:.6f} | {v['sample_variance']:.6g} |"
        )
nextstep = (
    "Design an explicit opt-in maintained implementation with source-equivalence tests, preserving defaults, then qualify another workload scale. This pass concerns only balanced short segments at saved states; H138 remains failed and sustained deployment throughput remains unproven."
    if s["passed"]
    else "Do not promote the candidate. Inspect the retained failed gate before further GPU work. If device timing remains unstable, avoid repeating the same timing experiment; pursue an independent memory/FFN mechanism or a measurement change justified by the evidence."
)
report = f"""# H140: interleaved complete-training comparison

**{status}.** Independent numerical/native audit: **{a["passed"]}**.
H138's failed long-training qualification remains unchanged. No new activation,
parameter reduction, SOTA or broad-goal achievement is claimed.

| Corpus | Source seed | GPU allocation saved | Complete CUDA ratio | Wall ratio | Failed gates |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

All six ordinary H138 step800 model/Adam/sampler states are used. For each, four
independent30-update segments reset to that exact state, in ABBA or BAAB order
alternating by fixture. A=ordinary, B=buffer4. These are repeated continuations,
not120 updates of a continuous trajectory. Twenty updates per segment remain
after ten fixed warmups. Primary ratios compare all40 timed updates per arm.
Every seed must pass; descriptive aggregate means cannot rescue a failure.

## Neighboring temporal pairs

| Corpus | Source seed | Repeat | CUDA ratio | Wall ratio | Final native NLL ratio |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pairrows)}

The frozen gates are peak allocated memory<=0.90, complete CUDA/wall time<=1.15,
both repeat NLL ratios<=1.01, pinned hostpeak<=128MiB, every segment's half-median
stability<=1.15 and each arm's two-repeat median ratio<=1.15. Telemetry coverage
and numerical/native-score/initial-hash/batch-order checks are mandatory.

## Per-segment resources and setup

| Segment | Arm | GPU peak MiB | Complete CUDA ms | Within-segment stability | Setup ms | Entire segment ms | Median active SM MHz | P-states |
|---:|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(metricrows)}

Setup includes checkpoint/model/optimizer/data construction and initial hashes;
it is not isolated transfer time. Entire-segment wall time includes setup,
validation, saving, sampling and diagnostics; it excludes the following cleanup
boundary. `excluded_wall_ms` in the machine-readable result subtracts all30 update
wall times from that total. The primary training timer includes forward/backward,
clipping/Adam and candidate transfers/recomputation. First-step gradient copies
are in warmup only. No allocator inventories occur inside timed updates.
One model is resident at a time. Whole-job allocation includes setup, validation
and artifact operations, excludes driver/context/other processes; pinned host
memory is tracked separately and is not total CPU RSS. Reserved memory and all
phase inventories remain available in raw artifacts.

## Three-seed descriptive statistics

| Corpus | Ratio | Mean | Median | Sample variance |
|---|---|---:|---:|---:|
{chr(10).join(aggregate)}

All24 segments execute in one process to avoid repeated process startup. Passive
200ms telemetry is matched only to actual timed update intervals; missing sensors
remain missing. Short temporal pairing reduces separation but does not guarantee
identical device conditions. Neither clocks nor recorded times are normalized.
No hardware settings changed. Segment/repeat stability gates retain all data.

## Evidence and limits

720 updates/backwards,2,949,120 targets,48 study full-validation scores plus24
independent native scores,72 tensor artifacts,1,608 phase records and50 zero
allocator boundaries. Independent NumPy checks verify clipping, Adam parameters
and moments from the first actual step; native evaluation replays all720 batches
and scores each final state. Raw/clipped gradients are checked against ordinary
repeat noise with the unchanged capped tolerances. No extra audit backwards.
All9,099,648 parameters, ordinary FP32/TF32off policy,8.125MiB workspace, original
AdamW/LR/batch/context/checkpointing and fourCPUthreads remain fixed. Python is
UV-managed. An AST audit proves the reused H134 loop differs only in fixture seed
metadata and setup-wall instrumentation. Frozen input/source/library/maintained
hashes and independent audit artifacts are checked before publication.

This measures short complete-update segments at trained states. It does not
replace the H138 convergence study, show sustained deployment throughput, or
validate another scale/domain. The earlier failures are preserved. Original
maintained defaults are unchanged.

## Next decision

{nextstep}

[Prospective plan](interleaved_training_plan.md),
[summary](../results/interleaved_training_v1/summary.json),
[receipt](../results/interleaved_training_v1/receipt.json).
"""
reportpath = Path("research/interleaved_training_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, before in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (before + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if before == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: interleaved complete-training timing\n\n[H140](interleaved_training_results.md): **{status}.**\n720 updates across all six saved corpus/seed states; independent audit: {a['passed']}.\n\n{nextstep}\nThe broader VRAM/parameter-efficient FFN goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H140 interleaved timing](research/interleaved_training_results.md): **{status}.** H138 remains failed; no maintained default changes."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/interleaved_training_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H140",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=720,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
