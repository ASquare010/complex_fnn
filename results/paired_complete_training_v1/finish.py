"""Publish full-update evidence without promoting a short run to a broad claim."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_complete_training_v1")
p, s = read(ROOT / "protocol.json"), read(ROOT / "summary.json")
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
hashes(json.loads((ROOT / "recovery_protocol.json").read_text(encoding="utf-8-sig"))["files"])
for path, digest in read("results/paired_workspace_timing_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("recovery_prepare_audit", "audit", "analyze"):
    assert (ROOT / ("recovery_" + stage + "_exit.txt")).read_text().strip() == "0"
status = "PASS complete short-training gate" if s["passed"] else "FAIL complete short-training gate"
comparisons = []
for c in s["comparisons"]:
    q = c["ratios"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {c['dataset']} | {c['repeat']} | {c['control']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f} | {q['wall_ms']:.4f} | {q['nll']:.8f} | {failed} |"
    )
metrics = []
for m in s["metrics"]:
    metrics.append(
        f"| {m['index']} | {m['dataset']} | {m['arm']} | {m['peak_mib']:.3f} | {m['host_mib']:.3f} | {m['event_ms']:.3f} | {m['wall_ms']:.3f} | {m['nll']:.7f} | {m['stability']:.4f} | {m['telemetry_samples']} |"
    )
next_step = (
    "Run fresh longer training with multiple independent seeds and ordinary controls. Preserve this exact workspace/offload/classifier configuration and matched initialization/data order, with telemetry and complete-update timing. Establish full-run quality before claiming a useful architecture-independent memory optimization. Parameter-efficient FFN architecture research remains separately unproven."
    if s["passed"]
    else "Keep longer training paused. Use the failed gates and retained timing/telemetry to choose one narrower falsifiable follow-up; do not relax thresholds, drop a repeat or merge this result with H133's timing pass."
)
if (
    not s["passed"]
    and s["audit_passed"]
    and all(c["passed"] for c in s["comparisons"] if c["control"] == "native")
):
    next_step = "Test the same candidate under ordinary/default attention policy, alongside native loss with and without checkpoint-input offload. Calibrate full-model numerical tolerances against repeated ordinary native controls, retaining the existing exact operator qualification. This isolates whether strict deterministic execution is the remaining practical runtime cost; do not assume it is the cause. Keep paired complete-update timing, native scoring and memory gates, and do not extend long training yet."
report = f"""# H134: complete training with an 8-MiB workspace

**{status}.** Independent numerical/native-score audit: **{s["audit_passed"]}**.
All eight corpus/control/repeat comparisons below must pass individually.
These are short continuations from trained step-800 fixtures, not fresh training
or a multi-seed quality result. Prior failed gates remain recorded.

| Corpus | Repeat | Control | GPU allocation saved | Complete CUDA ratio | Wall ratio | NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

Candidate is H128 buffer-reuse classifier plus checkpoint-input CPU offload,
strict deterministic default attention and explicit 8-MiB external cuBLAS
workspaces. Matched native uses the same configuration with native loss.
Ordinary uses native loss, GPU-resident checkpoint inputs and ordinary default
execution (observed 8.125-MiB workspace). Parameters remain 9,099,648.
This uses the existing [PyTorch workspace API](https://docs.pytorch.org/docs/stable/backends);
it is not a novel activation or FFN architecture.

## Runs and measured scope

Each corpus executes reuse, ordinary, native, native, ordinary, reuse in fresh
processes. Both mirrored repeats must meet memory <=0.90, complete-update CUDA
and wall <=1.15, NLL <=1.01, candidate pinned host peak <=128 MiB, and within-run
halves timing ratio <=1.15. No best-run selection or averaging away failures.
Repeats share one fixture/seed; they do not estimate independent-seed variance.

| Run | Corpus | Arm | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing halves ratio | Telemetry samples |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Whole-update CUDA elapsed time spans forward, backward, clipping and Adam,
including checkpoint recomputation and offload transfers. Synchronized wall
time ends before post-step memory inventory. Timing excludes batch generation,
gradient clearing, validation, serialization and inventory, while reported
whole-job peak CUDA allocation includes them. First ten updates are warmup;
all raw steps, means, medians, sample variance and timing stability are retained.
Reserved CUDA and pinned-host peaks are reported separately in machine-readable
results. CUDA allocation excludes driver/context and other processes; pinned
host peak is not total CPU RSS. This harness differs from H130's phase-barrier
instrumentation, so no cross-study absolute-runtime causal claim is valid.

The loop is derived from H120 by asserted textual instrumentation edits only:
remove three phase inventory barriers and stop whole-update timers before the
remaining post-step inventory. Optimizer, clipping, data, checkpointing and
validation numerical operations are unchanged. The derived source is frozen. An independent syntax-tree check reconstructs
the declared edits in memory and verifies exact agreement after formatting.

## Verification and telemetry

360 optimizer updates/backwards, 1,474,560 training targets, 24 full study
validation scores and 12 independent native scores. There are 36 tensor
artifacts, 804 memory intervals, and 37 zero allocator boundaries including
the independent audit. All 61 maintained source/config/test files are unchanged.

Inherited NumPy checks verify original clipping and first Adam updates/moments.
Every final checkpoint is scored through independent native code and all batch
hashes are replayed. Native/reuse deterministic runs are compared bitwise for
first/final model, optimizer and sampler state, first raw/clipped gradients,
all 30 loss/norm pairs and before/after validation. The audit result above
includes these checks. Ordinary execution is audited under original numerical
and validation tolerances, not an invalid bytewise reproducibility requirement.

Read-only GPU telemetry samples every 200 ms; every monitor is stopped in a
finally block. Run-level clock, power and temperature ranges/medians and raw
samples are retained. Missing sensors stay missing. Coverage is required within
updates 11-30; this is sparse observational telemetry, not proof of a throttling
or kernel-selection cause. No power or clock settings were changed.

The initial launch crashed during NumPy import with Windows access violation
3221225477, before any case artifacts or updates. The driver stopped. A frozen
recovery plan switched all scheduled and audit processes to PYTHONMALLOC=pymalloc,
retaining original logs and scientific sources. This operational change does not
establish the crash cause; no completed scientific case was repeated.

## Next decision

{next_step}
The broad VRAM, quality and parameter-efficiency goal remains open.

[Prospective plan](paired_complete_training_plan.md),
[recovery plan](paired_complete_training_recovery_plan.md),
[summary](../results/paired_complete_training_v1/summary.json),
[receipt](../results/paired_complete_training_v1/receipt.json).
"""
report_path = Path("research/paired_complete_training_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + f"\n\n## Latest: paired complete-update training\n\n[H134](paired_complete_training_results.md): **{status}.**\n360 updates, mirrored run order, complete-update timing, passive telemetry.\nIndependent numerical/native-score audit: {s['audit_passed']}.\n\n{next_step}\nEarlier failures remain recorded; the full research goal remains open.\n\n"
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
    + f"\n\nLatest: [paired complete training](research/paired_complete_training_results.md).\n**{status}.** Full update/peak-memory comparison; 360 updates and independent audit.\nThe broader research goal remains open.\n\n"
    + rest,
    encoding="utf-8",
    newline="\n",
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "recovery_finish.log", "recovery_finish_exit.txt")
]
files += [
    report_path,
    Path("research/paired_complete_training_plan.md"),
    Path("research/paired_complete_training_recovery_plan.md"),
    current,
    readme,
]
receipt = dict(
    study="H134",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=360,
    files={f.as_posix(): sha(f) for f in files},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
