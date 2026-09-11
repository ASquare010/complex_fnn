"""Publish H159's bounded timing result with all failures and audit scope."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/checkpoint_fp16_timing_v1")
for stage in ("worker", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert not (ROOT / "receipt.json").exists()
p = read(ROOT / "protocol.json")
a = read(ROOT / "audit.json")
s = read(ROOT / "summary.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert sha("results/checkpoint_host_isolation_v1/receipt.json") == p["prior_receipt"]
hashes(read("results/checkpoint_host_isolation_v1/receipt.json")["files"])
status = (
    "PASS complete-update resource/timing gate"
    if s["passed"]
    else "FAIL complete-update qualification; no long-training promotion"
)
rows = []
for f in s["fixtures"]:
    for c in f["comparisons"]:
        failed = []
        if not c["passed"]:
            failed.append("timing/resource/short-NLL comparison")
        if not f["stability_pass"]:
            failed.append("timing stability")
        if not f["telemetry_pass"]:
            failed.append("telemetry")
        if not f["host_pass"]:
            failed.append("pinned-host cap")
        if not a["passed"]:
            failed.append("independent audit")
        rows.append(
            f"| {f['dataset']} | {f['seed']} | {c['reference']} | {100 * (1 - c['memory_ratio']):.2f}% | {c['timing']['event_complete_ms']:.4f} | {c['timing']['wall_ms']:.4f} | {c['nll_ratios'][0]:.6f} / {c['nll_ratios'][1]:.6f} | {', '.join(failed) or 'none'} |"
        )
unstable = []
for m in s["metrics"]:
    for key in ("wall_ms", "event_complete_ms"):
        if m["stability"][key] > 1.15:
            unstable.append(
                f"- Case{m['index']:02d}, {m['dataset']} seed{m['seed']} {m['arm']} repeat{m['repeat']}: {key} half-median ratio {m['stability'][key]:.5f}."
            )
for f in s["fixtures"]:
    for arm, values in f["between_repeat_stability"].items():
        for key, value in values.items():
            if value > 1.15:
                unstable.append(
                    f"- {f['dataset']} seed{f['seed']} {arm}: {key} between-repeat ratio {value:.5f}."
                )
resources = []
for arm in ("ordinary", "buffer4", "fp16"):
    ms = [m for m in s["metrics"] if m["arm"] == arm]
    resources.append(
        f"| {arm} | {min(m['peak_bytes'] for m in ms) / 2**20:.3f}-{max(m['peak_bytes'] for m in ms) / 2**20:.3f} | {max(m['host_bytes'] for m in ms) / 2**20:.3f} |"
    )
report = f"""# H159: FP16 checkpoint storage under complete optimizer updates

**{status}.** All 36 segments completed: 1,080 updates, 8,847,360 training targets,
72 study validation scores and 36 independent native final-state scores.
Independent clipping/first-Adam/moment, gradient, batch and native-scoring audit:
**{a["passed"]}**. The broader goal remains open.

Across these six fixtures, FP16 saved 23.77-23.99% of native peak GPU allocation.
Complete CUDA medians ranged from 0.60% faster to 0.56% slower than native and
3.25-4.13% faster than offload4. These are observed paired short-run timings;
fresh-training convergence and sustained throughput remain untested.

## Every fixture, against both controls

The candidate is buffered classifier loss plus eight FP16 GPU checkpoint inputs.
Ordinary uses native loss and GPU FP32 inputs; buffer4 uses the same buffered loss
with four FP32 checkpoint inputs offloaded to pinned CPU memory. Parameter count
is 9,099,648 for every arm. No learned activation or parameter reduction is added.

| Corpus | Seed | Reference | GPU allocation saved | Complete CUDA ratio | Wall ratio | Final NLL ratios, repeats0/1 | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Ratios are candidate/reference. Timing uses the combined 40 measured updates per
arm and fixture after 10 warmups in each 30-update segment. The native limit is 1.15;
the offload limit is 1.05 for both CUDA and wall medians. GPU allocation uses the
maximum whole-segment peak across both repeats, capped at 0.90 native and 1.02
offload4. Each repeat's final NLL must be <=1.01 of its corresponding reference.
All fixtures and all stability checks must pass; aggregates do not rescue failures.

## Absolute resource measurements

| Arm | Whole-segment GPU allocated peak range MiB | Maximum allocator-owned pinned host MiB |
|---|---:|---:|
{chr(10).join(resources)}

Memory includes model, Adam state, resident data, construction, evaluation,
training, first-gradient copies, state serialization and diagnostic phases. It is
allocator memory, excluding driver/context and unrelated processes, and is not
isolated inference memory. This shared-process study may retain offload host
caches; its cap is 128 MiB. The isolated host advantage was qualified separately in
[H158](checkpoint_host_isolation_results.md), including native's five-byte peak.
H157's failed absolute-zero host gate remains failed.

## Timing stability

Every segment's two 10-update half-medians and every arm's between-repeat medians
must have max/min <=1.15 for both CUDA and wall time. Failures below are descriptive;
no repeat removal, clock normalization or retrospective threshold changes occur.

{chr(10).join(unstable) if unstable else "All within-segment and between-repeat stability gates pass."}

Passive 200 ms telemetry covers every timed segment if its telemetry gate passes.
Per-segment states, clock/temperature/power samples, all paired repeat ratios, and
mean/median/sample variance are retained in telemetry.csv and summary.json. This
is not a long-duration throughput measurement; sensor variation alone cannot
prove the cause of a slowdown.

## Controlled execution and independent evidence

Use the six H156 ordinary step800 model/Adam/sampler states, two corpora,
seeds 101/113/127. Each segment resets to the same fixture. ABC CBA order reverses
to CBA ABC on alternating fixtures. Batch 16, context512, validation batch8,
FP32, TF32 off, four CPU threads, ordinary attention, 8.125MiB cuBLAS workspace,
AdamW LR0.0006/betas0.9,0.95/eps1e-8, original decay and gradient clipping1.

The H142 training loop is imported unchanged. Timers encompass forward/backward,
FP16 casts and FP32 restoration, checkpoint recomputation, CPU offload transfers,
clipping and Adam. Sampling, gradient clearing, logging, validation and saving are
outside update timing. First-step gradient/state copies occur only during warmup.
No completed case is rerun. All 36 segments have clean GPU boundaries; the 36
native audit evaluations have separate clean boundaries, 74 boundary records total.

The FP16 codec is unchanged from H157. It saves 240 block inputs and unpacks 240
per segment; no full activation snapshots are retained. H157 established local
approximate-VJP semantics, not exact differentiation of the FP32 forward. H159
compares full saved first gradients against native per repeat under the same
separately frozen FP16 caps (global.002, max tensor.02, cosine.99999). Exact
controls retain 1e-5 global/1e-4 tensor limits; loss relative error<=1e-6 for all.

Independent NumPy verifies clipping and first Adam/moment updates using each arm's
actual saved gradients, with unchanged H142 tolerances. This checks the optimizer
implementation; it does not assert that approximate gradients yield the same
parameter trajectory. Native scoring independently verifies all 36 final states,
and saved sampler states regenerate all 1,080 batches. First/final model and moment
hashes, counters 801/830, finite states, stored artifacts and source hashes verify.
The audit adds zero backwards or optimizer updates. Full results remain in audit.json.

## Decision

{"The candidate earns fresh multi-seed long training. That next test must assess convergence and sustained timing from initialization before any quality-preserving claim." if s["passed"] else "Do not promote this recipe to long training. Keep the passing subchecks and all failed gates; distinguish consistent added compute from measurement instability before redesigning."}
This short continuation cannot establish fresh-training quality, stronger complex
pattern learning, novel activation geometry, fewer parameters, SOTA or a breakthrough.
H156 remains the existing long-training qualification of the exact offload helper.

[Prospective plan](checkpoint_fp16_timing_plan.md),
[machine-readable summary](../results/checkpoint_fp16_timing_v1/summary.json),
[independent audit](../results/checkpoint_fp16_timing_v1/audit.json).
"""
path = Path("research/checkpoint_fp16_timing_results.md")
assert not path.exists()
path.write_text(report, encoding="utf-8")
for target, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = target.read_text(encoding="utf-8")
    assert old.startswith(title)
    rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    link = (
        "research/checkpoint_fp16_timing_results.md"
        if target.name == "README.md"
        else "checkpoint_fp16_timing_results.md"
    )
    intro = f"Latest: [H159 FP16 complete-update test]({link}): **{status}.**\n1,080 updates across six fixtures, paired repeats and independent optimizer/score audit.\nFresh long-training quality remains unproven. Goal open.\n\n"
    target.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    path,
    Path("research/checkpoint_fp16_timing_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write_json(
    ROOT / "receipt.json",
    dict(
        study="H159",
        status="EVIDENCE_VERIFIED",
        gate=status,
        training_updates=1080,
        training_backwards=1080,
        training_targets=8847360,
        goal_achieved=False,
        files={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
