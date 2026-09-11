"""Publish the scoped ordinary-policy result and seal the evidence."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/ordinary_complete_training_v1")
p, s, a = [read(ROOT / name) for name in ("protocol.json", "summary.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
hashes(json.loads((ROOT / "recovery_protocol.json").read_text(encoding="utf-8-sig"))["files"])
for path, digest in read("results/paired_complete_training_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "seal", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
status = "PASS ordinary short-training gate" if s["passed"] else "FAIL ordinary short-training gate"
comparisons = []
for c in s["comparisons"]:
    q = c["ratios"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {c['dataset']} | {c['repeat']} | {c['control']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f} | {q['wall_ms']:.4f} | {q['nll']:.8f} | {failed} |"
    )
metrics = [
    f"| {m['index']} | {m['dataset']} | {m['arm']} | {m['peak_mib']:.3f} | {m['host_mib']:.3f} | {m['event_ms']:.3f} | {m['wall_ms']:.3f} | {m['nll']:.7f} | {m['stability']:.4f} | {m['telemetry_samples']} |"
    for m in s["metrics"]
]
calibration = []
for group in a["calibrated"]:
    for c in group["checks"]:
        calibration.append(
            f"| {group['dataset']} | {group['control']} | {c['kind']} | {c['repeat']} | {c['noise']['global_relative']:.3e} | {c['error']['global_relative']:.3e} | {c['limits']['global']:.3e} | {c['noise']['tensor_relative']:.3e} | {c['error']['tensor_relative']:.3e} | {c['limits']['tensor']:.3e} | {c['passed']} |"
        )
finals = [
    f"| {c['dataset']} | {c['control']} | {c['repeat']} | {c['error']['global_relative']:.3e} | {c['error']['tensor_relative']:.3e} | {c['error']['max_absolute']:.3e} |"
    for c in a["final_distances"]
]
next_step = (
    "Run fresh longer training with at least three independent seeds on both corpora, paired ordinary/native-offload/buffer-offload controls, native validation and complete-update resource accounting. The short result earns that test; it does not prove fresh convergence or long-run quality. Preserve the existing structural FFN failures while pursuing the broader parameter-efficiency goal."
    if s["passed"]
    else "Do not extend long training. Choose the next narrower experiment from the failed numerical/resource gate and retained telemetry. Keep thresholds and failed repeats unchanged; ordinary-policy execution does not automatically establish practical efficiency."
)
report = f"""# H135: ordinary-policy complete training

**{status}.** Independent clipping/Adam, native-score and bounded gradient-noise
audit: **{a["passed"]}**. Each corpus/control/repeat must pass individually.
This is a fixed-seed short continuation, not a breakthrough or fresh-training claim.

| Corpus | Repeat | Control | GPU allocation saved | Complete CUDA ratio | Wall ratio | NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

All arms use ordinary/default attention, deterministic algorithms disabled,
no cuBLAS environment override or capacity setter, and the same observed
8.125-MiB workspace. `ordinary`: native loss, checkpoint inputs on GPU.
`native`: native loss with checkpoint-input CPU offload. `reuse`: H128 buffer
classifier with the same CPU offload. All retain 9,099,648 parameters, FP32,
TF32 off, four CPU threads, native RMSNorm, original clipping/Adam schedule,
and the UV-managed Python runtime with PYTHONMALLOC=pymalloc.

## Resource and timing evidence

| Run | Corpus | Arm | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing halves ratio | Telemetry samples |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Mirrored order per corpus is reuse, ordinary, native, native, ordinary, reuse.
Both repeats must meet allocation ratio <=0.90, complete CUDA/wall <=1.15,
NLL <=1.01, candidate pinned host peak <=128 MiB and timing halves <=1.15.
No failed repeat is averaged away. Repeats share a seed and starting checkpoint.
Twenty measured updates follow ten warmups per run; means, medians, variance,
all steps and reserved allocation are retained in raw machine-readable results.

The unchanged H134 loop times forward, backward, clipping and Adam together,
including recomputation and offload transfers. Gradient clearing, batch sampling,
validation, serialization and inventory are outside timing, but included in
whole-job peak allocation. Allocation excludes driver/context and other processes;
pinned-host measurements are not total CPU RSS. The current matched comparisons
support conclusions within this policy. Cross-study timing differences cannot
prove that a particular deterministic attention kernel caused H134's slowdown.

## Numerical qualification

H128's exact operator qualification remains evidence for the buffer classifier.
H129 showed ordinary full-model gradients are nondeterministic. Accordingly,
this study prospectively uses repeated native controls, not an invalid full-model
bitwise criterion. Symmetric relative L2 uses a 1e-12 norm floor. Limits are:
global min(1e-5,max(1e-6,10*native noise)); tensor min(1e-4,max(1e-5,10*native noise)).
Native-repeat noise must itself remain within the fixed caps. These two repeats
are a noise diagnostic, not a statistical confidence interval. Bitwise match
flags and maximum absolute errors are also retained in the audit JSON.

| Corpus | Control | Gradient | Repeat | Native global noise | Candidate global | Global limit | Native tensor noise | Candidate tensor | Tensor limit | Pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(calibration)}

Final model weight distances are descriptive; final native validation and the
original finite/state/optimizer audits decide short-run qualification. This
separates numerical drift from a claim of exact trajectory identity.

| Corpus | Control | Repeat | Final global distance | Max tensor distance | Max absolute difference |
|---|---|---|---:|---:|---:|
{chr(10).join(finals)}

## Reproducibility and limits

360 updates/backwards, 1,474,560 training targets, 24 full study scores and 12
independent native scores. Thirty-six tensor artifacts, 804 memory intervals,
37 zero allocator boundaries; all 61 maintained source/config/test files stay
unchanged. Frozen source, fixture/data and library hashes verify. Independent
NumPy checks clipping and the first Adam update/moments for every run; checkpoint
steps, finite states, saved hashes and all 360 batches are verified separately.

Read-only GPU telemetry samples every 200 ms and each monitor is stopped in
finally. All cases require coverage during measured updates. Clock/power and
temperature ranges, medians and raw samples are retained; missing sensors remain
missing. No GPU clock or power setting changed. Sparse telemetry cannot identify
a causal bottleneck. No historical failed gate is revised by this result.

This tests a memory implementation on existing fixtures, not novel activation
geometry, parameter reduction, multiple independent seeds, fresh convergence,
held-out broad task performance or SOTA superiority. No maintainable default
is changed until the stronger training evidence exists.

Case07 failed during torch.jit import with a Python SystemError before GPU work.
Cases00-06 (210 updates) were retained. A prospective bounded recovery restarted
only unfinished case07, then ran cases08-11 in the frozen order and unchanged
environment. The failure log is retained. No completed scientific case was repeated.
Automatic approval review initially could not complete recovery setup; inspection
confirmed no files/process were created, and the subsequent setup succeeded.

CPU audit preparation then crashed in Python's indented JSON encoder before
writing either aggregate result or audit manifest. One CPU-only compact-JSON
recovery completed the same seal; original source and failure log are retained.
No GPU work was repeated for this operational recovery.

## Next decision

{next_step}
The full research goal remains open.

[Prospective plan](ordinary_complete_training_plan.md),
[summary](../results/ordinary_complete_training_v1/summary.json),
[receipt](../results/ordinary_complete_training_v1/receipt.json).
"""
report_path = Path("research/ordinary_complete_training_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + f"\n\n## Latest: ordinary-policy complete training\n\n[H135](ordinary_complete_training_results.md): **{status}.**\n360 updates; all arms share ordinary execution and default workspace.\nIndependent numerical/native-score/noise audit: {a['passed']}.\n\n{next_step}\nEarlier failures and the broad research goal remain open.\n\n"
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
    + f"\n\nLatest: [ordinary-policy complete training](research/ordinary_complete_training_results.md).\n**{status}.** Matched memory/runtime controls and calibrated numerical audit.\nFull training and the broader research goal remain open.\n\n"
    + rest,
    encoding="utf-8",
    newline="\n",
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/ordinary_complete_training_plan.md"),
    Path("research/ordinary_complete_training_recovery_plan.md"),
    current,
    readme,
]
receipt = dict(
    study="H135",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=360,
    files={f.as_posix(): sha(f) for f in files},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
