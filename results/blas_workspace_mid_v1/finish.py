"""Report the explicit intermediate-workspace comparison without rescuing old gates."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/blas_workspace_mid_v1")
p, s = read(ROOT / "protocol.json"), read(ROOT / "summary.json")
r = {m: read(ROOT / (m + "_result.json")) for m in ("high", "low")}
a = {m: read(ROOT / ("audit_" + m + ".json")) for m in ("high", "low")}
assert not (ROOT / "receipt.json").exists()
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/blas_workspace_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in (
    "prepare",
    "high",
    "low",
    "prepare_audit",
    "audit_high",
    "audit_low",
    "analyze",
    "plot",
):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
for mode in ("high", "low"):
    assert r[mode]["settings"]["workspace_env"] == ":4096:8"
    assert r[mode]["settings"]["workspace_bytes"] == (32 if mode == "high" else 8) * 2**20
status = "PASS diagnostic gates" if s["passed"] else "FAIL combined diagnostic gate"
audit = all(x["passed"] for x in a.values())
rows = [
    f"| {x['dataset']} | {x['workspace']} | {x['arm']} | {x['peak_mib']:.3f} | {x['event_ms']:.3f} | {x['wall_ms']:.3f} |"
    for x in s["metrics"]
]
comparisons = []
for x in s["comparisons"]:
    q = x["ratios"]
    failed = ", ".join(k for k, v in x["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {x['dataset']} | {x['control']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f}x | {q['wall_ms']:.4f}x | {failed} |"
    )
next_step = (
    "Run a separately frozen complete-update comparison, retaining ordinary execution and matched deterministic native controls. Include paired timing and GPU telemetry because absolute control timing varied between studies."
    if s["passed"]
    else "Do not expand to longer training. Use a separately frozen timing experiment with alternating candidate/control probes and GPU clock/power telemetry to distinguish a persistent runtime penalty from execution-order drift. No threshold or old-result changes; the present gate remains failed."
)
phases = sum(len(x["measurement"]["phases"]) for v in r.values() for x in v["cases"])
report = f"""# H132: 8-MiB explicit workspace versus 32 MiB

**{status}.** Native replay: **{audit}**; release accounting:
**{s["accounting_passed"]}**. The deterministic environment remains fixed while
only the external cuBLAS workspace capacity changes. No optimizer updates.\nMemory falls by 48 MiB. Three of four resource comparisons pass, but TinyStories\nagainst its 8-MiB native classifier control is 18.57% slower by CUDA events\nand 17.74% slower by wall time, exceeding the fixed 15% limit.

| Corpus | Workspace | Classifier | Diagnostic peak MiB | Median event ms | Median wall ms |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

Low-workspace buffer classifier versus each predeclared control:

| Corpus | Control | Peak allocation saved | Event ratio | Wall ratio | Failed gates |
|---|---|---:|---:|---:|---|
{chr(10).join(comparisons)}

![Workspace resource comparison](figures/blas_workspace_mid.png)

High/low mean explicitly set 32/8 MiB via
`torch.backends.cuda.cublas_workspace_size`; both processes retain
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The getter verifies the setting before
model work. Strict deterministic algorithms, default SDPA, cuDNN deterministic
execution, disabled TF32, FP32 training and four CPU threads remain fixed.
Both native and buffer classifiers use checkpoint-input CPU offload, ordinary
RMSNorm and resident gradients. Parameters remain 9,099,648.

## What the evidence proves

After case objects are released, clearing cuBLAS workspaces releases 64 MiB
for the 32-MiB policy and 16 MiB for the 8-MiB policy. Live allocation then
reaches zero. The measured 48-MiB difference equals 2 x (32-8), and all four
matched diagnostic-peak differences are 48 MiB. The factor two follows from
measured bytes, not direct handle enumeration. The external capacity setter
therefore controls a concrete allocator cost while leaving the deterministic
environment unchanged. No internal GEMM algorithm or kernel cause is inferred.

The fresh native replay includes the same one-batch BF16 evaluation warmup
and verifies every saved gradient, loss and evaluation result bitwise at its
own workspace size. Cross-workspace bitwise equality is not required. Model,
optimizer and sampler states remain unchanged. The H128 operator and H131
worker/replay implementations are reused rather than reimplemented.

This existing [PyTorch workspace API](https://docs.pytorch.org/docs/stable/backends)
takes precedence over the environment for external workspace allocation.
[NVIDIA's reproducibility discussion](https://docs.nvidia.com/cuda/cublas/index.html)
provides context; this experiment is not a novel activation or FFN.

## Limits and reproducibility

**88 backwards, zero training updates**: 80 resource probes and eight native
replays, 327,680 and 32,768 diagnostic targets respectively. Sixteen one-batch
BF16 evaluations cover 65,536 targets, not full validation sweeps. Eight gradient
artifacts, {phases} memory intervals and 20 zero allocator boundaries are
recorded. All 61 maintained file hashes and frozen source/input hashes verify.
The CPU analysis recomputes timing summaries, peak maxima, artifact hashes and
workspace accounting. No completed scientific case was retried.

Timing uses seven measured probes after three warmup probes; all raw steps,
mean, median and variance are retained. Workspaces run in separate sequential
processes and arm order reverses across corpora. The fresh 32-MiB controls are
substantially slower than H131's 32-MiB controls despite the same shapes. That
observed variation makes cross-study timing attribution unsafe. No GPU clock
or power history was recorded here, so its cause is unknown. Within-study
predeclared thresholds still decide acceptance; this limitation does not
convert a failure into a pass or prove a specific workspace-induced slowdown.

The diagnostic peak includes evaluation workspace warmup, all probes and
serialization/state checks; it does not represent a complete training job.
CUDA allocation excludes driver/context and other processes. Pinned-host RAM
is separate. No long-run quality, parameter-efficiency or broad success claim.
The read-only inspection command had a PowerShell formatting error that was
corrected. One CPU-analysis launch request timed out in automatic permission\nreview before process creation; its single permitted retry succeeded. No\nscientific process was restarted.

## Next decision

{next_step}
H131/H130 failures stay recorded, maintained defaults stay unchanged, and the
full VRAM/quality and parameter-efficient FFN goal remains open.

[Plan](blas_workspace_mid_plan.md), [summary](../results/blas_workspace_mid_v1/summary.json),
[receipt](../results/blas_workspace_mid_v1/receipt.json).
"""
assert all(
    x["release_difference_bytes"] == 48 * 2**20 and x["peak_difference_bytes"] == 48 * 2**20
    for x in s["accounting"]
)
report_path = Path("research/blas_workspace_mid_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
current.write_text(
    head
    + f"\n\n## Latest: explicit intermediate-workspace comparison\n\n[H132](blas_workspace_mid_results.md): **{status}.**\n8-MiB external workspaces save 48 MiB versus 32-MiB workspaces while keeping\nthe deterministic environment fixed. Native replay: {audit}; accounting verified.\n88 backwards, zero updates, eight artifacts; all 61 maintained files unchanged.\n\n{next_step}\nThe full research goal and all earlier failed gates remain unchanged.\n\n"
    + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
readme.write_text(
    head
    + f"\n\nLatest: [intermediate-workspace comparison](research/blas_workspace_mid_results.md).\n**{status}.** Verified allocation and gradient evidence; runtime limitations\nand next experiment are documented. The full research goal remains open.\n\n"
    + rest.replace("Latest:", "Earlier workspace study:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
for path in ROOT.glob("*/*/runs/*/result.json"):
    path.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    *ROOT.glob("*.before.md"),
    ROOT / ".gitignore",
    *ROOT.glob("*/*/runs/*/*"),
    current,
    readme,
    report_path,
    Path("research/blas_workspace_mid_plan.md"),
    Path("research/figures/blas_workspace_mid.png"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_status=status,
    diagnostic_gate_passed=s["passed"],
    native_replay_passed=audit,
    workspace_accounting_passed=s["accounting_passed"],
    backwards=88,
    training_updates=0,
    gradient_artifacts=8,
    maintained_hashes_verified=61,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
