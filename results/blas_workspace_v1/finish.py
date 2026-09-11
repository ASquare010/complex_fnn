"""Seal the isolated workspace result and record the next measured question."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/blas_workspace_v1")
p, s = read(ROOT / "protocol.json"), read(ROOT / "summary.json")
results = {m: read(ROOT / (m + "_result.json")) for m in ("high", "low")}
audits = {m: read(ROOT / ("audit_" + m + ".json")) for m in ("high", "low")}
assert not (ROOT / "receipt.json").exists()
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/native_buffer_training_v1/receipt.json")["files"].items():
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
status = "PASS diagnostic gates" if s["passed"] else "FAIL combined diagnostic gate"
rows = [
    f"| {x['dataset']} | {x['workspace']} | {x['arm']} | {x['peak_mib']:.3f} | {x['event_ms']:.3f} | {x['wall_ms']:.3f} | {x['released_bytes'] / 2**20:.3f} |"
    for x in s["metrics"]
]
comparisons = []
for x in s["comparisons"]:
    ratios = x["ratios"]
    failed = ", ".join(k for k, v in x["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {x['dataset']} | {x['control']} | {100 * (1 - ratios['peak_mib']):.2f}% | {ratios['event_ms']:.4f}x | {ratios['wall_ms']:.4f}x | {failed} |"
    )
account = [
    f"| {x['dataset']} | {x['arm']} | {x['release_difference_bytes'] / 2**20:.3f} | {x['predicted_bytes'] / 2**20:.3f} | {x['peak_difference_bytes'] / 2**20:.3f} |"
    for x in s["accounting"]
]
passed_audit = all(x["passed"] for x in audits.values())
phases = sum(len(row["measurement"]["phases"]) for r in results.values() for row in r["cases"])
next_step = (
    "A separate complete-update test at the smaller workspace is earned, including original-runtime and quality controls."
    if s["passed"]
    else "Do not extend this 128-KiB workspace to longer training. Test an intermediate explicit workspace size through the installed PyTorch workspace API, keeping the deterministic environment and algorithm policy fixed. That isolates external workspace capacity without accepting this speed penalty. Require fresh exactness/resource comparisons; no threshold or old gate changes."
)
report = f"""# H131: workspace memory is measurable, but the smallest setting is costly

**{status}.** Exact native replay audit: **{passed_audit}**.
Workspace-release prediction: **{s["accounting_passed"]}**. This is an isolated
storage diagnostic; no optimizer updates or long-run quality claims.\nThe smaller setting removes 63.75 MiB, but the buffer classifier takes\n2.76x/2.16x the high-workspace event time on WikiText/TinyStories. It fails\nthe 1.15x runtime limit despite exact native replay and lower memory.

| Corpus | Workspace | Classifier | Diagnostic peak MiB | Median event ms | Median wall ms | Live workspace released MiB |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Low-workspace buffer classifier compared with the preregistered controls:

| Corpus | Control | Peak allocation saved | Event ratio | Wall ratio | Failed gates |
|---|---|---:|---:|---:|---|
{chr(10).join(comparisons)}

![Workspace memory and runtime](figures/blas_workspace.png)

Both policies retain strict deterministic algorithms, default attention,
cuDNN deterministic execution, FP32 training, disabled TF32 and four CPU
threads. Both classifier arms offload checkpoint inputs and retain ordinary
RMSNorm and resident gradients. The only policy change is
`CUBLAS_WORKSPACE_CONFIG`: high `:4096:8`, low `:16:8`.
The installed getter reports 32 MiB and 128 KiB respectively; no size setter
or backend override is used. Lower workspace may change GEMM algorithms.
Cross-workspace bitwise equality is not required or claimed.

## Direct workspace accounting

After each case releases its model/data objects and Python garbage collection
runs, clear only cuBLAS workspaces and measure the live CUDA allocation drop.
All remaining live allocation falls to zero. The high setting releases 64 MiB;
the low setting releases 0.25 MiB. The difference matches twice the queried
per-workspace size difference: 2 x (32 - 0.125) = 63.75 MiB.

| Corpus | Classifier | Measured release difference MiB | Predicted difference MiB | Diagnostic peak difference MiB |
|---|---|---:|---:|---:|
{chr(10).join(account)}

This establishes the workspace allocation effect in these probes. The factor
of two is inferred from released bytes; it is not an enumeration of internal
library handles. The earlier H130 excess of 47.75 MiB is consistent with two
pools changing from the Ada default 8.125 MiB to 32 MiB. That default follows
[PyTorch's allocation source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CublasHandlePool.cpp);
the current isolated release measurement is stronger evidence than attributing
H130's entire policy bundle by subtraction alone. It does not establish which
GEMM selection or kernel explains the runtime change.

[NVIDIA](https://docs.nvidia.com/cuda/cublas/index.html) documents both deterministic
settings and warns the smaller one may limit performance.
[PyTorch](https://docs.pytorch.org/docs/main/cuda_environment_variables.html)
documents the size/count syntax. These are existing controls, not a new
activation, FFN or algorithmic novelty. Parameters remain 9,099,648.

## Verification and limits

Eighty fixed-state probe backwards plus eight fresh native replays =
**88 backwards, zero optimizer updates**. Probe targets: 327,680; replay targets:
32,768, all diagnostic. Each probe case and each replay evaluates one 4,096-target
BF16 validation batch (65,536 evaluation targets total), creating the evaluation
workspaces present in training. This is not a full validation sweep.
Eight saved gradient artifacts, all {phases} probe memory intervals and 20 zero
GPU allocator boundaries are retained. All 61 maintained file hashes and frozen
source/input hashes verify. No completed case was retried.

Every native replay uses its corresponding workspace setting, repeats evaluation,
and checks loss, all parameter gradients and evaluation metadata bitwise against
the saved case. Model/optimizer/sampler states remain unchanged. H123's existing
checks against the older reference retain their original numerical tolerances.
The resource loop is reused unchanged, with evaluation inside its construction
interval and no intervening peak reset. Thus that evaluation's transient and
persistent allocations count toward the diagnostic peak.

Timing uses seven measured probes after three warmup probes, with every raw
value, mean, median and variance retained. Workspace sizes run in sequential
processes; classifier arm order reverses across corpora. Scheduling/thermal
variation remains possible, and these short timings are not long-run throughput
estimates. Pinned host allocation is separate additional RAM. CUDA allocation
excludes driver/context allocations and other processes. Whole training-job
VRAM, Adam updates and long-run quality remain untested at the smaller workspace.

## Next decision

{next_step}
H130's runtime failure and all other earlier gates remain unchanged. No maintained
default changes. The full VRAM/quality and parameter-efficient FFN goal is open.

[Plan](blas_workspace_plan.md), [summary](../results/blas_workspace_v1/summary.json),
[receipt](../results/blas_workspace_v1/receipt.json).
"""
report_path = Path("research/blas_workspace_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
current.write_text(
    head
    + f"\n\n## Latest: deterministic workspace isolation\n\n[H131](blas_workspace_results.md): **{status}.**\nWorkspace release accounting verifies the 63.75 MiB high-low difference.\nNative replay audit: {passed_audit}. 88 backwards, no updates, eight gradients;\nall 61 maintained files unchanged. No long-run quality or novelty claim.\n\n{next_step}\nPrior failed gates and the full research goal remain unchanged.\n\n"
    + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
readme.write_text(
    head
    + f"\n\nLatest: [deterministic workspace isolation](research/blas_workspace_results.md).\n**{status}.** Direct allocation and gradient evidence; see measured runtime\ntradeoffs and the next test. The full research goal remains open.\n\n"
    + rest.replace("Latest:", "Earlier complete-training study:", 1),
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
    Path("research/blas_workspace_plan.md"),
    Path("research/figures/blas_workspace.png"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_status=status,
    diagnostic_gate_passed=s["passed"],
    native_replay_passed=passed_audit,
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
