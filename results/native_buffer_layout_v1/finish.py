"""Seal a resource improvement with its unresolved full-model exactness gate."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_layout_v1")
p, r, a, s, pairs = [
    read(ROOT / n)
    for n in ("protocol.json", "result.json", "audit.json", "summary.json", "pair_analysis.json")
]
assert not (ROOT / "receipt.json").exists()
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
hashes(read(ROOT / "pair_protocol.json")["files"])
prior = read("results/checkpoint_input_offload_v1/protocol.json")
for key in ("sources", "input_hashes", "maintained_files", "library_hashes"):
    hashes(prior[key])
for path, digest in read("results/native_buffer_loss_v1/receipt.json")["files"].items():
    target = ROOT / "CURRENT_STATE.before.md" if path == "research/CURRENT_STATE.md" else Path(path)
    assert sha(target) == digest
for stage in (
    "prepare",
    "study",
    "prepare_audit",
    "audit",
    "analyze",
    "prepare_pairs",
    "analyze_pairs",
    "plot",
):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert not a["passed"] and not s["passed"] and all(x["exact_loss"] for x in a["cases"])
assert all(not x["passed"] for x in a["cases"])
assert all(x["exact_gradients"]["norm.weight"] for x in a["cases"])
assert r["qualification"]["passed"]
rows = [
    f"| {x['dataset']} | {x['arm']} | {x['peak_mib']:.3f} | {x['event_ms']:.3f} | {x['wall_ms']:.3f} |"
    for x in s["metrics"]
]
ratios = [
    f"| {x['dataset']} | {100 * (1 - x['ratios']['peak_mib']):.2f}% | {x['ratios']['event_ms']:.4f}x | {x['ratios']['wall_ms']:.4f}x | failed |"
    for x in s["comparisons"]
]
errors = [
    f"| {x['dataset']} | {x['global_relative_l2']:.3e} | {x['exact_tensors']}/{x['total_tensors']} |"
    for x in pairs["pairs"]
]
phases = sum(len(x["measurement"]["phases"]) for x in r["cases"])
report = f"""# H128: buffer reuse saves memory; full-model exactness remains unresolved

**FAIL combined acceptance gate.** The revised classifier passes every
operator exactness check and all resource gates, but independent full-model
bitwise replay fails for both candidate and native control. This is useful
memory evidence, not qualification for longer training or proof of quality.

| Corpus | Arm | Diagnostic peak MiB | Median event ms | Median wall ms |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

| Corpus | CUDA allocation saved | Event ratio | Wall ratio | Full-model bitwise gate |
|---|---:|---:|---:|---|
{chr(10).join(ratios)}

![Diagnostic resource comparison](figures/native_buffer_layout.png)

Both arms use CPU checkpoint-input offload, ordinary RMSNorm, resident gradients,
FP32 native classifier arithmetic, default attention and original Adam state.
Only the classifier implementation changes. Peak pinned-host allocation is
64.000 MiB in both arms, additional to GPU allocation. Parameters stay at
9,099,648; there is no parameter reduction. The measured 51.968 MiB GPU saving
is incremental to input offloading, not a comparison with the unmodified
no-offload model. CUDA allocation excludes driver/context and other processes.

## What changed after H127

H127 reused native log-softmax/NLL buffers but calculated every input gradient
as G W. PyTorch's [matrix backward implementation](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/csrc/autograd/FunctionsManual.cpp)
uses (W^T G^T)^T when the hidden input is column-major. H128 follows that layout
choice while preserving the other kernels. All eight original FP32/FP64 cases
now match native loss and both gradients bitwise. A ninth case covers the
actual contiguous [8,512,384] hidden tensor and [4096,384] classifier, also
bitwise equal. All six directional derivatives pass, inputs/upstream gradients
remain unchanged and all outputs are finite. These results support the layout
explanation for H127's isolated failure. They do not establish whole-model
numerical identity across arbitrary shapes, backends or versions.

The scope is first derivatives, no autocast, contiguous weights, and 2D or
contiguous 3D hidden tensors. Native logits are privately allocated before
reuse; caller inputs and saved log-probabilities are not overwritten by backward.
[In-place classifier prior art](https://github.com/mgmalek/efficient_cross_entropy)
and [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) precede this study.
No new activation, FFN architecture or algorithmic novelty is claimed.

## Why the gate still fails

All four independent native replays reproduce loss exactly, but none reproduces
every saved parameter gradient bitwise, including the two ordinary native
controls. Final-normalization gradients match in every replay. For example,
the WikiText native-control replay matches the last FFN and attention-output
projection but differs at the last attention QKV projection and earlier layers.
This makes attention/backward reproducibility a targeted next diagnostic;
the current data do not isolate nondeterministic kernels, scheduling, storage
changes or another cause. It would be incorrect to attribute all differences
to the classifier or to claim a proven nondeterminism cause.

A separately frozen CPU-only analysis compares the already saved candidate and
native gradients, without replaying any GPU cases:

| Corpus | Whole-gradient relative L2 | Bitwise-equal parameter tensors |
|---|---:|---:|
{chr(10).join(errors)}

Those small differences do not rescue the predeclared exactness gate and do
not predict long-run loss. Independent replay recorded equality rather than
numeric magnitudes, so this table describes stored arm pairs, not replay error.

## Evidence, budget and next decision

H128 executed 18 qualification, 40 model-probe and four replay backwards:
**62 backwards, zero optimizer updates**, plus 12 finite-difference forwards.
The model probes process 163,840 diagnostic targets and replays 16,384; neither
is training exposure. Four saved gradient artifacts, all {phases} diagnostic
memory intervals, all 11 zero GPU allocator boundaries and source/input hashes
verify. All 61 maintained files are unchanged. Median timing uses repetitions
4–10; raw values, means and variances are retained. The memory peak includes
construction, all probes, transfers, serialization and state checking. No
optimizer update, validation sweep or full training-job peak was measured.

H127 additionally used 16 backwards and 12 finite-difference forwards before
its gate stopped model work. This research iteration therefore used 78
backwards and no training updates. No completed scientific case was retried.
A read-only PowerShell process crashed while inspecting audit JSON; a fresh
read succeeded. Scientific stages completed normally and were not restarted.

Next: a prospective reproducibility test with native-versus-native controls
and explicit attention-backward execution policy. Establish a trustworthy
reference before judging candidate-induced drift or committing to training.
Keep the original failures and thresholds. No defaults change. The broader
VRAM/quality and parameter-efficient FFN goal remains open.

[Plan](native_buffer_layout_plan.md),
[summary](../results/native_buffer_layout_v1/summary.json),
[CPU pair analysis](../results/native_buffer_layout_v1/pair_analysis.json),
[receipt](../results/native_buffer_layout_v1/receipt.json),
[rejected predecessor](native_buffer_loss_results.md).
"""
report_path = Path("research/native_buffer_layout_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
current.write_text(
    head
    + "\n\n## Latest: classifier resource gain, replay gate unresolved\n\n[H128](native_buffer_layout_results.md): layout-aware native-buffer reuse saves\n14.50–14.75% diagnostic GPU allocation with near-equal runtime. Nine operator\ncases pass bitwise checks, but independent full-model gradient replay fails\nfor candidate AND native controls. All loss values match. Combined gate fails.\n62 backwards, zero updates, four artifacts; H127's earlier 16-backward failure\nis preserved. No defaults or parameter counts change. Next: native reference\nreproducibility and explicit attention-backward policy before any longer run.\nThe full research goal remains open.\n\n"
    + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
old_readme_hash = read("results/compact_long_training_v1/receipt.json")["files"]["README.md"]
assert sha(readme) == old_readme_hash
(ROOT / "README.before.md").write_bytes(readme.read_bytes())
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
readme.write_text(
    head
    + "\n\nLatest diagnostic: [native-buffer classifier](research/native_buffer_layout_results.md)\nsaves 14.5–14.8% GPU allocation; full-model bitwise replay remains unresolved\nin both candidate and native controls. No training-quality or breakthrough claim.\n\n"
    + rest.replace("Latest:", "Earlier long-training study:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
for path in ROOT.glob("*/runs/*/result.json"):
    path.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    *ROOT.glob("*.before.md"),
    ROOT / ".gitignore",
    *ROOT.glob("*/runs/*/*"),
    current,
    readme,
    report_path,
    Path("research/native_buffer_layout_plan.md"),
    Path("research/figures/native_buffer_layout.png"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_gate="FAIL_FULL_MODEL_BITWISE_REPLAY",
    operator_qualification=True,
    resource_gates=True,
    full_model_exactness=False,
    backwards=62,
    training_updates=0,
    gradient_artifacts=4,
    maintained_hashes_verified=61,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
assert all(all(v for k, v in x["gates"].items() if k != "exact") for x in s["comparisons"])
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
