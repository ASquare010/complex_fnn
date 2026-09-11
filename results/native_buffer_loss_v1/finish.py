"""Preserve the failed exactness gate without launching unearned model work."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_loss_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert r["backwards"] == 16 and r["training_updates"] == 0 and not r["cases"]
q = r["qualification"]
assert not q["passed"] and len(q["records"]) == 8
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])
assert all(row["finite"] and row["inputs_unchanged"] for row in q["records"])
assert sum(len(row["directions"]) for row in q["records"]) == 6
assert all(d["passed"] for row in q["records"] for d in row["directions"])
rows = [
    f"| {x['dtype']} | {tuple(x['shape'])} | {x['bitwise_equal'][0]} | {x['bitwise_equal'][1]} | {x['bitwise_equal'][2]} | {x['max_abs'][0]:.3e} |"
    for x in q["records"]
]
report = (
    """# H127: native-buffer classifier qualification fails

**FAIL exactness; no full-model resource or training claim.** Sixteen operator
backwards and twelve finite-difference forwards ran. The strict gate stopped
all forty proposed model probes and all four proposed model replays. No
optimizer updates, training targets, model gradient artifacts or run retries.

| Precision | (tokens, vocabulary, width) | Exact loss | Exact input gradient | Exact weight gradient | Input max absolute difference |
|---|---|---|---|---|---:|
"""
    + "\n".join(rows)
    + """

All outputs/gradients were finite, all caller inputs/upstream gradients were
unchanged, and all six directional derivative checks passed (maximum absolute
error 4.410e-10). The numerical derivative appears correct within the tested
tolerances, but five of eight cases fail the deliberately stronger bitwise
requirement. Every hidden input has column-major strides; this study does
not establish whether row-major/full-model cases differ.

The candidate owns its logits buffer and uses native log-softmax/NLL kernels
with aliased output storage. It then computes dH=G W directly. PyTorch's
[mm_mat1_backward implementation](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/csrc/autograd/FunctionsManual.cpp)
selects (W^T G^T)^T for column-major inputs. These are algebraically identical,
but matrix-product dispatch can change floating-point results. This provides
a specific layout hypothesis; it is not a demonstrated causal attribution yet.
A separate prospectively frozen layout-aware variant is the next experiment.
The failed original implementation and thresholds remain unchanged.

In-place cross-entropy already has [prior art](https://github.com/mgmalek/efficient_cross_entropy).
[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) avoids materializing the full
logit matrix, unlike this native-buffer prototype. No novelty, parameter
reduction, VRAM improvement or long-run quality result is established here.

Source/input hashes, all 61 maintained files and two zero GPU allocator
boundaries verify. [Protocol](native_buffer_loss_plan.md),
[raw qualification](../results/native_buffer_loss_v1/qualification.json),
[receipt](../results/native_buffer_loss_v1/receipt.json).
The broader research goal remains open.
"""
)
Path("research/native_buffer_loss_results.md").write_text(report, encoding="utf-8")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
current.write_text(
    head
    + "\n\n## Latest: native-buffer exactness failure\n\n[H127](native_buffer_loss_results.md): native-buffer reuse fails bitwise input-gradient\nchecks in five of eight operator cases. Loss/weight gradients are exact; finite\ndifferences pass. Sixteen backwards, zero updates; all model profiling stopped.\nNext: test PyTorch's layout-dependent input-gradient multiplication explicitly.\nNo VRAM/quality claim or maintained default change. Full goal remains open.\n\n"
    + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.log.gz"),
    *ROOT.glob("*_exit.txt"),
    ROOT / "CURRENT_STATE.before.md",
    ROOT / ".gitignore",
    current,
    Path("research/native_buffer_loss_results.md"),
    Path("research/native_buffer_loss_plan.md"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_gate="FAIL_OPERATOR_BITWISE_EXACTNESS",
    backwards=16,
    finite_difference_forwards=12,
    model_probes=0,
    model_replays=0,
    training_updates=0,
    maintained_hashes_verified=61,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
assert not (ROOT / "receipt.json").exists()
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(receipt["scientific_gate"])
