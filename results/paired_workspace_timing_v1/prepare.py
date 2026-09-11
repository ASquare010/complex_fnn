"""Freeze the paired timing test and its unchanged predecessor evidence."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_workspace_timing_v1")
assert not (ROOT / "protocol.json").exists()
old = read("results/blas_workspace_mid_v1/receipt.json")
hashes(old["files"])
base = read("results/checkpoint_input_offload_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files", "library_hashes"):
    hashes(base[field])
prior = read("results/blas_workspace_mid_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(prior[field])
for name, source in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(source).read_bytes())
inputs = [
    "results/blas_workspace_mid_v1/receipt.json",
    "results/blas_workspace_mid_v1/protocol.json",
    "results/checkpoint_input_offload_v1/protocol.json",
    "results/blas_workspace_mid_v1/common.py",
    "results/blas_workspace_v1/common.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/source/common.py",
    "results/checkpoint_input_offload_v1/source/offload.py",
    "results/streamed_evaluation_v1/source/evaluation.py",
]
p = dict(
    study="H133",
    fixtures=[f for f in base["fixtures"] if f["loss_policy"] == "fp32_default_native"],
    datasets=base["datasets"],
    maintained_files=base["maintained_files"],
    base_verified={field: base[field] for field in ("sources", "input_hashes", "library_hashes")},
    input_hashes={f: sha(f) for f in inputs},
    sources={
        f.as_posix(): sha(f)
        for f in [*ROOT.glob("*.py"), Path("research/paired_workspace_timing_plan.md")]
    },
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    controls=["native8", "reuse32"],
    cycles=["ABBA", "BAAB"],
    warmup=3,
    measured=5,
    backwards=260,
    evaluations=36,
    boundaries=7,
    timing_ratio_limit=1.15,
)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("H133 frozen: 260 backwards; no updates")
