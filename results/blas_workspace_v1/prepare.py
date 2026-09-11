"""Freeze the workspace-only diagnostic comparison."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/blas_workspace_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/native_buffer_training_v1/receipt.json")["files"])
prior = read("results/checkpoint_input_offload_v1/protocol.json")
for key in ("sources", "input_hashes", "maintained_files", "library_hashes"):
    hashes(prior[key])
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for mode in ("high", "low"):
    for arm in ("native", "reuse"):
        (ROOT / mode / arm / "runs").mkdir(parents=True)
files = [*ROOT.glob("*.py"), Path("research/blas_workspace_plan.md")]
inputs = [
    "results/native_buffer_training_v1/receipt.json",
    "results/checkpoint_input_offload_v1/protocol.json",
    "results/checkpoint_input_offload_v1/result.json",
    "results/gradient_staging_v1/study.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/streamed_evaluation_v1/source/evaluation.py",
    ".venv/Lib/site-packages/torch/backends/cuda/__init__.py",
]
p = dict(
    study="H131",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=prior["maintained_files"],
    fixtures=[f for f in prior["fixtures"] if f["loss_policy"] == "fp32_default_native"],
    repetitions=10,
    warmup=3,
    backwards=88,
    training_updates=0,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    host_peak_max=128 * 2**20,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
