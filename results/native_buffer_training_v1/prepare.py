"""Freeze all complete-update arms and inherited numerical audit requirements."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/attention_reproducibility_v1/receipt.json")["files"])
base = read("results/optimizer_memory_v1/protocol.json")
for key in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(base[key])
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for arm in ("ordinary", "native", "offload", "reuse"):
    (ROOT / arm / "runs").mkdir(parents=True)
inputs = [
    "results/attention_reproducibility_v1/receipt.json",
    "results/optimizer_memory_v1/protocol.json",
    "results/optimizer_memory_v1/source/profile.py",
    "results/optimizer_memory_v1/source/audit.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/compact_training_v1/study.py",
]
files = [*ROOT.glob("*.py"), Path("research/native_buffer_training_plan.md")]
p = dict(
    study="H130",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=base["maintained_files"],
    fixtures=[f for f in base["fixtures"] if f["loss_policy"] == "fp32_default_native"],
    steps=30,
    warmup=10,
    modes={"default": dict(foreach=None, fused=None)},
    datasets=base["datasets"],
    backwards=240,
    training_updates=240,
    training_targets=983040,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    timing_stability_max=1.15,
    host_peak_max=128 * 2**20,
    nll_ratio_max=1.01,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    broad_goal_achieved=False,
)
for key in (
    "score_tolerance",
    "parameter_relative_tolerance",
    "parameter_absolute_tolerance",
    "moment_tolerance",
    "clip_tolerance",
    "gradient_global_tolerance",
    "gradient_tensor_tolerance",
):
    p[key] = base[key]
assert len(p["fixtures"]) == 2
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen H130: 240 updates, 24 artifacts, 16 study and 8 audit scores.")
