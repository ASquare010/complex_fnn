"""Freeze complete-update controls and inherited audit tolerances."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_training_v1")
assert not (ROOT / "protocol.json").exists()
previous = read("results/compact_rmsnorm_v1/receipt.json")
hashes(previous["files"])
base = read("results/optimizer_memory_v1/protocol.json")
for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(base[field])
(ROOT / "CURRENT_STATE.before.md").write_bytes(Path("research/CURRENT_STATE.md").read_bytes())
for arm in ("baseline", "combined"):
    (ROOT / arm / "runs").mkdir(parents=True, exist_ok=False)
inputs = [
    "results/compact_rmsnorm_v1/receipt.json",
    "results/optimizer_memory_v1/protocol.json",
    "results/optimizer_memory_v1/source/profile.py",
    "results/optimizer_memory_v1/source/audit.py",
]
files = [*ROOT.glob("*.py"), Path("research/compact_training_plan.md")]
p = dict(
    study="H125",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=base["maintained_files"],
    fixtures=[f for f in base["fixtures"] if f["loss_policy"].endswith("chunks")],
    steps=30,
    warmup=10,
    modes={"default": dict(foreach=None, fused=None)},
    datasets=base["datasets"],
    backwards=122,
    training_updates=120,
    training_targets=491520,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    timing_stability_max=1.15,
    host_peak_max=128 * 2**20,
    nll_ratio_max=1.01,
    current_state_before=sha("research/CURRENT_STATE.md"),
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
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen 120 complete updates plus two qualification backwards.")
