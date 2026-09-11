"""Freeze original states and the six long-training comparisons."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_long_training_v1")
assert not (ROOT / "protocol.json").exists()
previous = read("results/compact_training_v1/receipt.json")
hashes(previous["files"])
old = read("results/fp32_training_replication_recovery_v1/protocol.json")
hashes(old["sources"])
hashes(old["maintained_files"])
fixtures = [f for f in old["fixtures"] if f["seed"] == 101]
for name in ("CURRENT_STATE", "README"):
    source = Path("research/CURRENT_STATE.md" if name == "CURRENT_STATE" else "README.md")
    (ROOT / (name + ".before.md")).write_bytes(source.read_bytes())
for arm in ("native", "chunks", "combined"):
    (ROOT / arm / "runs").mkdir(parents=True, exist_ok=False)
files = [*ROOT.glob("*.py"), Path("research/compact_long_training_plan.md")]
inputs = [
    "results/compact_training_v1/receipt.json",
    "results/compact_training_v1/adapter.py",
    "results/compact_training_v1/study.py",
    "results/compact_rmsnorm_v1/norm.py",
    "results/gradient_staging_v1/staging.py",
    "results/fp32_training_replication_recovery_v1/protocol.json",
    "results/fp32_training_replication_recovery_v1/source/study.py",
    "results/fp32_training_replication_recovery_v1/source/audit.py",
]
checkpoint_hashes = {f["checkpoint"]: sha(f["checkpoint"]) for f in fixtures}
p = dict(
    study="H126",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=old["maintained_files"],
    fixtures=fixtures,
    datasets=old["datasets"],
    checkpoint_hashes=checkpoint_hashes,
    training_updates=4800,
    training_targets=19660800,
    backwards=4812,
    study_scores=24,
    audit_scores=20,
    tensor_artifacts=24,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    stability_max=1.15,
    nll_ratio_max=1.01,
    host_peak_max=128 * 2**20,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen six 800-update runs, original seed101 states, matched controls.")
