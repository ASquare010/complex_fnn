"""Freeze the selected partial-offload variant against fresh ordinary controls."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/partial_offload_long_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/partial_offload_training_v1/receipt.json")["files"])
base = read("results/ordinary_long_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes", "schedule")}
p.update(
    study="H138",
    previous_goal_turn="progress",
    training_updates=9600,
    training_targets=39321600,
    study_backwards=9612,
    audit_backwards=12,
    backwards=9624,
    study_scores=48,
    audit_scores=42,
    tensor_artifacts=48,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
)
schedule = []
for i, f in enumerate(p["fixtures"]):
    for arm in ("ordinary", "buffer4") if i % 2 == 0 else ("buffer4", "ordinary"):
        schedule.append(dict(index=len(schedule), fixture=f, arm=arm))
p["schedule"] = schedule
inputs = [
    "results/partial_offload_training_v1/receipt.json",
    "results/partial_offload_training_v1/offload.py",
    "results/ordinary_long_training_v1/protocol.json",
    "results/ordinary_long_training_v1/loop.py",
    "results/ordinary_long_training_v1/io.py",
    "results/fp32_training_replication_recovery_v1/source/audit.py",
    "results/fp32_training_replication_v1/source/initialize.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/source/offload.py",
    "results/compact_training_v1/study.py",
]
p["input_hashes"] = {f: sha(f) for f in inputs}
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [*ROOT.glob("*.py"), Path("research/partial_offload_long_plan.md")]
}
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in schedule:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(ROOT / "protocol.json", p)
print("H138 frozen:12 fresh runs,9600 updates,three seeds,two corpora")
