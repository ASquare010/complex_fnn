"""Freeze the larger-batch maintained-helper long run before any GPU work."""

import shutil
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/batch_scale_long_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/augmentation_barrier_v1/receipt.json")["files"])
base = read("results/partial_offload_long_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
qualified = read("results/batch_scale_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(qualified[field])
assert read("results/batch_scale_training_v1/receipt.json")["goal_achieved"] is False
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes", "schedule", "fixtures")}
p.update(
    study="H156",
    previous_goal_turn="progress",
    training_updates=9600,
    training_targets=78643200,
    study_backwards=9612,
    audit_backwards=12,
    backwards=9624,
    study_scores=48,
    audit_scores=42,
    tensor_artifacts=48,
    batch_size=16,
    validation_batch_size=16,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    prior_receipt=sha("results/augmentation_barrier_v1/receipt.json"),
    free_disk_bytes=shutil.disk_usage(".").free,
)
assert p["free_disk_bytes"] > 12 * 2**30
p["fixtures"] = [dict(f, batch=16) for f in base["fixtures"]]
p["maintained_files"] = dict(qualified["maintained_files"])
p["schedule"] = []
for i, f in enumerate(p["fixtures"]):
    for arm in ("ordinary", "buffer4") if i % 2 == 0 else ("buffer4", "ordinary"):
        p["schedule"].append(dict(index=len(p["schedule"]), fixture=f, arm=arm))
p["input_hashes"] = {
    **base["input_hashes"],
    **{
        f: sha(f)
        for f in (
            "results/ordinary_long_training_v1/loop.py",
            "results/batch_scale_training_v1/receipt.json",
            "results/batch_scale_training_v1/protocol.json",
            "results/partial_offload_long_v1/protocol.json",
            "results/fp32_training_replication_recovery_v1/source/audit.py",
            "src/core/training_memory.py",
        )
    },
}
p["sources"] = {
    f.as_posix(): sha(f) for f in [*ROOT.glob("*.py"), Path("research/batch_scale_long_plan.md")]
}
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in p["schedule"]:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(ROOT / "protocol.json", p)
print("H156 frozen: 12 fresh batch16 runs; 9600 updates; 78643200 targets.")
