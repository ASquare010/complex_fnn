"""Freeze the fresh FP16 comparison without changing the existing training loop."""

import shutil
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/checkpoint_fp16_long_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/checkpoint_fp16_timing_v1/receipt.json")["files"])
base = read("results/batch_scale_long_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(base[field])
p = {k: v for k, v in base.items() if k not in ("sources", "schedule", "prior_receipt")}
p.update(
    study="H160",
    previous_goal_turn="progress",
    prior_receipt=sha("results/checkpoint_fp16_timing_v1/receipt.json"),
    preflight_backwards=12,
    backwards=9636,
    tensor_artifacts=60,
    free_disk_bytes=shutil.disk_usage(".").free,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
)
assert p["free_disk_bytes"] > 8 * 2**30
p["schedule"] = []
for i, f in enumerate(p["fixtures"]):
    for arm in ("ordinary", "fp16") if i % 2 == 0 else ("fp16", "ordinary"):
        p["schedule"].append(dict(index=len(p["schedule"]), fixture=f, arm=arm))
reused = [
    "results/batch_scale_long_v1/worker.py",
    "results/checkpoint_fp16_v1/codec.py",
    "results/fp32_training_replication_recovery_v1/source/audit.py",
    "results/fp32_decoder_resource_v1/source/audit.py",
    "results/ordinary_long_training_v1/loop.py",
    "results/compact_training_v1/study.py",
]
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [
        *ROOT.glob("*.py"),
        Path("research/checkpoint_fp16_long_plan.md"),
        *[Path(v) for v in reused],
    ]
}
for name, path in (("README", "README.md"), ("CURRENT_STATE", "research/CURRENT_STATE.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in p["schedule"]:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(ROOT / "protocol.json", p)
print("Frozen:12 preflight backwards,12 fresh800-update runs,12 native audit backwards.")
