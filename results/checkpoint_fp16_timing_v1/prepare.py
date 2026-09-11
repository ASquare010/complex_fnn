"""Freeze H159; reuse the prior loop and numerical audit without edits."""

import shutil
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/checkpoint_fp16_timing_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/checkpoint_host_isolation_v1/receipt.json")["files"])
base = read("results/batch_scale_training_v1/protocol.json")
prior = read("results/checkpoint_fp16_v1/protocol.json")
p = {
    k: base[k]
    for k in (
        "steps",
        "warmup",
        "modes",
        "parameter_relative_tolerance",
        "parameter_absolute_tolerance",
        "clip_tolerance",
        "moment_tolerance",
        "score_tolerance",
    )
}
fixtures = []
for f in prior["fixtures"]:
    fixtures.append(
        dict(**f, label=f["dataset"] + "_s" + str(f["seed"]), loss_policy="fp32_default_native")
    )
schedule = []
for i, f in enumerate(fixtures):
    arms = ["ordinary", "buffer4", "fp16"]
    if i % 2:
        arms.reverse()
    seen = {a: 0 for a in arms}
    for arm in arms + arms[::-1]:
        schedule.append(dict(index=len(schedule), fixture=f, arm=arm, repeat=seen[arm]))
        seen[arm] += 1
p.update(
    study="H159",
    previous_goal_turn="progress",
    fixtures=fixtures,
    schedule=schedule,
    datasets=prior["datasets"],
    maintained_files=prior["maintained_files"],
    input_hashes=prior["input_hashes"],
    training_updates=1080,
    training_targets=8847360,
    prior_receipt=sha("results/checkpoint_host_isolation_v1/receipt.json"),
    free_disk_bytes=shutil.disk_usage(".").free,
)
assert p["free_disk_bytes"] > 16 * 2**30
reused = [
    "results/batch_scale_training_v1/loop.py",
    "results/batch_scale_training_v1/worker.py",
    "results/batch_scale_training_v1/native_audit.py",
    "results/checkpoint_fp16_v1/codec.py",
    "results/compact_training_v1/study.py",
    "results/checkpoint_input_offload_v1/source/common.py",
    "results/fp32_classifier_profile_v1/source/common.py",
    "results/streamed_evaluation_v1/source/evaluation.py",
    "results/adam_update_sensitivity_v1/source/audit.py",
]
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [
        *ROOT.glob("*.py"),
        Path("research/checkpoint_fp16_timing_plan.md"),
        *[Path(v) for v in reused],
    ]
}
for name, path in [("README", "README.md"), ("CURRENT_STATE", "research/CURRENT_STATE.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in schedule:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(ROOT / "protocol.json", p)
print("Frozen:36 segments,1080 updates; reused training loop unchanged.")
