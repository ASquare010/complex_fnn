"""Freeze evidence and the native-buffer hypothesis before GPU execution."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_loss_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/compact_long_training_v1/receipt.json")["files"])
prior = read("results/checkpoint_input_offload_v1/protocol.json")
for key in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(prior[key])
for arm in ("native", "reuse"):
    (ROOT / arm / "runs").mkdir(parents=True)
(ROOT / "CURRENT_STATE.before.md").write_bytes(Path("research/CURRENT_STATE.md").read_bytes())
files = [*ROOT.glob("*.py"), Path("research/native_buffer_loss_plan.md")]
inputs = [
    "results/compact_long_training_v1/receipt.json",
    "results/checkpoint_input_offload_v1/protocol.json",
    "results/checkpoint_input_offload_v1/result.json",
    "results/gradient_staging_v1/study.py",
]
p = dict(
    study="H127",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=prior["maintained_files"],
    fixtures=[f for f in prior["fixtures"] if f["loss_policy"] == "fp32_default_native"],
    repetitions=10,
    warmup=3,
    maximum_backwards=60,
    training_updates=0,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    host_peak_max=128 * 2**20,
    current_state_before=sha("research/CURRENT_STATE.md"),
    broad_goal_achieved=False,
)
assert len(p["fixtures"]) == 2
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen H127: maximum 60 backwards, zero updates; exactness-first gate.")
