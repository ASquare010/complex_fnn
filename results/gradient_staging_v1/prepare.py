"""Freeze the storage experiment and verify the preceding evidence."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/gradient_staging_v1")
assert not (ROOT / "protocol.json").exists()
previous = read("results/backward_allocation_v1/receipt.json")
hashes(previous["files"])
prior = read("results/checkpoint_input_offload_v1/protocol.json")
for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(prior[field])
(ROOT / "runs").mkdir(exist_ok=False)
(ROOT / "CURRENT_STATE.before.md").write_bytes(Path("research/CURRENT_STATE.md").read_bytes())
inputs = [
    Path("results/backward_allocation_v1/receipt.json"),
    Path("results/checkpoint_input_offload_v1/protocol.json"),
    Path("results/checkpoint_input_offload_v1/result.json"),
    Path(".venv/Lib/site-packages/torch/_tensor.py"),
]
files = [*ROOT.glob("*.py"), Path("research/gradient_staging_plan.md")]
p = dict(
    study="H123",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f.as_posix(): sha(f) for f in inputs},
    maintained_files=prior["maintained_files"],
    fixtures=[f for f in prior["fixtures"] if f["loss_policy"].endswith("chunks")],
    repetitions=10,
    warmup=3,
    backwards=42,
    training_updates=0,
    diagnostic_targets=163840,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    host_peak_max=128 * 2**20,
    current_state_before=sha("research/CURRENT_STATE.md"),
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen four cases / 42 backwards including toy; zero updates.")
