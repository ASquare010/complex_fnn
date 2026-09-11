"""Freeze a small allocator diagnostic, preserving H121 evidence."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/backward_allocation_v1")
assert not (ROOT / "protocol.json").exists()
prior = read("results/checkpoint_input_offload_v1/protocol.json")
receipt = read("results/verification/checkpoint_input_offload_final_v1.json")
for field in ("files", "tensor_hashes", "local_metric_hashes"):
    hashes(receipt[field])
for path, info in receipt["packed"].items():
    assert sha(path) == info["sha256"]
for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(prior[field])
(ROOT / "runs").mkdir(exist_ok=False)
(ROOT / "CURRENT_STATE.before.md").write_bytes(Path("research/CURRENT_STATE.md").read_bytes())
inputs = [
    "results/verification/checkpoint_input_offload_final_v1.json",
    "results/checkpoint_input_offload_v1/protocol.json",
    "results/checkpoint_input_offload_v1/result.json",
]
sources = [*ROOT.glob("*.py"), Path("research/backward_allocation_plan.md")]
p = dict(
    study="H122",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in sources},
    inputs={f: sha(f) for f in inputs},
    maintained_files=prior["maintained_files"],
    fixtures=[f for f in prior["fixtures"] if f["loss_policy"].endswith("chunks")],
    backwards=12,
    training_updates=0,
    diagnostic_targets=49152,
    max_entries=100000,
    broad_goal_achieved=False,
    current_state_before=sha("research/CURRENT_STATE.md"),
)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen four cases / twelve backwards; no updates.")
