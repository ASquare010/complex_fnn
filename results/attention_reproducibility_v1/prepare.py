"""Freeze the three policy matrix and verify the current authoritative evidence."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/attention_reproducibility_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/native_buffer_layout_v1/receipt.json")["files"])
prior = read("results/checkpoint_input_offload_v1/protocol.json")
for key in ("sources", "input_hashes", "maintained_files", "library_hashes"):
    hashes(prior[key])
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
files = [*ROOT.glob("*.py"), Path("research/attention_reproducibility_plan.md")]
inputs = [
    "results/native_buffer_layout_v1/receipt.json",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/protocol.json",
    "results/checkpoint_input_offload_v1/result.json",
]
p = dict(
    study="H129",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=prior["maintained_files"],
    fixtures=[f for f in prior["fixtures"] if f["loss_policy"] == "fp32_default_native"],
    policies=["original_default", "deterministic_default", "deterministic_math"],
    maximum_backwards=36,
    maximum_artifacts=18,
    maximum_diagnostic_targets=147456,
    training_updates=0,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    broad_goal_achieved=False,
)
assert len(p["fixtures"]) == 2
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("H129 frozen: at most 36 backwards, zero updates.")
