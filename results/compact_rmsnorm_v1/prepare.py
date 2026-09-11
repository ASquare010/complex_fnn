"""Freeze RMSNorm comparison and inherited numerical references."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_rmsnorm_v1")
assert not (ROOT / "protocol.json").exists()
previous = read("results/gradient_staging_v1/receipt.json")
hashes(previous["files"])
prior = read("results/checkpoint_input_offload_v1/protocol.json")
for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(prior[field])
(ROOT / "CURRENT_STATE.before.md").write_bytes(Path("research/CURRENT_STATE.md").read_bytes())
for kind in ("ordinary", "compact"):
    (ROOT / kind / "runs").mkdir(parents=True, exist_ok=False)
files = [*ROOT.glob("*.py"), Path("research/compact_rmsnorm_plan.md")]
inputs = [
    "results/gradient_staging_v1/receipt.json",
    "results/gradient_staging_v1/study.py",
    "results/checkpoint_input_offload_v1/protocol.json",
    "results/checkpoint_input_offload_v1/result.json",
]
p = dict(
    study="H124",
    previous_goal_turn="progress",
    sources={f.as_posix(): sha(f) for f in files},
    input_hashes={f: sha(f) for f in inputs},
    maintained_files=prior["maintained_files"],
    fixtures=[f for f in prior["fixtures"] if f["loss_policy"].endswith("chunks")],
    repetitions=10,
    warmup=3,
    backwards=86,
    training_updates=0,
    diagnostic_targets=327680,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    host_peak_max=128 * 2**20,
    current_state_before=sha("research/CURRENT_STATE.md"),
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen eight cases, 86 backwards including qualification, zero updates.")
