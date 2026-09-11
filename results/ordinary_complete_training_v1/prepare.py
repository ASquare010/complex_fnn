"""Freeze ordinary-policy controls and bounded native-noise calibration."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/ordinary_complete_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/paired_complete_training_v1/receipt.json")["files"])
base = read("results/paired_complete_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes")}
p.update(
    study="H135",
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    calibration=dict(
        global_floor=1e-6, global_cap=1e-5, tensor_floor=1e-5, tensor_cap=1e-4, multiplier=10
    ),
)
inputs = [
    "results/paired_complete_training_v1/receipt.json",
    "results/paired_complete_training_v1/protocol.json",
    "results/paired_complete_training_v1/loop.py",
    "results/paired_complete_training_v1/derivation_audit.json",
    "results/optimizer_memory_v1/source/audit.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_layout_v1/receipt.json",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/source/offload.py",
    "results/compact_training_v1/study.py",
]
assert read("results/native_buffer_layout_v1/receipt.json")["operator_qualification"]
p["input_hashes"] = {f: sha(f) for f in inputs}
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [*ROOT.glob("*.py"), Path("research/ordinary_complete_training_plan.md")]
}
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in p["schedule"]:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("H135 frozen: ordinary policy, fixed numerical caps, 360 updates")
