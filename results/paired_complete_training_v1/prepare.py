"""Freeze H134 without modifying historical source or numerical rules."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/paired_complete_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/paired_workspace_timing_v1/receipt.json")["files"])
base = read("results/native_buffer_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(base[field])
h121 = read("results/checkpoint_input_offload_v1/protocol.json")
for field in ("sources", "input_hashes", "library_hashes"):
    hashes(h121[field])
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes")}
p.update(
    study="H134",
    backwards=360,
    training_updates=360,
    training_targets=1474560,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    base_verified={k: h121[k] for k in ("sources", "input_hashes", "library_hashes")},
)
p["schedule"] = [
    dict(index=i * 6 + j, fixture=f, arm=arm, repeat=0 if j < 3 else 1)
    for i, f in enumerate(p["fixtures"])
    for j, arm in enumerate(("reuse", "ordinary", "native", "native", "ordinary", "reuse"))
]
files = [*ROOT.glob("*.py"), Path("research/paired_complete_training_plan.md")]
p["sources"] = {f.as_posix(): sha(f) for f in files}
inputs = [
    "results/paired_workspace_timing_v1/receipt.json",
    "results/native_buffer_training_v1/protocol.json",
    "results/optimizer_memory_v1/source/profile.py",
    "results/optimizer_memory_v1/source/audit.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/source/offload.py",
    "results/compact_training_v1/study.py",
]
p["input_hashes"] = {f: sha(f) for f in inputs}
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in p["schedule"]:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("Frozen H134: 360 updates, 36 artifacts, original audit thresholds")
