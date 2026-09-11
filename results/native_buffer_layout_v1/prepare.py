"""Freeze the revised arithmetic path and exact same model resource gates."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_layout_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/native_buffer_loss_v1/receipt.json")["files"])
old = read("results/native_buffer_loss_v1/protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(old[key])
p = {k: v for k, v in old.items() if k not in ("sources", "input_hashes", "current_state_before")}
p.update(study="H128", maximum_backwards=62, current_state_before=sha("research/CURRENT_STATE.md"))
files = [*ROOT.glob("*.py"), Path("research/native_buffer_layout_plan.md")]
p["sources"] = {f.as_posix(): sha(f) for f in files}
inputs = [
    "results/native_buffer_loss_v1/receipt.json",
    "results/native_buffer_loss_v1/protocol.json",
    "results/native_buffer_loss_v1/study.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/native_buffer_loss_v1/audit.py",
    "results/gradient_staging_v1/study.py",
]
p["input_hashes"] = {f: sha(f) for f in inputs}
(ROOT / "CURRENT_STATE.before.md").write_bytes(Path("research/CURRENT_STATE.md").read_bytes())
for arm in ("native", "reuse"):
    (ROOT / arm / "runs").mkdir(parents=True)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print("H128 frozen: at most 62 backwards, 0 updates.")
