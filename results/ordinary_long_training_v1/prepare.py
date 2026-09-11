"""Freeze fresh fixtures, balanced order and unchanged long-run gates."""

import ast
import json
import runpy
from pathlib import Path
from unittest.mock import patch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/ordinary_long_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/ordinary_complete_training_v1/receipt.json")["files"])
old = read("results/fp32_training_replication_recovery_v1/protocol.json")
for field in ("sources", "maintained_files"):
    hashes(old[field])
recent = read("results/ordinary_complete_training_v1/protocol.json")
for values in recent["base_verified"].values():
    hashes(values)
captured = []
with patch.object(Path, "write_text", lambda self, text, **kwargs: captured.append((self, text))):
    runpy.run_path(str(ROOT / "derive.py"))
assert len(captured) == 1 and ast.dump(ast.parse(captured[0][1])) == ast.dump(
    ast.parse((ROOT / "loop.py").read_text())
)
fixtures = old["fixtures"]
assert len(fixtures) == 6 and all(f["source_step"] == 0 for f in fixtures)
orders = (
    ("ordinary", "native", "reuse"),
    ("native", "reuse", "ordinary"),
    ("reuse", "ordinary", "native"),
)
schedule = []
for i, f in enumerate(fixtures):
    order = orders[i % 3] if f["dataset"] == "wikitext2" else tuple(reversed(orders[i % 3]))
    for arm in order:
        schedule.append(dict(index=len(schedule), fixture=f, arm=arm))
inputs = [
    "results/ordinary_complete_training_v1/receipt.json",
    "results/ordinary_complete_training_v1/protocol.json",
    "results/fp32_training_replication_recovery_v1/protocol.json",
    "results/fp32_training_replication_recovery_v1/source/study.py",
    "results/fp32_training_replication_recovery_v1/source/audit.py",
    "results/fp32_training_replication_v1/source/initialize.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/source/offload.py",
    "results/compact_training_v1/study.py",
]
p = dict(
    study="H136",
    previous_goal_turn="progress",
    fixtures=fixtures,
    schedule=schedule,
    datasets=old["datasets"],
    maintained_files=old["maintained_files"],
    base_verified=recent["base_verified"],
    checkpoint_hashes={f["checkpoint"]: sha(f["checkpoint"]) for f in fixtures},
    sources={
        f.as_posix(): sha(f)
        for f in [*ROOT.glob("*.py"), Path("research/ordinary_long_training_plan.md")]
    },
    input_hashes={f: sha(f) for f in inputs},
    training_updates=14400,
    training_targets=58982400,
    study_backwards=14418,
    audit_backwards=18,
    backwards=14436,
    study_scores=72,
    audit_scores=60,
    tensor_artifacts=72,
    memory_ratio_max=0.9,
    time_ratio_max=1.15,
    stability_max=1.15,
    nll_ratio_max=1.01,
    host_peak_max=128 * 2**20,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    derivation_verified=True,
    broad_goal_achieved=False,
)
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in schedule:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
(ROOT / "protocol.json").write_text(json.dumps(p) + "\n", encoding="utf-8")
print("H136 frozen:18 fresh runs,14400 updates,three seeds,two corpora")
