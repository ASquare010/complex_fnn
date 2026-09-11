"""Freeze the maintained-helper batch-scaling study and exact source derivations."""

import ast
import shutil
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/batch_scale_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/training_memory_integration_v1/receipt.json")["files"])
old = Path("results/interleaved_training_v1/loop.py").read_text()
expected = (
    old.replace("batch_size=8", "batch_size=16")
    .replace("data.batch(8, 512)", "data.batch(16, 512)")
    .replace("training_targets=30 * 4096", "training_targets=30 * 8192")
)
assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((ROOT / "loop.py").read_text()))
old = Path("results/optimizer_memory_v1/source/audit.py").read_text()
expected = old.replace("data.batch(8, 512)", "data.batch(16, 512)")
assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((ROOT / "native_audit.py").read_text()))
base = read("results/interleaved_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes")}
p.update(
    study="H142",
    previous_goal_turn="progress",
    batch_size=16,
    validation_batch_size=8,
    training_targets=5898240,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    free_disk_bytes=shutil.disk_usage(".").free,
)
assert p["free_disk_bytes"] > 12 * 2**30
p["maintained_files"] = dict(base["maintained_files"])
for path in ("src/core/training_memory.py", "tests/test_training_memory.py"):
    p["maintained_files"][path] = sha(path)
inputs = dict(base["input_hashes"])
for path in (
    "results/training_memory_integration_v1/receipt.json",
    "results/interleaved_training_v1/loop.py",
    "results/optimizer_memory_v1/source/audit.py",
    "results/interleaved_training_v1/protocol.json",
):
    inputs[path] = sha(path)
p["input_hashes"] = inputs
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [*ROOT.glob("*.py"), Path("research/batch_scale_training_plan.md")]
}
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in p["schedule"]:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(
    ROOT / "derivation_audit.json",
    dict(
        passed=True,
        training_changes=["batch size", "target count"],
        native_audit_changes=["batch size"],
    ),
)
write_json(ROOT / "protocol.json", p)
print("Frozen batch16:24 segments,720 updates,5898240 targets; exact derivation checks passed.")
