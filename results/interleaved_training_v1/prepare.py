"""Freeze source states, schedule and exact loop derivation before GPU execution."""

import ast
import shutil
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/interleaved_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/timing_drift_audit_v1/receipt.json")["files"])
for path, digest in read("results/partial_offload_long_v1/receipt.json")["files"].items():
    target = (
        Path("results/timing_drift_audit_v1/CURRENT_STATE.before.md")
        if path == "research/CURRENT_STATE.md"
        else Path("results/timing_drift_audit_v1/README.before.md")
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest


class Normalize(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(
            isinstance(t, ast.Name) and t.id in ("setup_started", "setup_wall_ms")
            for t in node.targets
        ):
            return None
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "fixture"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "seed"
        ):
            return ast.Constant(value=101)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Constant)
            and node.left.value == 101
            and isinstance(node.right, ast.Constant)
            and node.right.value == 10000
        ):
            return ast.Constant(value=10101)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg != "setup_wall_ms"]
        return node


old = ast.parse(Path("results/paired_complete_training_v1/loop.py").read_text())
new = Normalize().visit(ast.parse((ROOT / "loop.py").read_text()))
assert ast.dump(old) == ast.dump(new), "Unexpected loop change"
write_json(
    ROOT / "derivation_audit.json",
    dict(passed=True, changes=["fixture seed metadata", "setup wall timing"]),
)
base = read("results/partial_offload_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes", "fixtures", "schedule")}
fixtures = []
for index in (0, 3, 4, 7, 8, 11):
    row = read(f"results/partial_offload_long_v1/case{index:02d}/case.json")
    assert row["arm"] == "ordinary"
    f = dict(row["fixture"])
    c = row["measurement"]["checkpoint"]
    f.update(
        checkpoint=c["path"],
        source_step=800,
        source_policy="H138 ordinary step800",
        source_sha256=c["sha256"],
        loss_policy="fp32_default_native",
    )
    fixtures.append(f)
p.update(
    study="H140",
    previous_goal_turn="progress",
    fixtures=fixtures,
    backwards=720,
    training_updates=720,
    training_targets=2949120,
    stability_max=1.15,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
    free_disk_bytes=shutil.disk_usage(".").free,
)
assert p["free_disk_bytes"] > 12 * 2**30
schedule = []
for i, f in enumerate(fixtures):
    order = (
        ("ordinary", "buffer4", "buffer4", "ordinary")
        if i % 2 == 0
        else ("buffer4", "ordinary", "ordinary", "buffer4")
    )
    repeats = dict(ordinary=0, buffer4=0)
    for arm in order:
        schedule.append(dict(index=len(schedule), fixture=f, arm=arm, repeat=repeats[arm], round=i))
        repeats[arm] += 1
p["schedule"] = schedule
inputs = dict(base["input_hashes"])
for path in [
    "results/timing_drift_audit_v1/receipt.json",
    "results/partial_offload_long_v1/receipt.json",
    "results/partial_offload_training_v1/offload.py",
    "results/partial_offload_training_v1/audit.py",
    *[f["checkpoint"] for f in fixtures],
]:
    inputs[path] = sha(path)
p["input_hashes"] = inputs
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [*ROOT.glob("*.py"), Path("research/interleaved_training_plan.md")]
}
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in schedule:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(ROOT / "protocol.json", p)
print("Frozen:24 segments,720 updates,6 paired source states; loop derivation passed.")
