"""Freeze partial-offload hypotheses and current native-trained source states."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/partial_offload_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/ordinary_long_training_v1/receipt.json")["files"])
base = read("results/ordinary_complete_training_v1/protocol.json")
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(base[field])
for values in base["base_verified"].values():
    hashes(values)
p = {k: v for k, v in base.items() if k not in ("sources", "input_hashes", "fixtures", "schedule")}
fixtures = []
for index in (0, 11):
    source = read(f"results/ordinary_long_training_v1/case{index:02d}/case.json")
    assert source["arm"] == "ordinary" and source["fixture"]["seed"] == 101
    c = source["measurement"]
    f = dict(source["fixture"])
    f.update(
        label=f["dataset"] + "__fp32_default_native",
        checkpoint=c["checkpoint"]["path"],
        source_step=800,
        source_policy="H136 ordinary step800",
        source_sha256=c["checkpoint"]["sha256"],
        loss_policy="fp32_default_native",
    )
    fixtures.append(f)
p.update(
    study="H137",
    previous_goal_turn="progress",
    fixtures=fixtures,
    backwards=480,
    training_updates=480,
    training_targets=1966080,
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
)
order = ("ordinary", "buffer0", "buffer4", "buffer8", "buffer8", "buffer4", "buffer0", "ordinary")
p["schedule"] = [
    dict(index=i * 8 + j, fixture=f, arm=arm, repeat=0 if j < 4 else 1)
    for i, f in enumerate(fixtures)
    for j, arm in enumerate(order)
]
inputs = [
    "results/ordinary_long_training_v1/receipt.json",
    "results/ordinary_long_training_v1/io.py",
    "results/ordinary_complete_training_v1/protocol.json",
    "results/ordinary_complete_training_v1/audit.py",
    "results/paired_complete_training_v1/loop.py",
    "results/paired_complete_training_v1/derivation_audit.json",
    "results/optimizer_memory_v1/source/audit.py",
    "results/native_buffer_layout_v1/operator.py",
    "results/native_buffer_loss_v1/operator.py",
    "results/checkpoint_input_offload_v1/source/offload.py",
    "results/compact_training_v1/study.py",
    *[f["checkpoint"] for f in fixtures],
]
p["input_hashes"] = {f: sha(f) for f in inputs}
p["sources"] = {
    f.as_posix(): sha(f)
    for f in [*ROOT.glob("*.py"), Path("research/partial_offload_training_plan.md")]
}
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for row in p["schedule"]:
    (ROOT / f"case{row['index']:02d}" / "runs").mkdir(parents=True)
write_json(ROOT / "protocol.json", p)
print("Frozen H137:16 runs,480 updates,0/4/8 offloaded blocks")
