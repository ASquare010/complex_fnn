"""Derive the startup-only recovery without editing any original frozen source."""

from pathlib import Path

OLD = Path("results/fp32_training_replication_v1")
ROOT = Path("results/fp32_training_replication_recovery_v1")
text = (OLD / "source/study.py").read_text()
block = (
    "    initials = generate_initials(protocol)\n"
    '    write_json(ROOT / "initializations.json", initials)\n'
    '    protocol["checkpoint_hashes"] = {r["path"]: r["sha256"] for r in initials["initializations"]}\n'
)
assert text.count(block) == 1
text = text.replace(block, "")
assert text.count("    env = environment()") == 1
text = text.replace("    env = environment()", block + "    env = environment()")
text = text.replace(
    'ROOT = Path("results/fp32_training_replication_v1")',
    'ROOT = Path("results/fp32_training_replication_recovery_v1")',
)
for name, content in (
    ("study.py", text),
    (
        "launch.py",
        (OLD / "source/launch.py")
        .read_text()
        .replace("fp32_training_replication_v1", "fp32_training_replication_recovery_v1"),
    ),
    (
        "audit.py",
        (OLD / "source/audit.py")
        .read_text()
        .replace(
            'ROOT = Path("results/fp32_training_replication_v1")',
            'ROOT = Path("results/fp32_training_replication_recovery_v1")',
        ),
    ),
    (
        "prepare_audit.py",
        (OLD / "source/prepare_audit.py")
        .read_text()
        .replace(
            'ROOT = Path("results/fp32_training_replication_v1")',
            'ROOT = Path("results/fp32_training_replication_recovery_v1")',
        ),
    ),
    (
        "inspect_progress.py",
        (OLD / "source/inspect_progress.py")
        .read_text()
        .replace("fp32_training_replication_v1", "fp32_training_replication_recovery_v1"),
    ),
):
    target = ROOT / "source" / name
    assert not target.exists()
    target.write_text(content, encoding="utf-8", newline="\n")
print("Derived recovery: CPU initialization now precedes CUDA metadata collection")
