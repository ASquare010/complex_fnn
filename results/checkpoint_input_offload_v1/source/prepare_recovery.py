"""Record the failed pre-PyTorch startup and allocate only outstanding replay."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import ROOT, hashes, read, sha

assert not (ROOT / "recovery_protocol.json").exists()
assert not (ROOT / "audit.json").exists()
assert not list((ROOT / "runs").glob("*/replay.*"))
assert (ROOT / "audit_exit.txt").read_text().strip() != "0"
log = (ROOT / "audit.log").read_text()
assert "access violation" in log and "sympy" in log and "independent replay complete" not in log
hashes(read(ROOT / "protocol.json")["sources"])
hashes(read(ROOT / "audit_protocol.json")["files"])
files = [
    ROOT / n
    for n in (
        "audit.log",
        "audit_exit.txt",
        "audit_protocol.json",
        "protocol.json",
        "result.json",
        "qualification.json",
    )
]
files += [
    Path(__file__),
    ROOT / "source/audit_recovery.py",
    Path("research/checkpoint_input_offload_audit_recovery.md"),
]
(ROOT / "recovery_protocol.json").write_text(
    json.dumps(
        dict(
            reason="Native SymPy import access violation before PyTorch import or audit.run; no replay started",
            original_audit_exit=(ROOT / "audit_exit.txt").read_text().strip(),
            completed_backwards=254,
            completed_replays=0,
            remaining_backwards=8,
            additional_training_updates=0,
            repeated_completed_cases=0,
            files={
                f.as_posix() if not f.is_absolute() else f.relative_to(Path.cwd()).as_posix(): sha(
                    f
                )
                for f in files
            },
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Preserved failed startup; eight outstanding replays, no repeated case.")
