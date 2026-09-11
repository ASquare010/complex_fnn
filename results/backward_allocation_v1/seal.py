"""Seal compact artifact visibility and verify the final receipt."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, sha

root = Path("results/backward_allocation_v1")
p = root / "receipt.json"
r = json.loads(p.read_text())
hashes(r["files"])
for name in (".gitignore", "seal.py"):
    r["files"][(root / name).as_posix()] = sha(root / name)
p.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
hashes(r["files"])
print("Verified evidence; original allocated-peak gate remains failed.")
