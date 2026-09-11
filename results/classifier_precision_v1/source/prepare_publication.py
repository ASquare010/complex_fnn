"""Preserve the last mutable historical receipt input before updating Git policy."""

import hashlib
import json
from pathlib import Path

root = Path("results/classifier_precision_v1")
receipt = json.loads(Path("results/verification/whole_job_memory_final_v1.json").read_text())
original = Path(".gitignore").read_bytes()
assert hashlib.sha256(original).hexdigest() == receipt["files"][".gitignore"]
target = root / "before_documents/gitignore"
with target.open("xb") as handle:
    handle.write(original)
print("Preserved unchanged H112 Git-policy snapshot")
