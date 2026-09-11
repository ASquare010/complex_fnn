"""Preserve the failed audit; use the recorded partition for teacher references.

The original comparison changed teacher GEMM batch512 to257 and failed on one
near-zero value by 0.086e-6 above its absolute tolerance. This recovery does not
relax any tolerance. Teacher value/JVP verification keeps batch512; all student
checkpoint rescoring still uses257, direct tensor algebra and automatic JVPs.
"""

import hashlib
import json
from pathlib import Path

root = Path("results/sobolev_selection_v1")
before = root / "audit_recovery_before.json"
assert not before.exists() and not (root / "audit.json").exists()
paths = [root / name for name in ("audit.log", "audit_exit.txt", "audit_failure.json", "result.json", "protocol.json", "source/audit.py")]
before.write_text(json.dumps({"change": "Teacher-reference verification uses its original batch512; student rescoring remains257 and all tolerances unchanged", "training_repeated": False, "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2) + "\n")

import torch

from results.sobolev_selection_v1.source import audit

original_automatic = audit.automatic


def reference_partition(state, x, v, feature, batch):
    if not feature and "down.bias" not in state:
        batch = 512
    return original_automatic(state, x, v, feature, batch)


audit.automatic = reference_partition
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    audit.run()
except Exception:
    import traceback

    (root / "audit_recovery_failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}, indent=2) + "\n")
    raise
