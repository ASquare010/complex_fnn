"""Use native unchunked H110 audit on all completed mixed-root evidence."""

from pathlib import Path

import torch

from results.token_memory_duration_v1.source import audit
from src.core.reproducibility import write_json

audit.ROOT = Path("results/token_memory_duration_fresh_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    audit.run()
except Exception:
    import traceback

    write_json(audit.ROOT / "audit_failure.json", {"traceback": traceback.format_exc()})
    raise
