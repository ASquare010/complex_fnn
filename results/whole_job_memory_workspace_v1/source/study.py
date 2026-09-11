"""Clear demonstrated library workspaces between trials; count them within trials."""

import gc
from pathlib import Path

import torch

from results.whole_job_memory_continuation_v1.source import study as original
from src.core.reproducibility import write_json

ROOT = Path("results/whole_job_memory_workspace_v1")
boundary = original.clear_boundary


def clear_boundary(label):
    gc.collect()
    before = torch.cuda.memory_allocated()
    torch._C._cuda_clearCublasWorkspaces()
    row = boundary(label)
    row["before_cublas_clear_allocated_bytes"] = before
    row["workspace_bytes_subtracted_from_trial_measurements"] = 0
    return row


original.ROOT = ROOT
original.clear_boundary = clear_boundary
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    original.run()
except Exception:
    import traceback

    write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
    write_json(ROOT / "coordinator_status.json", {"status": "FAILED", "see": "failure.json"})
    raise
