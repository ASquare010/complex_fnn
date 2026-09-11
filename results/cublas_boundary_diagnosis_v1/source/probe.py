"""Check whether post-qualification active bytes are cuBLAS workspaces; no SGD."""

import gc
import json
from pathlib import Path

import sympy
import torch
import torch._dynamo

from results.token_memory_duration_v1.source.common import qualify

ROOT = Path("results/cublas_boundary_diagnosis_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert not torch.cuda.is_initialized()
print(f"Imports passed: torch={torch.__version__} sympy={sympy.__version__}", flush=True)
records = []
for repetition in range(2):
    assert torch.cuda.memory_allocated() == 0
    checks = qualify()
    assert len(checks) == 8
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    blocks = [
        {"size": b["size"], "requested_size": b.get("requested_size"), "state": b["state"]}
        for segment in torch.cuda.memory_snapshot()
        for b in segment["blocks"]
        if b["state"] == "active_allocated"
    ]
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after == 0
    record = {
        "repetition": repetition,
        "qualification_cases": 8,
        "after_gc_empty_cache_bytes": before,
        "active_blocks_before_clear": blocks,
        "after_cublas_clear_bytes": after,
        "optimizer_updates": 0,
    }
    records.append(record)
    print(json.dumps(record), flush=True)
(ROOT / "result.json").write_text(
    json.dumps(
        {
            "status": "COMPLETE",
            "records": records,
            "optimizer_updates": 0,
            "native_import_failure_root_cause_proved": False,
        },
        indent=2,
    )
    + "\n"
)
