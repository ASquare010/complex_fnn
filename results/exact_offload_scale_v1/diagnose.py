"""Isolate the missing cuBLAS workspace cleanup without model training."""

import gc
import json
from pathlib import Path

import torch

torch.set_num_threads(4)
a = torch.randn(512, 512, device="cuda")
b = a @ a
torch.cuda.synchronize()
del a, b
gc.collect()
torch.cuda.empty_cache()
before = [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()]
torch._C._cuda_clearCublasWorkspaces()
torch.cuda.empty_cache()
torch.cuda.synchronize()
after = [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()]
result = dict(
    after_original_cleanup=before,
    after_workspace_clear=after,
    optimizer_updates=0,
    backwards=0,
    matrix_products=1,
)
assert before[0] > 0 and after == [0, 0]
with Path("results/exact_offload_scale_v1/cleanup_diagnosis.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(result)
