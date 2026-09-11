"""Preload CPU compiler dependencies before CUDA; invoke unchanged H110 worker."""

import sys
from pathlib import Path

import sympy
import torch
import torch._dynamo

assert not torch.cuda.is_initialized(), "Preload must precede CUDA model/data allocation"
print(
    f"Dependency preload passed: torch={torch.__version__}, sympy={sympy.__version__}", flush=True
)
if len(sys.argv) == 1:
    raise SystemExit(0)

from results.token_memory_duration_v1.source import worker

worker.ROOT = Path("results/token_memory_duration_recovery_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
args = sys.argv[1:]
worker.run(int(args[0]), int(args[1]), int(args[2]), args[3])
