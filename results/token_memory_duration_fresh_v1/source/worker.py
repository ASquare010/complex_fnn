"""Bypass bytecode at process launch; qualify or invoke the frozen H110 worker."""

import sys
from pathlib import Path

import sympy
import torch
import torch._dynamo

from results.token_memory_duration_v1.source import worker
from results.token_memory_duration_v1.source.common import qualify
from src.core.reproducibility import write_json

ROOT = Path("results/token_memory_duration_fresh_v1")
assert not torch.cuda.is_initialized()
print(f"CPU dependency preload: torch={torch.__version__}, sympy={sympy.__version__}", flush=True)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if sys.argv[1] == "qualify":
    checks = qualify()
    assert len(checks) == 8
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print("Eight frozen qualification cases passed", flush=True)
else:
    worker.ROOT = ROOT
    args = sys.argv[1:]
    worker.run(int(args[0]), int(args[1]), int(args[2]), args[3])
