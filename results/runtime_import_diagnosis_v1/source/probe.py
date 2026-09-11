"""CPU-only dependency import and small symbolic computation probe."""

import json
import sys
import time

started = time.perf_counter()
import sympy

x = sympy.Symbol("x")
assert sympy.diff(sympy.sin(x), x) == sympy.cos(x)
import torch
import torch._dynamo

assert not torch.cuda.is_initialized()
print(json.dumps({"python": sys.version, "executable": sys.executable, "sympy": sympy.__version__, "torch": torch.__version__, "cuda_initialized": False, "elapsed_seconds": time.perf_counter()-started}), flush=True)
