"""H097: one logged small-matrix CPU/CUDA linalg probe; no data or fitting."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("results/affine_backend_probe_v1")


def record(phase, **extra):
    with (ROOT / "phases.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "pid": os.getpid(), **extra}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(phase, flush=True)


assert not (ROOT / "before.json").exists()
with (ROOT / "before.json").open("x") as stream:
    json.dump(
        {
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reason": "H096 native access violation before any fit record; root cause unknown",
            "dimension": 385,
            "repetitions": 1,
            "optimizer_updates": 0,
        },
        stream,
        indent=2,
    )
record("before_import")
import torch

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
record("after_import")
gen = torch.Generator().manual_seed(9851)
x = torch.randn(1024, 385, generator=gen, dtype=torch.float64)
c = torch.randn(385, 4, generator=gen, dtype=torch.float64)
g = x.T @ x + torch.eye(385, dtype=torch.float64)
outputs = {}
for device in ("cpu", "cuda"):
    gg, cc = g.to(device), c.to(device)
    record(f"{device}_before_eigvalsh")
    eig = torch.linalg.eigvalsh(gg)
    if device == "cuda":
        torch.cuda.synchronize()
    record(f"{device}_after_eigvalsh", minimum=float(eig[0]), maximum=float(eig[-1]))
    record(f"{device}_before_cholesky")
    lower, info = torch.linalg.cholesky_ex(gg)
    assert info.item() == 0
    record(f"{device}_after_cholesky")
    result = torch.cholesky_solve(cc, lower)
    outputs[device] = result.cpu()
    record(f"{device}_after_solve", relative_residual=float((gg @ result - cc).norm() / cc.norm()))
torch.testing.assert_close(outputs["cpu"], outputs["cuda"], atol=1e-10, rtol=1e-10)
record("PASS")
