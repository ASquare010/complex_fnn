"""Freeze the implementation and correctness/resource budgets before CUDA."""

from pathlib import Path

import torch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.fused_reconstruction_v1.model import ARMS, Model
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/fused_reconstruction_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/reversible_training_v1/receipt.json")["files"])
base = read("results/reversible_training_v1/protocol.json")
counts = {
    arm: base["counts"]["learned:reconstruct" if arm == "learned:fused" else arm] for arm in ARMS
}
# One small constructor check avoids repeated full QR setup at freeze time.
m = Model("learned:fused", d=12, depth=4)
assert sum(v.numel() for v in m.parameters()) == 96
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
assert not torch.cuda.is_initialized()
files = [
    *ROOT.glob("*.py"),
    Path("research/fused_reconstruction_plan.md"),
    Path(".venv/Lib/site-packages/triton/runtime/tcc/tcc.exe"),
]
sources = {**base["sources"], **{f.as_posix(): sha(f) for f in files}}
write_json(
    ROOT / "protocol.json",
    dict(
        study="H149",
        arms=ARMS,
        batches=[128, 2048],
        seeds=[223, 239, 251],
        counts=counts,
        training_updates=720,
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/reversible_training_v1/receipt.json"),
        sources=sources,
    ),
)
print("Frozen kernel, compiler, gradient checks and resource protocol.")
