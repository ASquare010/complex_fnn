"""Freeze sources and run independent rank/derivative preflight on CPU."""

import math
from pathlib import Path

import torch
from torch.func import functional_call
from torch.nn import functional as F

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.conjugated_ffn_v1.model import ARMS, Model
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/conjugated_ffn_v1")
torch.set_num_threads(4)
hashes(read("results/batch_scale_training_v1/receipt.json")["files"])
assert not (ROOT / "protocol.json").exists()
# Check shared parameter derivatives, including both uses, independently.
m = Model("shared_permuted", d=6, seed=7).double()
parameters = dict(m.named_parameters())
names = list(parameters)
x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(
    lambda a, *ps: functional_call(m, dict(zip(names, ps, strict=True)), (a,)),
    (x, *parameters.values()),
    fast_mode=True,
)
P = torch.eye(6, dtype=torch.float64)[:, m.permutation]


def branch(v):
    return (
        F.gelu(v @ m.first.up.weight.t() + m.first.up.bias) @ m.first.down.weight.t()
        + m.first.down.bias
    )


z = x + branch(x) / math.sqrt(2)
reference = z + branch(z @ P) @ P.t() / math.sqrt(2) - x
assert torch.allclose(m(x), reference, rtol=1e-12, atol=1e-12)
# Coordinate-half witness uses no learned bias and disjoint output subspaces.
witness = Model("shared_permuted", d=6, seed=7).double()
with torch.no_grad():
    witness.first.up.weight.copy_(torch.eye(6, dtype=torch.float64)[:3])
    witness.first.down.weight.copy_(torch.eye(6, dtype=torch.float64)[:, :3])
    witness.first.up.bias.zero_()
    witness.first.down.bias.zero_()
    witness.permutation.copy_(torch.tensor([3, 4, 5, 0, 1, 2]))
    witness.inverse.copy_(witness.permutation)
probe = torch.cat([torch.zeros(1, 6, dtype=torch.float64), torch.eye(6, dtype=torch.float64)])
y = witness(probe)
assert torch.allclose(y, F.gelu(probe) / math.sqrt(2), rtol=1e-12, atol=1e-12)
assert int(torch.linalg.matrix_rank(y - y.mean(0))) == 6
same = Model("shared_same", d=6, seed=7).double()
y = same(torch.randn(80, 6, dtype=torch.float64))
assert int(torch.linalg.matrix_rank(y - y.mean(0))) <= 3
counts = {
    arm: dict(
        parameters=sum(p.numel() for p in Model(arm).parameters()), macs=Model(arm).mac_per_token
    )
    for arm in ARMS
}
assert counts["shared_permuted"]["parameters"] == counts["narrow_gelu"]["parameters"]
assert (
    counts["shared_permuted"]["macs"]
    == counts["full_gelu"]["macs"]
    == counts["full_swiglu"]["macs"]
)
assert counts["shared_permuted"]["parameters"] <= 0.6 * min(
    counts[k]["parameters"] for k in ("full_gelu", "full_swiglu")
)
base = read("results/batch_scale_training_v1/protocol.json")
hashes(base["maintained_files"])
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
p = dict(
    study="H143",
    arms=ARMS,
    seeds=[17, 29, 43],
    tasks=["linear", "gelu_teacher", "cubic"],
    steps=300,
    training_updates=16200,
    counts=counts,
    preflight_passed=True,
    maintained_files=base["maintained_files"],
    sources={
        f.as_posix(): sha(f) for f in [*ROOT.glob("*.py"), Path("research/conjugated_ffn_plan.md")]
    },
    prior_receipt=sha("results/batch_scale_training_v1/receipt.json"),
)
write_json(ROOT / "protocol.json", p)
print(
    "Preflight passed: tied-weight gradcheck, explicit permutation algebra, rank witness, exact counts."
)
