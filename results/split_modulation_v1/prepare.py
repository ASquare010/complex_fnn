"""Preflight the partition, gradients and matched resource budgets."""

from pathlib import Path

import torch
from torch.func import functional_call

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.split_modulation_v1.model import ARMS, Model

ROOT = Path("results/split_modulation_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/modulated_capacity_v1/receipt.json")["files"])
base = read("results/modulated_capacity_v1/protocol.json")
hashes(base["maintained_files"])
torch.set_num_threads(4)
for arm in ("shared_gate", "split_gate"):
    model = Model(arm, d=12).double()
    with torch.no_grad():
        model.gate.weight.normal_(0, 0.02)
    params = dict(model.named_parameters())
    names = list(params)
    x = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, *ps: functional_call(model, dict(zip(names, ps, strict=True)), (a,)),
        (x, *params.values()),
        fast_mode=True,
    )
    z = torch.nn.functional.gelu(x @ model.up.weight.t() + model.up.bias)
    a, b = (z, z) if arm == "shared_gate" else (z[:, :6], z[:, 6:])
    expected = (
        a @ model.down.weight.t()
        + model.down.bias
        + x * (b @ model.gate.weight.t() + model.gate.bias)
    )
    assert torch.allclose(model(x), expected, rtol=1e-12, atol=1e-12)
a, b = Model("shared_gate"), Model("split_gate")
assert torch.equal(a.up.weight, b.up.weight[:192]) and torch.equal(a.up.bias, b.up.bias[:192])
assert all(torch.equal(x, y) for x, y in zip(a.down.parameters(), b.down.parameters(), strict=True))
probe = torch.randn(4, 384)
assert torch.allclose(a(probe), b(probe), rtol=1e-6, atol=1e-7)
counts = {
    arm: dict(parameters=sum(p.numel() for p in Model(arm).parameters()), macs=Model(arm).macs)
    for arm in ARMS
}
assert counts["split_gate"] == counts["static_diagonal"]
assert (
    counts["split_gate"]["parameters"] == 222240 and counts["shared_gate"]["parameters"] == 222144
)
assert counts["split_gate"]["macs"] == counts["shared_gate"]["macs"] == 221184
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
p = dict(
    study="H145",
    arms=ARMS,
    batches=[128, 2048],
    seeds=[53, 67, 79],
    counts=counts,
    training_updates=432,
    preflight_passed=True,
    maintained_files=base["maintained_files"],
    prior_receipt=sha("results/modulated_capacity_v1/receipt.json"),
    sources={
        f.as_posix(): sha(f)
        for f in [*ROOT.glob("*.py"), Path("research/split_modulation_plan.md")]
    },
)
write_json(ROOT / "protocol.json", p)
print("Preflight passed: both active-gate gradchecks, matched initialization, exact budgets.")
