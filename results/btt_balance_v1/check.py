"""CPU algebra and numeric gradient checks before H164 model training."""

import json
from pathlib import Path

import torch
from torch.func import functional_call

from results.btt_balance_v1.model import ARMS, BTT, Model, factors

torch.set_num_threads(4)
torch.manual_seed(164)
checks = []
for balanced in (False, True):
    layer = BTT(8, 12, balanced).double()
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    values = (
        layer.right.detach().requires_grad_(),
        layer.left.detach().requires_grad_(),
        layer.gains.detach().requires_grad_(),
    )

    def fn(x, right, left, gains):
        return functional_call(layer, {"right": right, "left": left, "gains": gains}, (x,))

    passed = torch.autograd.gradcheck(fn, (x, *values), eps=1e-6, atol=1e-5, rtol=1e-4)
    error = (layer(x) - x @ layer.dense()).abs().max().item()
    assert passed and error < 1e-12
    checks.append(dict(balanced=balanced, gradcheck=passed, dense_error=error))
assert factors(152, False) == (4, 38) and factors(152, True) == (8, 19)
counts = {arm: sum(p.numel() for p in Model(arm, 1).parameters()) for arm in ARMS}
assert counts == dict(
    wide=19672,
    narrow33=4321,
    narrow40=5224,
    narrow27=3547,
    blockshuffle=3672,
    greedy=5340,
    balanced=4380,
), counts
with Path("results/btt_balance_v1/check.json").open("x") as stream:
    json.dump(
        dict(checks=checks, counts=counts, passed=True, training_updates=0, gpu_work=False),
        stream,
        indent=2,
    )
print(counts, "checks pass")
