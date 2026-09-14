"""Pre-GPU float64 equivalence including all input and parameter derivatives."""

import json
from pathlib import Path

import torch
from torch.func import functional_call

from results.btt_balance_v1.model import BTT
from results.btt_execution_v1.model import Model

torch.set_num_threads(4)
checks = []
for seed in (431, 443, 457):
    a = Model(8, 12, seed, "direct").double()
    b = Model(8, 12, seed, "materialized").double()
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    u = torch.randn(5, 8, dtype=torch.float64)
    ya, yb = a(x), b(x)
    ga = torch.autograd.grad((ya * u).sum(), (x, *a.parameters()))
    gb = torch.autograd.grad((yb * u).sum(), (x, *b.parameters()))
    forward_error = (ya - yb).abs().max().item()
    gradient_error = max((v - w).abs().max().item() for v, w in zip(ga, gb, strict=True))
    assert forward_error < 1e-12 and gradient_error < 1e-11
    checks.append(dict(seed=seed, forward_error=forward_error, gradient_error=gradient_error))
layer = BTT(8, 12, True).double()
params = dict(layer.named_parameters())
x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)


def fn(x, *values):
    # Functional module returns its materialized matrix when applied to the identity.
    replacements = dict(zip(params, values, strict=True))

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return x @ self.layer.dense()

    return functional_call(Wrapper(), {"layer." + k: v for k, v in replacements.items()}, (x,))


assert torch.autograd.gradcheck(fn, (x, *params.values()), atol=1e-5, rtol=1e-4)
with Path("results/btt_execution_v1/check.json").open("x") as stream:
    json.dump(
        dict(passed=True, checks=checks, gradcheck=True, gpu_work=False, optimizer_updates=0),
        stream,
        indent=2,
    )
print("CPU execution-order checks passed")
