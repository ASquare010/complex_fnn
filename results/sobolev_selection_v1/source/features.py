"""Stream smooth teacher features/JVPs into CPU FP64 sufficient statistics."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PrunedFFN(nn.Module):
    def __init__(self, width: int, hidden: int, activation: str):
        super().__init__()
        self.activation = activation
        self.up = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False) if activation == "swiglu" else None
        self.down = nn.Linear(hidden, width, bias=True)

    def forward(self, x):
        z = self.up(x)
        h = F.gelu(z) if self.gate is None else F.silu(z) * self.gate(x)
        return self.down(h)


def hidden_jvp(model, x, direction):
    z, dz = model.up(x), F.linear(direction, model.up.weight)
    if model.gate is None:
        h = F.gelu(z)
        derivative = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        derivative = derivative + z * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
        return h, derivative * dz
    gate, dgate = model.gate(x), F.linear(direction, model.gate.weight)
    sigmoid = z.sigmoid()
    silu = F.silu(z)
    dsilu = sigmoid * (1 + z * (1 - sigmoid))
    return silu * gate, dsilu * dz * gate + silu * dgate


def value_jvp(model, x, direction):
    h, j = hidden_jvp(model, x, direction)
    return model.down(h), F.linear(j, model.down.weight)


def directions(rows: int, width: int, seed: int, sx: torch.Tensor):
    signs = torch.randint(0, 2, (rows, width), generator=torch.Generator().manual_seed(seed))
    return (signs.float() * 2 - 1) * sx.float()


@torch.no_grad()
def statistics(model, x: torch.Tensor, seed: int, batch: int = 512):
    assert x.device.type == "cpu"
    n, width = x.shape
    hidden = model.up.weight.shape[0]
    sx = x.double().std(0, correction=0).clamp_min(1e-3)
    v = directions(n, width, 27000 + seed, sx)
    hh = torch.zeros(hidden, hidden, dtype=torch.float64)
    kk = torch.zeros_like(hh)
    hy = torch.zeros(hidden, width, dtype=torch.float64)
    kt = torch.zeros_like(hy)
    sum_h, sum_y = torch.zeros(hidden, dtype=torch.float64), torch.zeros(width, dtype=torch.float64)
    sum_y2, sum_t2 = 0.0, 0.0
    model.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, n, batch):
        h, k = hidden_jvp(model, x[start : start + batch].cuda(), v[start : start + batch].cuda())
        y = model.down(h)
        t = F.linear(k, model.down.weight)
        h, k, y, t = [a.cpu().double() for a in (h, k, y, t)]
        assert all(bool(torch.isfinite(a).all()) for a in (h, k, y, t))
        hh.addmm_(h.T, h)
        kk.addmm_(k.T, k)
        hy.addmm_(h.T, y)
        kt.addmm_(k.T, t)
        sum_h.add_(h.sum(0))
        sum_y.add_(y.sum(0))
        sum_y2 += y.square().sum().item()
        sum_t2 += t.square().sum().item()
    peak = torch.cuda.max_memory_allocated()
    model.cpu()
    mean_h, mean_y = sum_h / n, sum_y / n
    cov_h = hh / n - torch.outer(mean_h, mean_h)
    scale_h = cov_h.diag().clamp_min(0).sqrt().clamp_min(1e-6)
    scale_y = max(sum_y2 / (n * width) - mean_y.square().mean().item(), 1e-12) ** 0.5
    value_gram = cov_h / scale_h[:, None] / scale_h[None, :]
    derivative_gram = kk / n / scale_h[:, None] / scale_h[None, :]
    value_cross = (hy / n - torch.outer(mean_h, mean_y)) / scale_h[:, None] / scale_y
    derivative_cross = kt / n / scale_h[:, None] / scale_y
    derivative_energy = sum_t2 / (n * width * scale_y**2)
    return {
        "value_gram": (value_gram + value_gram.T) / 2,
        "derivative_gram": (derivative_gram + derivative_gram.T) / 2,
        "value_cross": value_cross,
        "derivative_cross": derivative_cross,
        "mean_h": mean_h,
        "scale_h": scale_h,
        "mean_y": mean_y,
        "scale_y": scale_y,
        "sx": sx,
        "derivative_energy": derivative_energy,
        "beta": 0.1 / max(derivative_energy, 1e-12),
        "rows": n,
        "peak_cuda_bytes": peak,
    }


@torch.no_grad()
def export(model, state, selected, coefficient):
    """Fold feature/output normalization into the readout; keep raw input rows."""
    width = model.up.weight.shape[1]
    output = PrunedFFN(width, len(selected), model.activation).double()
    output.up.weight.copy_(model.up.weight[selected])
    if output.gate is not None:
        output.gate.weight.copy_(model.gate.weight[selected])
    weights = coefficient / state["scale_h"][selected, None] * state["scale_y"]
    output.down.weight.copy_(weights.T)
    output.down.bias.copy_(state["mean_y"] - state["mean_h"][selected] @ weights)
    return output
