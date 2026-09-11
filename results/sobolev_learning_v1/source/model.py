"""Loss-normalized training adapter; raw dense deployment stays unchanged."""

import torch
from torch import nn
from torch.nn import functional as F

from results.sobolev_selection_v1.source.features import PrunedFFN, directions, value_jvp

METHODS = ("value_greedy", "value_select_sobolev_fit", "sobolev_greedy", "random")
TRAIN, SELECT, TOTAL = 32768, 36864, 40960
SEEDS, LAYERS, RATES = (71, 83, 97), (0, 3, 7), (0.001, 0.003)


class TrainingAdapter(nn.Module):
    def __init__(self, raw: nn.Module, scale: float):
        super().__init__()
        if scale <= 0:
            raise ValueError("Loss scale must be positive")
        self.raw = raw
        self.scale = float(scale)

    def forward(self, x):
        return self.raw(x) / self.scale

    def optimizer_groups(self, rate):
        return [{"params": self.parameters(), "lr": rate}]

    def features(self, x):
        z = self.raw.up(x)
        h = F.gelu(z) if self.raw.gate is None else F.silu(z) * self.raw.gate(x)
        return z, h


def from_checkpoint(checkpoint):
    model = PrunedFFN(checkpoint["width"], checkpoint["hidden"], checkpoint["activation"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


@torch.no_grad()
def teacher_targets(model, x, sx, seed):
    model.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    values = torch.cat([model(chunk.cuda()).cpu() for chunk in x.split(512)])
    probes = [
        directions(len(x) - SELECT, x.shape[1], offset + seed, sx) for offset in (32000, 33000)
    ]
    derivatives = []
    for probe in probes:
        blocks = []
        for start in range(0, len(probe), 512):
            _, derivative = value_jvp(
                model,
                x[SELECT + start : SELECT + start + 512].cuda(),
                probe[start : start + 512].cuda(),
            )
            blocks.append(derivative.cpu())
        derivatives.append(torch.cat(blocks))
    peak = torch.cuda.max_memory_allocated()
    model.cpu()
    return values, torch.stack(derivatives), probes, peak


@torch.no_grad()
def endpoint_scores(model, x, y, target_j, probes, sy):
    model.cuda().eval()
    losses = {}
    for name, begin, end in (("selection_mse", TRAIN, SELECT), ("reporting_mse", SELECT, len(x))):
        errors = []
        for start in range(begin, end, 512):
            value = model(x[start : min(start + 512, end)].cuda()).cpu().double()
            errors.append((value - y[start : min(start + 512, end)].double()).square().mean(1))
        error = torch.cat(errors)
        losses[name] = error.mean().item() / sy**2
        if name == "reporting_mse":
            losses["max_value_sample_nmse"] = error.max().item() / sy**2
    squares = []
    for probe_index, probe in enumerate(probes):
        blocks = []
        for start in range(0, len(probe), 512):
            _, jvp = value_jvp(
                model,
                x[SELECT + start : SELECT + start + 512].cuda(),
                probe[start : start + 512].cuda(),
            )
            blocks.append(
                (jvp.cpu().double() - target_j[probe_index, start : start + 512].double())
                .square()
                .mean(1)
            )
        squares.append(torch.cat(blocks))
    errors = torch.stack(squares)
    energy = target_j.double().square().mean(-1)
    losses["derivative_relative_mse"] = errors.mean().item() / energy.mean().item()
    losses["max_derivative_sample_relative_mse"] = (errors / energy.clamp_min(1e-12)).max().item()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    model.cpu()
    return losses
