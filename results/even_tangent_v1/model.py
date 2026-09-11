"""Explicit two-sided scalar model and analytic scalar-output tangents."""

import torch


def evaluate(s, x, probe, eta=None, theta_delta=None, bias_delta=None, output_bias=0.0):
    theta = s["core.theta"] + (0 if theta_delta is None else theta_delta)
    bias = s["core.bias"] + (0 if bias_delta is None else bias_delta)
    for j in range(4):
        z = x @ s["core.q"][j].T + bias[j]
        t = theta[j] if eta is None else theta[j] + torch.where(z >= 0, eta[j], -eta[j])
        x = z + 0.5 * t.tanh() * z / (1 + z.abs())
    return x @ s["readout.weight"][probe] + s["readout.bias"][probe] + output_bias


def features(s, x, probe):
    zs = []
    h = x
    for j in range(4):
        z = h @ s["core.q"][j].T + s["core.bias"][j]
        zs.append(z)
        h = z + 0.5 * s["core.theta"][j].tanh() * z / (1 + z.abs())
    g = s["readout.weight"][probe].expand(len(x), -1)
    theta = [None] * 4
    eta = [None] * 4
    bias = [None] * 4
    for j in reversed(range(4)):
        z = zs[j]
        t = s["core.theta"][j].tanh()
        theta[j] = g * z / (1 + z.abs()) * 0.5 * (1 - t.square())
        eta[j] = g * z.abs() / (1 + z.abs()) * 0.5 * (1 - t.square())
        bias[j] = g * (1 + 0.5 * t / (1 + z.abs()).square())
        g = bias[j] @ s["core.q"][j]
    return {
        name: torch.cat(vals, dim=1)
        for name, vals in [("theta", theta), ("eta", eta), ("bias", bias)]
    }
