"""Reference scalar bijection and analytical reconstructed VJP; no custom autograd."""

import torch


def phi(x, a):
    return x + a * (x / (1 + x.abs()))


def inverse(y, a):
    v = y.abs()
    b = 1 + a - v
    disc = torch.hypot(b, 2 * v.sqrt())
    safe = torch.where(b >= 0, disc + b, torch.ones_like(b))
    root = torch.where(b >= 0, 2 * v / safe, 0.5 * (disc - b))
    return y.sign() * root


def forward(x, q, theta, bias, rho):
    for j in range(len(q)):
        x = phi(x @ q[j].T + bias[j], rho * theta[j].tanh())
    return x


@torch.no_grad()
def reverse(y, g, q, theta, bias, rho):
    gt, gb = torch.empty_like(theta), torch.empty_like(bias)
    x = y
    for j in reversed(range(len(q))):
        t = theta[j].tanh()
        a = rho * t
        z = inverse(x, a)
        dz = g * (1 + a / (1 + z.abs()).square())
        gt[j] = (g * z / (1 + z.abs())).sum(0) * rho * (1 - t.square())
        gb[j] = dz.sum(0)
        g = dz @ q[j]
        x = (z - bias[j]) @ q[j]
    return x, g, gt, gb
