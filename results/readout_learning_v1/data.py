"""Deterministic CPU datasets, training-only target normalization and common batches."""

import torch

TASKS = ("linear", "teacher", "product")


def make_data(task, seed):
    idx = TASKS.index(task)
    gen = torch.Generator().manual_seed(1000 + seed + 10000 * idx)
    x = torch.randn(8192, 32, generator=gen)
    teacher = torch.Generator().manual_seed(71000 + idx)
    if task == "teacher":
        w = torch.randn(128, 32, generator=teacher) / 32**0.5
        v = torch.randn(32, 128, generator=teacher) / 128**0.5
        y = torch.nn.functional.gelu(x @ w.T) @ v.T
    else:
        q = torch.linalg.qr(torch.randn(32, 32, generator=teacher)).Q
        y = (x if task == "linear" else x * torch.roll(x, -1, dims=1)) @ q
    mean = y[:4096].mean(0)
    scale = (y[:4096] - mean).square().mean().sqrt()
    y = (y - mean) / scale
    order = torch.randint(
        4096, (300, 128), generator=torch.Generator().manual_seed(90000 + seed + idx)
    )
    return dict(x=x, y=y, mean=mean, scale=scale, order=order)
