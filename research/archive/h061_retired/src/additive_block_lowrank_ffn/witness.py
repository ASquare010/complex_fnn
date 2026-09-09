"""Explicit finite quadratic construction; unrelated to train-time initialization."""

import torch

from src.additive_block_lowrank_ffn import AdditiveBlockLowRankFFN


@torch.no_grad()
def set_quadratic_witness(ffn: AdditiveBlockLowRankFFN) -> None:
    """Overwrite a disposable module to realize the H058 vector quadratic."""
    d = ffn.width
    if ffn.groups != 8 or ffn.hidden != 8 * d // 3 or ffn.rank != d // 8:
        raise ValueError("The witness requires the specified eight-group full-width recipe")
    for parameter in ffn.parameters():
        parameter.zero_()
    a, b = d // 8, d // 3
    for i in range(d):
        group, j = divmod(i, a)
        positive, negative = group * b + 2 * j, group * b + 2 * j + 1
        for projection in (ffn.up, ffn.gate):
            projection.local.weight[group, 2 * j, j] = 1
            projection.local.weight[group, 2 * j + 1, j] = -1
        ffn.down.right.weight[0, positive] = 0.5
        ffn.down.right.weight[0, negative] = 0.5
        ffn.down.right.weight[1, positive] = 0.5 * (i + 1)
        ffn.down.right.weight[1, negative] = 0.5 * (i + 1)
    for projection in (ffn.up, ffn.gate):
        projection.left.weight[2 * a, 0] = 1
        projection.left.weight[2 * a + 1, 0] = -1
        projection.right.weight[0].fill_(1)
    for j in range(3):
        ffn.down.left.weight[j, j] = 1
    ffn.down.right.weight[2, 2 * a] = 0.5
    ffn.down.right.weight[2, 2 * a + 1] = 0.5
