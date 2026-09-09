"""H070 isolated Givens stage inside a two-factor structured projection."""

import torch
from torch import nn

from src.blockshuffle_ffn import BlockShuffleFFN, BlockShuffleLinear, unshuffle_channels
from src.core.structured_linear import shuffle_channels


def rotate_pairs(z, theta, shift):
    half = z.shape[-1] // 2
    u, v = z[..., :half], z[..., half:]
    if shift:
        v = torch.roll(v, -shift, -1)
    cosine, sine = theta.cos().to(z.dtype), theta.sin().to(z.dtype)
    left = cosine * u - sine * v
    right = sine * u + cosine * v
    if shift:
        right = torch.roll(right, shift, -1)
    return torch.cat((left, right), -1)


class PairRotation(nn.Module):
    def __init__(self, width, shift):
        super().__init__()
        if width % 2 or shift not in (0, 1):
            raise ValueError("Even width and shift0/1 required")
        self.shift = shift
        self.theta = nn.Parameter(torch.zeros(width // 2))

    def forward(self, z):
        if z.shape[-1] != 2 * self.theta.numel():
            raise ValueError("Rotation width mismatch")
        return rotate_pairs(z, self.theta, self.shift)


class RotatedBlockShuffleLinear(BlockShuffleLinear):
    def __init__(self, input_width, output_width, groups, shift):
        super().__init__(input_width, output_width, groups)
        self.rotation = PairRotation(min(input_width, output_width), shift)

    def forward(self, x):
        z = self.rotation(shuffle_channels(self.first(x), self.groups))
        return unshuffle_channels(self.second(z), self.groups)


class RotatedBlockShuffleFFN(BlockShuffleFFN):
    def __init__(self, width, hidden, groups, shift):
        super().__init__(width, hidden, groups)
        self.up = RotatedBlockShuffleLinear(width, hidden, groups, shift)
        self.gate = RotatedBlockShuffleLinear(width, hidden, groups, shift)
        self.down = RotatedBlockShuffleLinear(hidden, width, groups, shift)
