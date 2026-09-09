"""Evaluation-only affine resets and exact-real factor/bias folding."""

import copy
from types import MethodType

import torch

from src.blockshuffle_ffn import unshuffle_channels
from src.core.structured_linear import shuffle_channels

MODES = ("original", "reset_gain", "reset_offset", "reset_both", "fold_gain", "fold_all")


def bias_forward(self, x):
    hidden = shuffle_channels(self.first(x), self.groups)
    return unshuffle_channels(self.second(hidden), self.groups) + self.output_bias


@torch.no_grad()
def transform(model, mode):
    if mode not in MODES:
        raise ValueError(mode)
    result = copy.deepcopy(model)
    for projection in (result.up, result.gate, result.down):
        if mode in ("reset_gain", "reset_both"):
            projection.curve.theta_a.zero_()
        if mode in ("reset_offset", "reset_both"):
            projection.curve.theta_b.zero_()
        if mode.startswith("fold_"):
            gain = 1 + 0.5*projection.curve.theta_a.tanh()
            projection.first.weight.mul_(gain[:, None, None])
            projection.curve.theta_a.zero_()
        if mode == "fold_all":
            offset = 0.5*projection.curve.theta_b.tanh()
            middle = projection.first.output_width
            b = offset[:, None].expand(projection.groups, middle//projection.groups).reshape(middle)
            c = unshuffle_channels(projection.second(shuffle_channels(b, projection.groups)), projection.groups)
            del projection.curve
            projection.register_buffer("output_bias", c)
            projection.forward = MethodType(bias_forward, projection)
    return result
