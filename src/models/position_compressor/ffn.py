"""Checkpoint-safe use of the selected Branch Sigmoid FFN."""

import torch

from models.branch_sigmoid.transformer import BranchFFN


class CompressorBranchFFN(BranchFFN):
    """Selected FFN with checkpointed calibration for seed-independent loading."""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("output_scale", torch.ones((), dtype=torch.float64))

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        super().initialize(seed, prefix, residual_scale)
        self.output_scale.fill_(self.scale * self.route_scale)

    def forward(self, x):
        y = self.down(self.components(self.up(x)))
        return y * self.output_scale.to(dtype=y.dtype)
