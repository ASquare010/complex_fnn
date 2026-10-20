"""Native256 core blocks, deterministic initialization and budgeted FFN widths.

Memory amplitude handling and low-bit payloads belong to the integrating model.
This module contains no memory-width projection or reconstruction decoder.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.components import RMSNorm, initialize, parameter_generator
PARAMETER_BUDGET = 23209216

@dataclass
class Config:
    width: int = 256
    heads: int = 4
    layers: int = 8
    hidden: int = 1024
    vocab_size: int = 4096
    context: int = 64
    latent_width: int = 256
    rope_base: float = 10000.0
    ffn_hidden_total: int = 38062

class Attention(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        head_dim = c.width // c.heads
        self.register_buffer('frequency', c.rope_base ** (-torch.arange(0, head_dim, 2).float() / head_dim), persistent=False)

    def forward(self, x, valid, positions):
        b, n, d = x.shape
        q, k, v = (t.transpose(1, 2) for t in self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).unbind(2))
        angle = positions[:, None, :, None] * self.frequency[None, None, None]

        def rotate(t):
            a, z = (t[..., ::2], t[..., 1::2])
            return torch.stack((a * angle.cos() - z * angle.sin(), a * angle.sin() + z * angle.cos()), -1).flatten(-2)
        allowed = valid[:, None, None, :] & torch.ones(n, n, device=x.device, dtype=torch.bool).tril()[None, None]
        y = F.scaled_dot_product_attention(rotate(q), rotate(k), v, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape(b, n, d))

class NativeBranchFFN(nn.Module):
    """The selected gated squared-ReLU FFN with a budgeted per-layer width."""

    def __init__(self, c, hidden):
        super().__init__()
        if hidden < 4:
            raise ValueError('FFN hidden width must be at least four')
        self.hidden = hidden
        self.widths = tuple((hidden // 4 + (i < hidden % 4) for i in range(4)))
        self.up = nn.Linear(c.width, hidden + 4, bias=False)
        self.down = nn.Linear(hidden, c.width, bias=False)
        points, weights = np.polynomial.hermite.hermgauss(128)
        variance = c.width * 0.02 ** 2
        z = torch.from_numpy(points) * math.sqrt(2 * variance)
        w = torch.from_numpy(weights) / math.sqrt(math.pi)
        target = float((w * F.silu(z).square()).sum()) * variance * c.hidden / hidden
        self.scale = math.sqrt(target / float((w * F.relu(z).square()).sum()))
        self.route_scale = 1.0
        self.register_buffer('output_scale', torch.ones((), dtype=torch.float64))

    def components(self, z):
        values, logits = z.split((self.hidden, 4), -1)
        values = values.relu().square()
        gates = (2 * logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32).sigmoid()).to(logits.dtype)
        return torch.cat([part * gates[..., i:i + 1] for i, part in enumerate(values.split(self.widths, -1))], -1)

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        self.up.weight.normal_(0, 0.02, generator=parameter_generator(seed, f'{prefix}.up.weight'))
        self.down.weight.normal_(0, 0.02 * residual_scale, generator=parameter_generator(seed, f'{prefix}.down.weight'))
        x = torch.randn(512, self.up.in_features, generator=torch.Generator().manual_seed(9127))
        x = x / x.square().mean(-1, keepdim=True).sqrt()
        z = self.up(x)
        self.route_scale = math.sqrt(float(z[..., :self.hidden].relu().pow(4).mean() / self.components(z).square().mean()))
        self.output_scale.fill_(self.scale * self.route_scale)

    def forward(self, x):
        y = self.down(self.components(self.up(x)))
        return y * self.output_scale.to(dtype=y.dtype)

class Block(nn.Module):

    def __init__(self, c, hidden):
        super().__init__()
        self.attention_norm = RMSNorm(c.width)
        self.attention = Attention(c)
        self.ffn_norm = RMSNorm(c.width)
        self.ffn = NativeBranchFFN(c, hidden)

class Core(nn.Module):

    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        base, extra = divmod(config.ffn_hidden_total, config.layers)
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config, base + (i < extra)) for i in range(config.layers)])
        self.norm = RMSNorm(config.width)
        initialize(self, seed)
