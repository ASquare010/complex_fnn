"""Complete Transformer with a signed context predicate per detector group."""

import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
from torch.nn import functional as F

from ffn_experiments.components import Attention, RMSNorm, counts, initialize, parameter_generator
from ffn_experiments.context_activation import ContextActivation
from ffn_experiments.settings import ModelConfig

FORWARD = """template <typename T> T atlas_signed_forward(T u,T v,T bounded) {
    T gate=bounded ? ::tanh(v) : v;
    return u/(T(1)+::exp(-u))*gate;
}"""
BACKWARD = """template <typename T> void atlas_signed_backward(T u,T v,T grad,T bounded,T& du,T& dv) {
    T s=T(1)/(T(1)+::exp(-u)); T gate=bounded ? ::tanh(v) : v;
    du=grad*s*(T(1)+u*(T(1)-s))*gate;
    dv=grad*u*s*(bounded ? T(1)-gate*gate : T(1));
}"""
forward_kernel = _create_jit_fn(FORWARD, bounded=0.0)
backward_kernel = _create_multi_output_jit_fn(BACKWARD, num_outputs=2, bounded=0.0)


class SignedActivation(Function):
    @staticmethod
    def forward(ctx, bank, groups, bounded):
        hidden = bank.shape[-1] - groups
        u = bank[..., :hidden].reshape(*bank.shape[:-1], groups, -1)
        dtype = torch.float64 if bank.dtype == torch.float64 else torch.float32
        v = bank[..., hidden:].to(dtype).unsqueeze(-1)
        ctx.save_for_backward(bank)
        ctx.groups = groups
        ctx.bounded = bounded
        if bank.is_cuda:
            value = forward_kernel(u, v, bounded=float(bounded))
        else:
            value = F.silu(u.to(dtype)) * (v.tanh() if bounded else v)
        return value.flatten(-2).to(bank.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (bank,) = ctx.saved_tensors
        groups = ctx.groups
        hidden = bank.shape[-1] - groups
        u = bank[..., :hidden].reshape(*bank.shape[:-1], groups, -1)
        dtype = torch.float64 if bank.dtype == torch.float64 else torch.float32
        v = bank[..., hidden:].to(dtype).unsqueeze(-1)
        g = gradient.reshape_as(u)
        if bank.is_cuda:
            du, dv = backward_kernel(u, v, g, bounded=float(ctx.bounded))
        else:
            u = u.to(dtype)
            g = g.to(dtype)
            s = u.sigmoid()
            gate = v.tanh() if ctx.bounded else v
            du = g * s * (1 + u * (1 - s)) * gate
            dv = g * u * s * (1 - gate.square() if ctx.bounded else 1)
        return torch.cat((du.flatten(-2).to(bank.dtype), dv.sum(-1).to(bank.dtype)), -1), None, None


class SignedFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden = config.hidden
        self.groups = config.groups
        self.mode = config.ffn
        self.coefficient = (
            nn.Parameter(torch.full((config.hidden,), 1.0 if self.mode == "open_gain" else 0.0))
            if self.mode in ("closed_gain", "open_gain")
            else None
        )
        contexts = config.groups if self.mode != "plain" else 0
        self.up = nn.Linear(config.width, config.hidden + contexts, bias=False)
        self.down = nn.Linear(config.hidden, config.width, bias=False)
        self.projection_parameters = config.width * (2 * config.hidden + contexts)

    def forward(self, x):
        bank = self.up(x)
        if self.mode == "plain":
            value = F.silu(bank)
        elif self.coefficient is not None:
            value = ContextActivation.apply(bank, self.coefficient, self.groups, True)
        else:
            value = SignedActivation.apply(bank, self.groups, self.mode == "signed_tanh")
        return self.down(value)

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        self.up.weight[: self.hidden].normal_(
            0, 0.02, generator=parameter_generator(seed, f"{prefix}.up.weight")
        )
        if self.mode != "plain":
            self.up.weight[self.hidden :].normal_(
                0, 0.02, generator=parameter_generator(seed, f"{prefix}.context.weight")
            )
        self.down.weight.normal_(
            0, 0.02 * residual_scale, generator=parameter_generator(seed, f"{prefix}.down.weight")
        )


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = SignedFFN(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    model_name = "signed_context_transformer"

    def __init__(self, config: ModelConfig, seed=17):
        super().__init__()
        config.validate()
        if config.name != self.model_name:
            raise ValueError("Wrong model name")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width)
        initialize(self, seed)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("Invalid token shape")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    def counts(self):
        return counts(self)
