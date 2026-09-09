"""Experimental correction-only fusion; retain native SiLU and gate backward."""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from src.learnable_activation_ffn import LearnableBlockShuffleFFN, RationalResidualActivation
from src.learnable_activation_ffn.compiled_product import _compiled_forward


class RationalCorrection(nn.Module):
    def __init__(self, curve):
        super().__init__()
        self.curve = curve

    def forward(self, u):
        grouped = u.float().reshape(
            *u.shape[:-1], self.curve.groups, self.curve.hidden // self.curve.groups
        )
        return self.curve.residual(grouped).reshape_as(u).to(u.dtype)


class NativeSiluRationalProduct(nn.Module):
    def __init__(self, curve):
        super().__init__()
        self.correction = torch.compile(
            RationalCorrection(curve),
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True, "triton.cudagraphs": False},
        )

    def forward(self, u, v):
        return (F.silu(u) + self.correction(u)) * v


def install_compiled_corrections(model):
    if not all(
        isinstance(b.ffn, LearnableBlockShuffleFFN)
        and isinstance(b.ffn.curve, RationalResidualActivation)
        and not hasattr(b.ffn, "_compiled_product")
        for b in model.blocks
    ):
        raise ValueError("Expected unwrapped rational BlockShuffle FFNs")
    identities = {n: id(p) for n, p in model.named_parameters()}
    for block in model.blocks:
        ffn = block.ffn
        object.__setattr__(ffn, "_native_forward", ffn.forward)
        object.__setattr__(ffn, "_compiled_product", NativeSiluRationalProduct(ffn.curve))
        ffn.forward = MethodType(_compiled_forward, ffn)
    assert {n: id(p) for n, p in model.named_parameters()} == identities
    return model
