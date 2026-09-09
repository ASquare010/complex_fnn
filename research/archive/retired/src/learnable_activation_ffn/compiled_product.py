"""Optional Inductor fusion of a rational activation and its multiplicative gate.

Original weights and registered names stay intact. The native training recipe
never imports or enables this backend implicitly.
"""

from types import MethodType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.learnable_activation_ffn import LearnableBlockShuffleFFN, RationalResidualActivation


class RationalProduct(nn.Module):
    def __init__(self, curve: RationalResidualActivation) -> None:
        super().__init__()
        self.curve = curve

    def forward(self, u, v):
        return self.curve(u) * v


def compile_product(curve: RationalResidualActivation):
    if not isinstance(curve, RationalResidualActivation):
        raise ValueError("This audited backend is specific to rational residual activations")
    return torch.compile(
        RationalProduct(curve),
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"emulate_precision_casts": True, "triton.cudagraphs": False},
    )


def _compiled_forward(self, x):
    u, v = self.up(x), self.gate(x)
    if self.recompute_gate and self.training and torch.is_grad_enabled():
        z = checkpoint(self._compiled_product, u, v, use_reentrant=False, preserve_rng_state=False)
    else:
        z = self._compiled_product(u, v)
    return self.down(z)


def install_compiled_products(model):
    """Install after moving the model to its device; build no packed weight cache."""
    if not all(
        isinstance(b.ffn, LearnableBlockShuffleFFN)
        and isinstance(b.ffn.curve, RationalResidualActivation)
        and not hasattr(b.ffn, "_compiled_product")
        for b in model.blocks
    ):
        raise ValueError("Expected unwrapped rational BlockShuffle FFNs")
    parameters = {n: id(p) for n, p in model.named_parameters()}
    for block in model.blocks:
        ffn = block.ffn
        # The compiled callable references the already registered curve. Bypass
        # module registration to avoid duplicate state-dict paths for its weights.
        object.__setattr__(ffn, "_native_forward", ffn.forward)
        object.__setattr__(ffn, "_compiled_product", compile_product(ffn.curve))
        ffn.forward = MethodType(_compiled_forward, ffn)
    assert {n: id(p) for n, p in model.named_parameters()} == parameters
    return model
