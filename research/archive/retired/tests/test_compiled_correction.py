"""Numerical isolation of rational correction compilation."""

import pytest
import torch

from src.learnable_activation_ffn import RationalResidualActivation
from src.learnable_activation_ffn.compiled_correction import NativeSiluRationalProduct


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_correction_compile_keeps_native_zero_input_gradients_and_nonzero_fidelity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    torch.manual_seed(738)
    curve = RationalResidualActivation(2048, 8).cuda()
    product = NativeSiluRationalProduct(curve)
    u = torch.randn(2, 7, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(u, requires_grad=True)
    for randomized in (False, True):
        if randomized:
            with torch.no_grad():
                for p in curve.parameters():
                    p.uniform_(-0.3, 0.3)
        native = curve(u) * v
        expected = torch.autograd.grad(native.float().square().mean(), (u, v, *curve.parameters()))
        actual = product(u, v)
        gradients = torch.autograd.grad(actual.float().square().mean(), (u, v, *curve.parameters()))
        if not randomized:
            assert torch.equal(native, actual)
            assert all(torch.equal(a, b) for a, b in zip(expected[:2], gradients[:2]))
        for a, b in zip(expected, gradients):
            assert torch.isfinite(b).all()
            error = (a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)
            assert error < 0.01
        assert (actual.float() - native.float()).norm() / native.float().norm() < 0.003


def test_correction_wrapper_preserves_registered_state_and_rejects_mixed_wrapping():
    from src.core.config import ModelConfig
    from src.core.transformer import Transformer
    from src.learnable_activation_ffn.compiled_correction import install_compiled_corrections
    from src.learnable_activation_ffn.compiled_product import install_compiled_products

    model = Transformer(
        ModelConfig(
            variant="blockshuffle_swiglu_rational",
            width=24,
            hidden=48,
            layers=1,
            heads=3,
            vocab_size=32,
            context=8,
        )
    )
    identities = {n: id(p) for n, p in model.named_parameters()}
    state = list(model.state_dict())
    install_compiled_corrections(model)
    assert {n: id(p) for n, p in model.named_parameters()} == identities
    assert list(model.state_dict()) == state
    for installer in (install_compiled_corrections, install_compiled_products):
        with pytest.raises(ValueError, match="unwrapped"):
            installer(model)
