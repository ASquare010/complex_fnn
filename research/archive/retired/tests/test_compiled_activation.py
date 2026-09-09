"""CUDA correctness for the optional rational product compiler."""

import pytest
import torch

from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.learnable_activation_ffn import RationalResidualActivation
from src.learnable_activation_ffn.compiled_product import compile_product, install_compiled_products


def relative_error(actual, expected):
    delta = (actual.float() - expected.float()).flatten()
    return (delta.norm() / expected.float().flatten().norm().clamp_min(1e-12)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_rational_product_outputs_and_every_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    torch.manual_seed(37)
    curve = RationalResidualActivation(2048, 8).cuda()
    with torch.no_grad():
        for p in curve.parameters():
            p.uniform_(-0.3, 0.3)
    u = torch.randn(2, 7, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(u, requires_grad=True)
    eager = curve(u) * v
    expected = torch.autograd.grad(eager.float().square().mean(), (u, v, *curve.parameters()))
    compiled = compile_product(curve)(u, v)
    actual = torch.autograd.grad(compiled.float().square().mean(), (u, v, *curve.parameters()))
    assert relative_error(compiled, eager) < 0.003
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all() and relative_error(a, b) < 0.01


def test_install_preserves_parameters_and_rejects_double_wrapping():
    config = ModelConfig(
        variant="blockshuffle_swiglu_rational",
        width=24,
        hidden=48,
        layers=1,
        heads=3,
        vocab_size=32,
        context=8,
    )
    model = Transformer(config)
    params = {n: id(p) for n, p in model.named_parameters()}
    state = list(model.state_dict())
    install_compiled_products(model)
    assert {n: id(p) for n, p in model.named_parameters()} == params
    assert list(model.state_dict()) == state
    with pytest.raises(ValueError, match="unwrapped"):
        install_compiled_products(model)


def test_diagnostics_use_native_reference_and_restore_compiled_forward():
    from src.core.diagnostics import inspect_layers

    config = ModelConfig(
        variant="blockshuffle_swiglu_rational",
        width=24,
        hidden=48,
        layers=1,
        heads=3,
        vocab_size=32,
        context=8,
    )
    model = Transformer(config)
    x = torch.arange(16).reshape(2, 8)
    expected = inspect_layers(model, x, "cpu", "fp32")
    install_compiled_products(model)
    forward = model.blocks[0].ffn.forward
    # Diagnostics must not trace Python hooks through the compiled callable.
    actual = inspect_layers(model, x, "cpu", "fp32")
    assert actual == expected
    assert model.blocks[0].ffn.forward is forward


def test_backend_configuration_is_explicit_and_validated():
    from src.core.config import TrainConfig

    assert TrainConfig().activation_backend == "eager"
    TrainConfig(activation_backend="inductor_rational").validate()
    TrainConfig(activation_backend="inductor_rational_correction").validate()
    with pytest.raises(ValueError, match="Activation backend"):
        TrainConfig(activation_backend="unknown").validate()
