"""Independent numerical and integration contracts for router-free headwise FFNs."""

import pytest
import torch
from torch.nn import functional as F

from src.core.benchmark import forward_flops
from src.core.config import ModelConfig
from src.core.diagnostics import inspect_layers
from src.core.transformer import Transformer
from src.multihead_ffn.headwise import CalibratedHeadwiseFFN


def test_headwise_matches_independent_loop_forward_and_all_derivatives():
    module = CalibratedHeadwiseFFN(6, 2, 5).double()
    module.initialize(19, "test", 0.3)
    x = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    q = F.linear(x, module.input_projection.weight).reshape(2, 3, 2, 3)
    pieces = []
    for h in range(2):
        z = q[:, :, h]
        pieces.append((F.silu(z @ module.gate[h]) * (z @ module.value[h])) @ module.down[h])
    reference = F.linear(torch.cat(pieces, -1), module.output_projection.weight)
    actual = module(x)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    variables = (x, *module.parameters())
    expected_grads = torch.autograd.grad((reference * probe).sum(), variables)
    actual_grads = torch.autograd.grad((actual * probe).sum(), variables)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=1e-11, atol=1e-12)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


def test_headwise_orthogonal_norms_count_and_invalid_dimensions():
    m = CalibratedHeadwiseFFN(24, 3, 24).double()
    m.initialize(17, "blocks.0.ffn", 0.25)
    x = torch.randn(11, 24, dtype=torch.float64)
    for projection, scale in ((m.input_projection, 1.0), (m.output_projection, 0.25)):
        torch.testing.assert_close(
            projection(x).square().sum(-1), scale**2 * x.square().sum(-1), rtol=1e-12, atol=1e-12
        )
    assert sum(p.numel() for p in m.parameters()) == m.parameter_count
    assert not hasattr(m, "router")
    for dims in ((24, 5, 3), (24, 0, 3), (24, 3, 0)):
        with pytest.raises(ValueError):
            CalibratedHeadwiseFFN(*dims)
    with pytest.raises(ValueError):
        m.initialize(17, "test", float("nan"))


def test_headwise_registered_common_parameters_gradients_and_diagnostics():
    c = ModelConfig(
        variant="headwise_swiglu",
        width=24,
        layers=2,
        heads=3,
        context=8,
        vocab_size=32,
        groups=3,
        hidden=24,
    )
    m, again = Transformer(c, 17), Transformer(c, 17)
    full = Transformer(
        ModelConfig(variant="swiglu", width=24, layers=2, heads=3, context=8, vocab_size=32), 17
    )
    assert all(torch.equal(t, again.state_dict()[n]) for n, t in m.state_dict().items())
    assert all(
        torch.equal(t, full.state_dict()[n]) for n, t in m.state_dict().items() if ".ffn." not in n
    )
    x = torch.arange(16).reshape(2, 8)
    before = m(x).detach()
    d = inspect_layers(m, x, "cpu", "fp32")
    assert all(r["finite"] for r in d.values())
    assert torch.equal(before, m(x))
    m.loss(x, x + 1).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in m.parameters()
    )
    c = ModelConfig(variant="headwise_swiglu", width=384, layers=8, heads=6, groups=24)
    c.validate()
    assert (
        c.ffn_width == 48 and c.unique_ffn_parameters == 2801664 and c.total_parameters == 9099648
    )
    assert forward_flops(c)["ffn_matrix_forward_flops_per_token"] == 5603328
    with pytest.raises(ValueError, match="heads must divide"):
        ModelConfig(variant="headwise_swiglu", width=24, heads=3, groups=5).validate()


def test_headwise_screen_keeps_the_full_control_training_recipe():
    from src.core.multihead_screen import RATES, configuration
    from src.core.wikitext_screen import configuration as control_configuration

    for rate in RATES:
        candidate, training = configuration("headwise", rate)
        full, control_training = control_configuration("full_swiglu", rate)
        assert training == control_training
        for field in ("width", "layers", "heads", "context", "vocab_size"):
            assert getattr(candidate, field) == getattr(full, field)
        assert candidate.unique_ffn_parameters == 2801664
