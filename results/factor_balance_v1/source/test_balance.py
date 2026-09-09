"""Four independent mathematical/software checks for H069."""

import torch

from results.factor_balance_v1.source.diagnosis import balance, initialized_ffn, scales
from src.blockshuffle_ffn import BlockShuffleLinear
from src.core.config import ModelConfig
from src.core.transformer import Transformer

torch.set_num_threads(4)


def test_scalar_optimum_and_discrete_rounding():
    g = torch.Generator().manual_seed(691)
    a = 2 ** (12 * torch.rand(1000, generator=g, dtype=torch.float64) - 6)
    b = 2 ** (12 * torch.rand(1000, generator=g, dtype=torch.float64) - 6)
    for s1, s2 in ((1.0, 1.0), (4.0, 4.0), (4.0, 64 / 3)):
        c, e, before, opt, after, ratio = scales(a, b, s1, s2)
        exact_c = torch.sqrt(b / a) * (s1 / s2) ** 0.25
        torch.testing.assert_close(
            a.square() * exact_c.square() / s1 + b.square() / exact_c.square() / s2,
            opt,
            rtol=2e-14,
            atol=0,
        )
        for neighbor in (c / 2, c * 2):
            alternative = a.square() * neighbor.square() / s1 + b.square() / neighbor.square() / s2
            assert (after <= alternative * (1 + 1e-12)).all()
        assert (after <= 1.25 * opt * (1 + 1e-12)).all() and (after <= before * (1 + 1e-12)).all()
        assert (ratio >= 0.5 * (1 - 1e-12)).all() and (ratio <= 2 * (1 + 1e-12)).all()
        assert torch.equal(c, 2**e)


def test_actual_shuffle_preserves_dense_matrix():
    for n, m in ((24, 48), (48, 24)):
        p = BlockShuffleLinear(n, m, 3).double()
        p.initialize(17, "probe")
        with torch.no_grad():
            p.first.weight.mul_(torch.linspace(0.25, 4, 24, dtype=torch.float64).reshape(3, 8, 1))
        before = p(torch.eye(n, dtype=torch.float64)).T.detach()
        record, _ = balance(p)
        after = p(torch.eye(n, dtype=torch.float64)).T.detach()
        assert record["all_bounds_pass"]
        torch.testing.assert_close(after, before, rtol=1e-13, atol=1e-14)
        torch.testing.assert_close(
            torch.linalg.svdvals(after), torch.linalg.svdvals(before), rtol=1e-12, atol=1e-14
        )


def test_gradient_covariance_against_direct_linear_products():
    a = torch.randn(
        6, 4, generator=torch.Generator().manual_seed(11), dtype=torch.float64, requires_grad=True
    )
    b = torch.randn(
        3, 6, generator=torch.Generator().manual_seed(12), dtype=torch.float64, requires_grad=True
    )
    x = torch.randn(
        8, 4, generator=torch.Generator().manual_seed(13), dtype=torch.float64, requires_grad=True
    )
    c = 2.0 ** torch.tensor([-3, -2, -1, 0, 1, 2], dtype=torch.float64)
    ap = (a.detach() * c[:, None]).requires_grad_()
    bp = (b.detach() / c).requires_grad_()
    xp = x.detach().clone().requires_grad_()
    output = (x @ a.T) @ b.T
    alternate = (xp @ ap.T) @ bp.T
    ga, gb, gx = torch.autograd.grad(output.square().sum(), (a, b, x))
    gap, gbp, gxp = torch.autograd.grad(alternate.square().sum(), (ap, bp, xp))
    for first, second in ((output, alternate), (ga, gap * c[:, None]), (gb, gbp / c), (gx, gxp)):
        torch.testing.assert_close(first, second, rtol=1e-13, atol=1e-12)


def test_initialization_matches_transformer_and_counts():
    config = ModelConfig(
        variant="blockshuffle_swiglu", width=384, layers=8, heads=6, hidden=2048, groups=8
    )
    transformer = Transformer(config, 17)
    assert sum(p.numel() for b in transformer.blocks for p in b.ffn.parameters()) == 2801664
    for layer in (0, 3, 7):
        standalone = initialized_ffn(17, layer)
        assert sum(p.numel() for p in standalone.parameters()) == 350208
        assert all(
            torch.equal(v, transformer.blocks[layer].ffn.state_dict()[n])
            for n, v in standalone.state_dict().items()
        )
