"""Independent checks of finite-epsilon VJPs and function-preserving FFN scaling."""

import copy

import torch

from src.core.residual_geometry import rms_vjp
from src.core.transformer import RMSNorm
from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN


def test_rms_vjp_and_unweighted_singular_values():
    generator = torch.Generator().manual_seed(91)
    for raw in (
        torch.zeros(5, dtype=torch.float64),
        torch.randn(5, dtype=torch.float64, generator=generator),
    ):
        x = raw.requires_grad_(True)
        gamma = torch.randn(5, dtype=torch.float64, generator=generator)
        adjoint = torch.randn(5, dtype=torch.float64, generator=generator)
        actual = torch.autograd.grad((gamma * x / (x.square().mean() + 1e-5).sqrt()), x, adjoint)[0]
        torch.testing.assert_close(rms_vjp(x, gamma, adjoint), actual, rtol=1e-12, atol=1e-12)
        jacobian = torch.autograd.functional.jacobian(
            lambda z: z / (z.square().mean() + 1e-5).sqrt(), x
        )
        s = (x.square().mean() + 1e-5).sqrt()
        expected = (
            torch.cat(((1 / s).repeat(4), (1e-5 / s.pow(3)).reshape(1)))
            .sort(descending=True)
            .values
        )
        torch.testing.assert_close(torch.linalg.svdvals(jacobian), expected, rtol=1e-10, atol=1e-12)


def test_real_rmsnorm_fp32_agrees_with_double_reference():
    generator = torch.Generator().manual_seed(39)
    norm = RMSNorm(12)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(12, generator=generator))
    for scale in (0.001, 1.0, 100.0):
        x = (scale * torch.randn(3, 7, 12, generator=generator)).requires_grad_(True)
        adjoint = torch.randn(x.shape, generator=generator)
        actual = torch.autograd.grad(norm(x), x, adjoint)[0]
        expected = rms_vjp(x.double(), norm.weight.double(), adjoint.double())
        error = (actual.double() - expected).norm() / expected.norm()
        assert error < 5e-6


def test_value_down_gauge_preserves_function_and_transforms_gradients():
    torch.set_num_threads(4)
    model = OvercompleteHeadwiseFFN(24, 32, 4, 2, 16).double()
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.value.mul_(8)
        changed.down.div_(8)
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 8, 24, generator=generator, dtype=torch.float64).requires_grad_(True)
    target = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    y, z = model(x), changed(x)
    torch.testing.assert_close(y, z, rtol=1e-13, atol=1e-13)
    a = torch.autograd.grad((y * target).sum(), (x, *model.parameters()))
    b = torch.autograd.grad((z * target).sum(), (x, *changed.parameters()))
    torch.testing.assert_close(a[0], b[0], rtol=1e-12, atol=1e-12)
    for (name, _), before, after in zip(model.named_parameters(), a[1:], b[1:], strict=True):
        expected = before / 8 if name == "value" else before * 8 if name == "down" else before
        torch.testing.assert_close(after, expected, rtol=1e-12, atol=1e-12)
