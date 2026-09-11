"""Independent counts, factorization derivatives and privileged capacity witnesses."""

import math

import torch
from torch.func import functional_call
from torch.nn import functional as F

from results.nonlinear_residual_v1.source.model import target_features
from results.spectral_fitting_v1.source.model import FORMS, FeatureFFN


def run_checks() -> list[dict]:
    torch.manual_seed(104)
    full_basis = torch.linalg.qr(torch.randn(384, 384, dtype=torch.float64)).Q
    bases = {m: full_basis[:, :32].float() for m in ("random", "raw", "bounded")}
    probe = torch.randn(512, 384)
    records = []
    for form in FORMS:
        model = FeatureFFN(form, 17, bases)
        h = model.hidden
        expected = (
            65536 + 128 + 384
            if form.startswith("factor")
            else (3 * 384 * h + 2 * h + 384 if "swiglu" in form else 2 * 384 * h + h + 384)
        )
        assert model.counts()["parameters"] == expected
        assert torch.count_nonzero(model(probe)) == 0
        variance = model.features(probe)[0].var(0, correction=0).mean().item()
        assert 0.7 <= variance <= 1.3
        model.cuda()
        with torch.no_grad():
            model.down.weight.normal_(0, 0.02)
        value = model(probe[:3].cuda())
        value.square().mean().backward()
        assert torch.isfinite(value).all() and all(
            torch.isfinite(p.grad).all() for p in model.parameters()
        )
        records.append(
            {"name": form, "parameters": expected, "input_variance": variance, "cuda_finite": True}
        )
        del model, value
    for activation in ("gelu", "cubic"):
        model = FeatureFFN(f"factor_{activation}_bounded", 17, bases).double()
        with torch.no_grad():
            model.down.weight.normal_(0, 0.02)
        x = torch.randn(2, 384, dtype=torch.float64, requires_grad=True)
        up = F.linear(x, model.up.weight @ model.compress.weight, model.up.bias)
        features = F.gelu(up) if activation == "gelu" else up * (up.square() - 3) / math.sqrt(6)
        expected = F.linear(features, model.down.weight, model.down.bias)
        actual = model(x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        inputs = (x, *model.parameters())
        direction = torch.randn_like(actual)
        for a, b in zip(
            torch.autograd.grad((actual * direction).sum(), inputs),
            torch.autograd.grad((expected * direction).sum(), inputs),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        names, parameters = zip(*model.named_parameters(), strict=True)

        def evaluate(*values):
            return functional_call(model, dict(zip(names, values[1:], strict=True)), (values[0],))

        assert torch.autograd.gradcheck(
            evaluate, (x, *parameters), fast_mode=True, atol=1e-5, rtol=1e-3
        )
        records.append({"name": f"{activation}_folded_gradient_and_gradcheck", "passed": True})
    for task in ("quadratic", "cubic", "product"):
        model = FeatureFFN("factor_cubic_bounded", 17, bases).double()
        rotation = torch.linalg.qr(torch.randn(384, 384, dtype=torch.float64)).Q[:16]
        with torch.no_grad():
            model.compress.weight.copy_(full_basis[:, :32].T)
            model.up.weight.zero_()
            model.up.bias.zero_()
            model.down.weight.zero_()
            if task == "cubic":
                for i in range(16):
                    model.up.weight[i, i] = 1
                    model.down.weight[:, i] = rotation[i]
            elif task == "quadratic":
                for i in range(16):
                    for j, sign in enumerate((1, -1)):
                        model.up.weight[2 * i + j, i] = 1
                        model.up.bias[2 * i + j] = sign
                        model.down.weight[:, 2 * i + j] = sign * math.sqrt(3) / 6 * rotation[i]
                model.down.bias.copy_(-rotation.sum(0) / (3 * math.sqrt(2)))
            else:
                for i in range(16):
                    for j, (sign, bias, coefficient) in enumerate(
                        ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
                    ):
                        row = 4 * i + j
                        model.up.weight[row, i] = 1
                        model.up.weight[row, (i + 1) % 16] = sign
                        model.up.bias[row] = bias
                        model.down.weight[:, row] = coefficient * math.sqrt(6) / 24 * rotation[i]
        x = torch.randn(7, 384, dtype=torch.float64, requires_grad=True)
        expected = target_features(x @ full_basis[:, :16], task) @ rotation
        actual = model(x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        a = torch.autograd.grad(actual.square().sum(), x)[0]
        b = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        records.append({"name": f"oracle_{task}", "passed": True})
    return records
