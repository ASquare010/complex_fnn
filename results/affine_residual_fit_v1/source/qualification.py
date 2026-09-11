"""Small mathematical checks before H107 accesses new experimental windows."""

import torch

from results.affine_residual_fit_v1.source.model import (
    SPECS,
    Student,
    fold_raw,
    initialize_hidden,
    normalization,
    normalized_teacher,
    ridge_readout,
)
from src.dense_ffn import DenseFFN


def run():
    torch.manual_seed(107)
    checks = []
    x = torch.randn(96, 8, dtype=torch.float64) * 1.7 + 0.3
    y = torch.randn(96, 8, dtype=torch.float64) * 2 - 0.5
    norm = normalization(x, y)
    xn, yn = (x - norm["mx"]) / norm["sx"], (y - norm["my"]) / norm["sy"]
    for form, hidden in SPECS.items():
        actual = Student(form)
        count = sum(p.numel() for p in actual.parameters())
        expected = hidden * (3 * 384 + 2) + 384 if "swiglu" in form else hidden * 769 + 384
        if form.startswith("affine_"):
            expected += 384 * 384
        expected += form == "prelu"
        assert count == expected, (form, count, expected)
        model = Student(form, 8, 12).double()
        point = (torch.randn(3, 8, dtype=torch.float64) + 0.123).requires_grad_()
        assert torch.autograd.gradcheck(model, (point,), fast_mode=True)
        names = list(dict(model.named_parameters()))
        parameters = tuple(model.parameters())
        assert torch.autograd.gradcheck(
            lambda *values: torch.func.functional_call(model, dict(zip(names, values)), point),
            parameters,
            fast_mode=True,
        )
        raw = fold_raw(model, norm)
        direct = model((point - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
        folded = raw(point)
        torch.testing.assert_close(direct, folded, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad(direct.sum(), point)[0]
        gb = torch.autograd.grad(folded.sum(), point)[0]
        torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-12)
        model, ridge = ridge_readout(model, xn, yn)
        assert ridge["stationarity_max"] < 1e-10
        # Independent augmented least-squares solution of the same ridge problem.
        model.double()
        with torch.no_grad():
            h = model.features(xn)[1]
            design = h if model.affine is None else torch.cat((xn, h), 1)
            mean, sd = design.mean(0), design.std(0, correction=0).clamp_min(1e-6)
            z = (design - mean) / sd
            augmented = torch.cat((z, (len(z) * 1e-4) ** 0.5 * torch.eye(z.shape[1])))
            target = torch.cat((yn - yn.mean(0), torch.zeros(z.shape[1], 8)))
            solution = torch.linalg.lstsq(augmented, target, driver="gelsd").solution
            torch.testing.assert_close(model(xn), z @ solution + yn.mean(0), atol=2e-5, rtol=2e-5)
        checks.append(
            {"form": form, "parameters": count, "gradients": True, "fold": True, "ridge": ridge}
        )
    for kind in ("gelu", "swiglu"):
        teacher = DenseFFN(8, 20, kind).double()
        full = normalized_teacher(teacher, kind, norm).double()
        torch.testing.assert_close(
            full(xn), (teacher(x) - norm["my"]) / norm["sy"], atol=1e-7, rtol=1e-6
        )
        a, b = Student("affine_swiglu", 8, 9).double(), Student("swiglu", 8, 15).double()
        ra, rb = initialize_hidden(a, teacher, norm, 71), initialize_hidden(b, teacher, norm, 71)
        torch.testing.assert_close(ra, rb[:9])
        torch.testing.assert_close(
            a.features(xn)[1], b.features(xn)[1][:, :9], atol=1e-12, rtol=1e-12
        )
        checks.append({"teacher": kind, "normalized_exact": True, "prefix": True})
    return checks


if __name__ == "__main__":
    torch.set_num_threads(4)
    print(run())
