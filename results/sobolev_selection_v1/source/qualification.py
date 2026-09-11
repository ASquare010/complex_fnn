"""Mathematical witnesses for JVPs, exports and every greedy marginal gain."""

import itertools
import math

import torch

from results.sobolev_selection_v1.source.features import PrunedFFN, export, hidden_jvp, value_jvp
from results.sobolev_selection_v1.source.selection import RIDGE, WIDTHS, greedy
from src.dense_ffn import DenseFFN


def run():
    torch.manual_seed(108)
    checks = []
    for kind in ("gelu", "swiglu"):
        model = DenseFFN(6, 11, kind).double()
        x = torch.randn(13, 6, dtype=torch.float64, requires_grad=True)
        v = torch.randn_like(x)
        output, derivative = value_jvp(model, x, v)
        expected, jvp = torch.func.jvp(model, (x,), (v,))
        torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(derivative, jvp, atol=1e-12, rtol=1e-12)
        finite = (model(x + 1e-5 * v) - model(x - 1e-5 * v)) / 2e-5
        torch.testing.assert_close(derivative, finite, atol=1e-9, rtol=1e-8)
        state = {
            "scale_h": torch.rand(11, dtype=torch.float64) + 0.3,
            "mean_h": torch.randn(11, dtype=torch.float64),
            "scale_y": 1.7,
            "mean_y": torch.randn(6, dtype=torch.float64),
        }
        ids = torch.tensor([1, 4, 6, 9])
        b = torch.randn(4, 6, dtype=torch.float64)
        deployed = export(model, state, ids, b)
        h, k = hidden_jvp(model, x, v)
        analytic = ((h[:, ids] - state["mean_h"][ids]) / state["scale_h"][ids]) @ b * state[
            "scale_y"
        ] + state["mean_y"]
        d_analytic = (k[:, ids] / state["scale_h"][ids]) @ b * state["scale_y"]
        actual, d_actual = value_jvp(deployed, x, v)
        torch.testing.assert_close(actual, analytic, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(d_actual, d_analytic, atol=1e-12, rtol=1e-12)
        assert torch.autograd.gradcheck(deployed, (x,), fast_mode=True)
        checks.append(
            {
                "activation": kind,
                "autograd_jvp": True,
                "finite_difference_jvp": True,
                "export_and_input_gradient": True,
            }
        )
        for name, width in WIDTHS[kind].items():
            deployed = PrunedFFN(384, width, kind)
            count = sum(p.numel() for p in deployed.parameters())
            assert count == (3 if kind == "swiglu" else 2) * 384 * width + 384
            checks.append({"activation": kind, "budget": name, "parameters": count})
    for deficient in (False, True):
        for beta in (0.0, 0.13):
            n, p, targets = 47, 11, 3
            h, k = torch.randn(n, p, dtype=torch.float64), torch.randn(n, p, dtype=torch.float64)
            y, t = (
                torch.randn(n, targets, dtype=torch.float64),
                torch.randn(n, targets, dtype=torch.float64),
            )
            h, y = h - h.mean(0), y - y.mean(0)
            if deficient:
                h[:, 1], k[:, 1] = h[:, 0], k[:, 0]
            g = (h.T @ h + beta * k.T @ k) / n + RIDGE * torch.eye(p, dtype=torch.float64)
            c = (h.T @ y + beta * k.T @ t) / n
            order, trace = greedy(g, c, 7)
            a = torch.cat(
                (
                    h / math.sqrt(n),
                    math.sqrt(beta / n) * k,
                    math.sqrt(RIDGE) * torch.eye(p, dtype=torch.float64),
                )
            )
            target = torch.cat(
                (
                    y / math.sqrt(n),
                    math.sqrt(beta / n) * t,
                    torch.zeros(p, targets, dtype=torch.float64),
                )
            )
            previous = target.square().sum().item()
            for index, selected in enumerate(order.tolist()):
                prefix = order[:index].tolist()
                objectives = {}
                for alternative in range(p):
                    if alternative in prefix:
                        continue
                    design = a[:, prefix + [alternative]]
                    fitted = torch.linalg.lstsq(design, target, driver="gelsd").solution
                    objectives[alternative] = (design @ fitted - target).square().sum().item()
                assert objectives[selected] <= min(objectives.values()) + 1e-11
                improvement = previous - objectives[selected]
                assert abs(improvement - trace["gains"][index]) < 1e-11
                previous = objectives[selected]
            checks.append(
                {
                    "rank_deficient": deficient,
                    "beta": beta,
                    "seven_greedy_gains_against_all_candidate_solves": True,
                }
            )
    signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=4)), dtype=torch.float64)
    jacobian = torch.randn(7, 4, dtype=torch.float64)
    sx = torch.rand(4, dtype=torch.float64)
    empirical = ((signs * sx) @ jacobian.T).square().sum(1).mean()
    exact = (jacobian * sx).square().sum()
    torch.testing.assert_close(empirical, exact, atol=1e-12, rtol=1e-12)
    checks.append({"rademacher_isotropy_exact_enumeration": True})
    return checks


if __name__ == "__main__":
    torch.set_num_threads(4)
    print(run())
