"""H072 fixed full-size ungated-control and parity qualification; zero updates."""

import copy
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from results.factor_balance_v2.source.logging_fix import tensor_hash
from results.ungated_blockshuffle_v1.source.candidate import make_model
from src.blockshuffle_ffn import BlockShuffleFFN
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import group_summary, parameter_groups
from src.core.reproducibility import sha256, write_json
from src.dense_ffn import DenseFFN

ROOT = Path("results/ungated_blockshuffle_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def save(name, record, raw=None):
    folder = ROOT / "checks"
    folder.mkdir(exist_ok=True)
    assert not (folder / (name + ".json")).exists()
    if raw is not None:
        path = folder / (name + ".pt")
        assert not path.exists()
        torch.save(raw, path)
        record["artifact_sha256"] = sha256(path)
    write_json(folder / (name + ".json"), record)


def values(model, x, incoming, device, recompute):
    model = model.to(device).train()
    x = x.to(device).requires_grad_(True)
    with torch.autocast(device, enabled=device == "cuda", dtype=torch.bfloat16):
        y = (
            checkpoint(model, x, use_reentrant=False, preserve_rng_state=True)
            if recompute
            else model(x)
        )
    gradients = torch.autograd.grad(
        y, (x, *model.parameters()), incoming.to(device=device, dtype=y.dtype)
    )
    return {
        "output": y.detach().cpu(),
        "input_gradient": gradients[0].cpu(),
        **{n: g.cpu() for (n, _), g in zip(model.named_parameters(), gradients[1:])},
    }


@pytest.mark.parametrize("hidden", (2048, 3264))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("seed", (17, 29, 43))
def test_checkpoint(hidden, device, seed):
    model = make_model(hidden, seed)
    weights = {n: tensor_hash(p) for n, p in model.state_dict().items()}
    g = torch.Generator().manual_seed(95000 + seed)
    shape = (16, 128, 384) if device == "cuda" else (2, 7, 384)
    x = torch.randn(shape, generator=g)
    incoming = torch.randn(shape, generator=g)
    first = values(copy.deepcopy(model), x.clone(), incoming, device, False)
    second = values(copy.deepcopy(model), x.clone(), incoming, device, True)
    compared = {
        n: {
            "exact": torch.equal(t, second[n]),
            "finite": bool(torch.isfinite(t).all() and torch.isfinite(second[n]).all()),
            "norm": t.float().norm().item(),
            "first_sha256": tensor_hash(t),
            "second_sha256": tensor_hash(second[n]),
        }
        for n, t in first.items()
    }
    passed = all(
        v["exact"] and v["finite"] and v["norm"] > 0 and v["first_sha256"] == v["second_sha256"]
        for v in compared.values()
    )
    save(
        f"checkpoint_h{hidden}_{device}_s{seed}",
        {
            "passed": passed,
            "comparisons": compared,
            "initial_weights": weights,
            "optimizer_updates": 0,
        },
        {"input": x, "incoming": incoming, "eager": first, "checkpointed": second},
    )
    assert passed


def test_counts_topology():
    models = {
        "gelu_same_width": make_model(2048, 17),
        "gelu_matched": make_model(3264, 17),
        "plain_swiglu": BlockShuffleFFN(384, 2048, 8),
        "full_gelu": DenseFFN(384, 1536, "gelu"),
        "full_swiglu": DenseFFN(384, 1024, "swiglu"),
        "narrow_gelu": DenseFFN(384, 456, "gelu"),
    }
    expected = dict(zip(models, (233472, 350208, 350208, 1179648, 1179648, 350208)))
    counts = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}
    assert counts == expected
    routes = torch.zeros(8, 8, dtype=torch.int64)
    for destination in range(384):
        origin = (destination % 8) * 48 + destination // 8
        routes[destination // 48, origin // 48] += 1
    assert torch.equal(routes, torch.full((8, 8), 6))
    save(
        "counts_topology",
        {
            "passed": True,
            "counts": counts,
            "routing_counts": routes.tolist(),
            "gelu_matched_ffn_total": counts["gelu_matched"] * 8,
            "gelu_matched_model_total": counts["gelu_matched"] * 8 + 6297984,
            "gelu_same_width_ffn_total": counts["gelu_same_width"] * 8,
            "gelu_same_width_model_total": counts["gelu_same_width"] * 8 + 6297984,
        },
    )


def test_calibration():
    rows, raw = [], {}
    for hidden in (2048, 3264):
        model = make_model(hidden, 17)
        wrapper = nn.Module()
        wrapper.add_module("ffn", model)
        wrapper.blocks = [SimpleNamespace(ffn=model)]
        wrapper.config = ModelConfig(variant="gelu", width=384, hidden=hidden)
        groups = parameter_groups(
            wrapper, TrainConfig(learning_rate=0.001, weight_decay=0, ffn_lr_mode="fan_in")
        )
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == 4 and set(ids) == {id(p) for p in model.parameters()}
        for name, parameter in model.named_parameters():
            scale = hidden / 96 if name == "down.second.weight" else 4
            group = next(g for g in groups if any(p is parameter for p in g["params"]))
            assert (
                group["lr_scale"] == scale
                and group["lr"] == 0.001 * scale
                and group["weight_decay"] == 0
            )
        model = model.double()
        for name in ("up", "down"):
            projection = getattr(model, name)
            first = torch.block_diag(*projection.first.weight.unbind())
            second = torch.block_diag(*projection.second.weight.unbind())
            perm = torch.arange(384).reshape(8, 48).T.flatten()
            inverse = (
                torch.arange(projection.output_width)
                .reshape(projection.output_width // 8, 8)
                .T.flatten()
            )
            matrix = (second @ first[perm])[inverse]
            norm = matrix.square().sum(-1).mean().item()
            expected = projection.input_width * 0.02**2 * (0.25**2 if name == "down" else 1)
            assert math.isclose(norm, expected, rel_tol=1e-5, abs_tol=1e-8)
            rows.append(
                {
                    "hidden": hidden,
                    "projection": name,
                    "mean_row_norm_squared": norm,
                    "expected": expected,
                    "optimizer_groups": group_summary(groups),
                }
            )
            raw[f"h{hidden}_{name}"] = {
                "matrix": matrix.detach(),
                "parameters": {n: t.detach() for n, t in projection.state_dict().items()},
            }
    save("calibration", {"passed": True, "rows": rows}, raw)


@pytest.mark.parametrize("hidden", (2048, 3264))
def test_gelu_parity(hidden):
    model = make_model(hidden, 17).double().eval()
    g = torch.Generator().manual_seed(96017)
    x = torch.randn(14, 384, generator=g, dtype=torch.float64)
    tangent = torch.randn(1, 384, generator=g, dtype=torch.float64)
    with torch.no_grad():
        difference = model(x) - model(-x)
        linear = model.down(model.up(x))
        expected_jvp = 0.5 * model.down(model.up(tangent))
    _, jvp = torch.autograd.functional.jvp(model, torch.zeros_like(tangent), tangent)
    torch.testing.assert_close(difference, linear, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(jvp, expected_jvp, rtol=1e-11, atol=1e-12)
    save(
        f"gelu_parity_h{hidden}",
        {
            "passed": True,
            "odd_identity_max_error": (difference - linear).abs().max().item(),
            "origin_jvp_max_error": (jvp - expected_jvp).abs().max().item(),
            "whole_network_gradient_bound_claimed": False,
        },
        {
            "input": x,
            "tangent": tangent,
            "difference": difference,
            "linear": linear,
            "jvp": jvp,
            "expected_jvp": expected_jvp,
        },
    )


def test_swiglu_parity_and_separation():
    model = BlockShuffleFFN(384, 2048, 8).double()
    for name in ("up", "gate", "down"):
        getattr(model, name).initialize(17, "blocks.0.ffn." + name, 0.25 if name == "down" else 1)
    g = torch.Generator().manual_seed(96017)
    x = torch.randn(14, 384, generator=g, dtype=torch.float64)
    tangent = torch.randn(1, 384, generator=g, dtype=torch.float64)
    with torch.no_grad():
        even_sum = model(x) + model(-x)
        quadratic = model.down(model.up(x) * model.gate(x))
    _, jvp = torch.autograd.functional.jvp(model, torch.zeros_like(tangent), tangent)
    torch.testing.assert_close(even_sum, quadratic, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(jvp) == 0
    gelu = make_model(3264, 17).double()
    with torch.no_grad():
        for net in (gelu, model):
            for parameter in net.parameters():
                parameter.zero_()
            for name in ("up", "gate", "down"):
                projection = getattr(net, name)
                if projection is not None:
                    projection.first.weight[0, 0, 0] = 1
                    projection.second.weight[0, 0, 0] = 1
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        inputs = torch.zeros(2, 384, dtype=torch.float64)
        inputs[:, 0] = t
        gp, gn, sp, sn = gelu(inputs), gelu(-inputs), model(inputs), model(-inputs)
        assert all(torch.count_nonzero(y[:, 1:]) == 0 for y in (gp, gn, sp, sn))
        torch.testing.assert_close(gp[:, 0], torch.nn.functional.gelu(t), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            sp[:, 0], t * torch.nn.functional.silu(t), rtol=1e-12, atol=1e-12
        )
        ge = 0.5 * (gp[:, 0] + gn[:, 0]) / t.square()
        so = 0.5 * (sp[:, 0] - sn[:, 0]) / t
        gelu_gap = abs((ge[1] - ge[0]).item())
        swiglu_gap = abs((so[1] - so[0]).item())
        assert gelu_gap > 0.01 and swiglu_gap > 0.01
    save(
        "swiglu_parity_separation",
        {
            "passed": True,
            "even_identity_max_error": (even_sum - quadratic).abs().max().item(),
            "origin_jvp_exact_zero": True,
            "gelu_even_quadratic_ratio_gap": gelu_gap,
            "swiglu_odd_linear_ratio_gap": swiglu_gap,
            "single_bias_free_layer_scope": True,
        },
        {
            "even_sum": even_sum,
            "quadratic": quadratic,
            "jvp": jvp,
            "witness_input": inputs,
            "gelu_positive": gp,
            "gelu_negative": gn,
            "swiglu_positive": sp,
            "swiglu_negative": sn,
            "gelu_state": gelu.state_dict(),
            "swiglu_state": model.state_dict(),
        },
    )


def test_population_bound():
    nodes = torch.tensor([-3 / math.sqrt(5), 0, 3 / math.sqrt(5)], dtype=torch.float64)
    weights = torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64)
    indices = torch.tensor(list(itertools.product(range(3), repeat=4)))
    x = nodes[indices]
    w = weights[indices].prod(-1)
    a, b, c, d = (torch.roll(x, -i, -1) for i in range(4))
    target = a * b + 2 * b * c * d + a.square() * d
    residual = 2 * b * c * d + (a.square() - 1) * d
    covariance = target.T @ (w[:, None] * target)
    residual_covariance = residual.T @ (w[:, None] * residual)
    linear_cross = x.T @ (w[:, None] * residual)
    torch.testing.assert_close(
        covariance, torch.eye(4, dtype=torch.float64) * 34 / 5, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        residual_covariance, torch.eye(4, dtype=torch.float64) * 24 / 5, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        linear_cross, torch.zeros(4, 4, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    generator = torch.Generator().manual_seed(96018)
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64)).Q
    scales = torch.tensor([0.8, 1.2, 0.9, 1.6], dtype=torch.float64)
    y, z = target @ rotation / scales, residual @ rotation / scales
    ratio = ((w[:, None] * z.square()).sum() / (w[:, None] * y.square()).sum()).item()
    assert math.isclose(ratio, 12 / 17, rel_tol=1e-12, abs_tol=1e-12)
    save(
        "population_bound",
        {
            "passed": True,
            "nodes": 81,
            "relative_population_error_floor": ratio,
            "finite_sample_bound_claimed": False,
            "optimizer_updates": 0,
        },
        {
            "x": x,
            "weights": w,
            "target": target,
            "residual": residual,
            "covariance": covariance,
            "residual_covariance": residual_covariance,
            "linear_cross": linear_cross,
            "rotation": rotation,
            "scales": scales,
        },
    )
