"""H070 fixed 21-check qualification; no optimization or corpus scoring."""

import copy
import math
from pathlib import Path

import pytest
import torch

from results.factor_balance_v2.source.logging_fix import tensor_hash
from results.rotated_shuffle_v1.source.candidate import (
    PairRotation,
    RotatedBlockShuffleFFN,
    RotatedBlockShuffleLinear,
    rotate_pairs,
)
from src.blockshuffle_ffn import BlockShuffleFFN, BlockShuffleLinear
from src.core.reproducibility import sha256, write_json
from src.core.structured_linear import shuffle_channels

ROOT = Path("results/rotated_shuffle_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def save(name, record, raw=None):
    folder = ROOT / "checks"
    folder.mkdir(exist_ok=True)
    path = folder / (name + ".json")
    assert not path.exists()
    if raw is not None:
        artifact = folder / (name + ".pt")
        assert not artifact.exists()
        torch.save(raw, artifact)
        record["artifact_sha256"] = sha256(artifact)
    write_json(path, record)


def pair(seed, device, shift):
    d, h = (384, 2048) if device == "cuda" else (64, 128)
    base = BlockShuffleFFN(d, h, 8)
    for name in ("up", "gate", "down"):
        getattr(base, name).initialize(seed, name, 0.25 if name == "down" else 1.0)
    model = RotatedBlockShuffleFFN(d, h, 8, shift)
    status = model.load_state_dict(base.state_dict(), strict=False)
    assert sorted(status.missing_keys) == [
        "down.rotation.theta",
        "gate.rotation.theta",
        "up.rotation.theta",
    ]
    assert not status.unexpected_keys
    for ffn in (base, model):
        ffn.recompute_gate = True
        ffn.gate_recompute_method = "checkpoint"
    shape = (16, 128, d) if device == "cuda" else (2, 7, d)
    g = torch.Generator().manual_seed(93000 + seed)
    x = torch.randn(shape, generator=g)
    incoming = torch.randn(shape, generator=g)
    return base, model, x, incoming


def values(model, x, incoming, device):
    model = model.to(device).train()
    x = x.to(device).requires_grad_(True)
    with torch.autocast(device, enabled=device == "cuda", dtype=torch.bfloat16):
        y = model(x)
    gradients = torch.autograd.grad(
        y, (x, *model.parameters()), incoming.to(device=device, dtype=y.dtype)
    )
    return {
        "output": y.detach().cpu(),
        "input_gradient": gradients[0].cpu(),
        **{n: g.cpu() for (n, _), g in zip(model.named_parameters(), gradients[1:])},
    }


def comparisons(first, second):
    records = {}
    for n, a in first.items():
        b = second[n]
        records[n] = {
            "exact": torch.equal(a, b),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs_error": (a - b).abs().max().item(),
            "first_sha256": tensor_hash(a),
            "second_sha256": tensor_hash(b),
        }
    return records


@pytest.mark.parametrize("seed", (17, 29, 43))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("shift", (0, 1))
def test_zero(seed, device, shift):
    base, model, x, incoming = pair(seed, device, shift)
    weights = {n: tensor_hash(t) for n, t in model.state_dict().items()}
    first = values(base, x.clone(), incoming, device)
    second = values(model, x.clone(), incoming, device)
    shared = comparisons(first, second)
    angles = {
        n: {"norm": g.norm().item(), "finite": bool(torch.isfinite(g).all())}
        for n, g in second.items()
        if n.endswith("rotation.theta")
    }
    passed = all(
        v["exact"] and v["finite"] and v["first_sha256"] == v["second_sha256"]
        for v in shared.values()
    ) and all(v["finite"] and v["norm"] > 0 for v in angles.values())
    save(
        f"zero_s{seed}_{device}_shift{shift}",
        {
            "passed": passed,
            "comparisons": shared,
            "angle_gradients": angles,
            "initial_weights": weights,
            "input_sha256": tensor_hash(x),
            "optimizer_updates": 0,
        },
        {"input": x, "incoming": incoming, "plain": first, "rotated": second},
    )
    assert passed


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("shift", (0, 1))
def test_nonzero(device, shift):
    _, model, x, incoming = pair(17, device, shift)
    with torch.no_grad():
        for name in ("up", "gate", "down"):
            theta = getattr(model, name).rotation.theta
            theta.copy_(torch.linspace(-0.3, 0.3, theta.numel()))
    eager = copy.deepcopy(model)
    eager.recompute_gate = False
    first = values(eager, x.clone(), incoming, device)
    second = values(model, x.clone(), incoming, device)
    compared = comparisons(first, second)
    passed = all(
        v["exact"] and v["finite"] and v["first_sha256"] == v["second_sha256"]
        for v in compared.values()
    )
    save(
        f"nonzero_{device}_shift{shift}",
        {"passed": passed, "comparisons": compared, "optimizer_updates": 0},
        {"input": x, "incoming": incoming, "eager": first, "checkpointed": second},
    )
    assert passed


def routing(k, groups):
    indices = shuffle_channels(torch.arange(k), groups)
    result = torch.zeros(groups, groups, dtype=torch.int64)
    for position, origin in enumerate(indices.tolist()):
        result[position // (k // groups), origin // (k // groups)] += 1
    return result


def test_counts_and_topology():
    counts = {}
    for shift in (0, 1):
        model = RotatedBlockShuffleFFN(384, 2048, 8, shift)
        total = sum(p.numel() for p in model.parameters())
        assert total == 350784 and total * 8 == 2806272
        assert 1 - total * 8 / 9437184 >= 0.7
        counts[str(shift)] = {
            "per_ffn": total,
            "ffn_total": total * 8,
            "total_model": total * 8 + 6297984,
            "reduction_percent": 100 * (1 - total * 8 / 9437184),
        }
    route = {str(k): routing(k, 8) for k in (48, 64, 384)}
    assert int((route["48"] == 0).sum()) == 16 and int((route["48"] == 1).sum()) == 48
    assert torch.equal(route["64"], torch.ones(8, 8, dtype=torch.int64))
    assert torch.equal(route["384"], torch.full((8, 8), 6, dtype=torch.int64))
    save(
        "counts_topology",
        {
            "passed": True,
            "counts": counts,
            "routing_counts": {k: v.tolist() for k, v in route.items()},
            "h068_small_topology_differs": True,
        },
    )


def test_full_size_rank_witness():
    model = RotatedBlockShuffleLinear(384, 2048, 8, 1).double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for j in range(6):
            model.first.weight[1, j, j] = 1
            model.second.weight[0, j, 1 + 8 * j] = 1
        model.first.weight[1, 24, 6] = 1
        model.second.weight[0, 6, 0] = 1
        model.rotation.theta[0] = math.pi / 4
        matrix = shuffle_channels(model(torch.eye(384, dtype=torch.float64)), 8).T
    expected = torch.zeros_like(matrix)
    for j in range(6):
        expected[j, 48 + j] = 1
    expected[6, 54] = -1 / math.sqrt(2)
    torch.testing.assert_close(matrix, expected, rtol=1e-12, atol=1e-12)
    block = matrix[:256, 48:96]
    singular = torch.linalg.svdvals(block)
    target = torch.zeros(48, dtype=torch.float64)
    target[:6] = 1
    target[6] = 1 / math.sqrt(2)
    torch.testing.assert_close(singular, target, rtol=1e-12, atol=1e-12)
    assert int(torch.linalg.matrix_rank(block, atol=1e-12)) == 7
    lower = singular[6:].square().sum().sqrt().item()
    assert math.isclose(lower, 1 / math.sqrt(2), rel_tol=1e-12)
    relative = lower / matrix.norm().item()
    assert math.isclose(relative, 1 / math.sqrt(13), rel_tol=1e-12)
    save(
        "rank_witness",
        {
            "passed": True,
            "rank": 7,
            "original_block_rank_bound": 6,
            "frobenius_error_lower_bound": lower,
            "relative_matrix_error_lower_bound": relative,
            "full_ffn_separation_claimed": False,
        },
        {
            "canonical_matrix": matrix,
            "expected": expected,
            "singular_values": singular,
            "parameters": model.state_dict(),
        },
    )


def test_absorbable_control():
    model = RotatedBlockShuffleLinear(64, 128, 8, 0).double()
    model.initialize(17, "absorb")
    with torch.no_grad():
        model.rotation.theta.copy_(torch.linspace(-0.3, 0.3, 32, dtype=torch.float64))
    identity = torch.eye(64, dtype=torch.float64)
    q = model.rotation(identity).T
    p = shuffle_channels(identity, 8).T
    transform = p.T @ q @ p
    off = transform.clone()
    for g in range(8):
        off[g * 8 : (g + 1) * 8, g * 8 : (g + 1) * 8] = 0
    assert off.abs().max() == 0
    dense_a = torch.block_diag(*model.first.weight)
    absorbed = transform @ dense_a
    plain = BlockShuffleLinear(64, 128, 8).double()
    with torch.no_grad():
        plain.second.weight.copy_(model.second.weight)
        for g in range(8):
            plain.first.weight[g].copy_(absorbed[g * 8 : (g + 1) * 8, g * 8 : (g + 1) * 8])
    first = model(identity)
    second = plain(identity)
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-14)
    save(
        "absorbable",
        {
            "passed": True,
            "off_block_max": off.abs().max().item(),
            "matrix_max_error": (first - second).abs().max().item(),
        },
        {
            "rotation": q.detach(),
            "absorbed_transform": transform.detach(),
            "rotated_matrix": first.detach(),
            "absorbed_matrix": second.detach(),
        },
    )


def test_independent_gradcheck():
    results = {}
    for shift in (0, 1):
        z = torch.randn(
            2,
            8,
            generator=torch.Generator().manual_seed(95017),
            dtype=torch.float64,
            requires_grad=True,
        )
        theta = torch.linspace(-0.3, 0.3, 4, dtype=torch.float64, requires_grad=True)
        results[str(shift)] = torch.autograd.gradcheck(
            lambda x, t: rotate_pairs(x, t, shift), (z, theta), eps=1e-6, atol=1e-6, rtol=1e-4
        )
    save("gradcheck", {"passed": all(results.values()), "checks": results})
    assert all(results.values())


def test_component_isometry():
    rows = []
    for shift in (0, 1):
        q = PairRotation(384, shift).double()
        with torch.no_grad():
            q.theta.copy_(torch.linspace(-math.pi, math.pi, 192, dtype=torch.float64))
        matrix = q(torch.eye(384, dtype=torch.float64)).T
        error = (matrix.T @ matrix - torch.eye(384, dtype=torch.float64)).abs().max().item()
        assert error <= 1e-13
        generator = torch.Generator().manual_seed(94017)
        x = torch.randn(16, 128, 384, generator=generator)
        incoming = torch.randn(x.shape, generator=generator)
        for device, dtype, tol in (("cpu", torch.float64, 1e-12), ("cuda", torch.bfloat16, 0.01)):
            model = copy.deepcopy(q).to(
                device=device, dtype=torch.float64 if device == "cpu" else torch.float32
            )
            inp = x.to(device=device, dtype=dtype).requires_grad_(True)
            grad = incoming.to(device=device, dtype=dtype)
            out = model(inp)
            input_grad = torch.autograd.grad(out, inp, grad)[0]
            forward = (out.double().norm(dim=-1) / inp.double().norm(dim=-1) - 1).abs().max().item()
            backward = (
                (input_grad.double().norm(dim=-1) / grad.double().norm(dim=-1) - 1)
                .abs()
                .max()
                .item()
            )
            assert forward <= tol and backward <= tol
            rows.append(
                {
                    "shift": shift,
                    "device": device,
                    "matrix_orthogonality_max_error": error,
                    "max_forward_norm_relative_error": forward,
                    "max_backward_norm_relative_error": backward,
                }
            )
    save("isometry", {"passed": True, "rows": rows, "whole_ffn_gradient_bound_claimed": False})
