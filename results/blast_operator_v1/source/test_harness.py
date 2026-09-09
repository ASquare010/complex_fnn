"""The frozen 34-check H079 qualification; no optimizer or corpus access."""

import copy
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from results.blast_operator_v1.source.model import BlastFFN, BlastLinear, contract
from src.blockshuffle_ffn import BlockShuffleLinear, unshuffle_channels

ROOT = Path("results/blast_operator_v1/observations")
DIMS = [(384, 1984), (1984, 384), (384, 3200), (3200, 384)]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(v) for v in value]
    return value


def record(label, kind, data, tensors=None):
    result = {"status": "PASS", "kind": kind, "optimizer_updates": 0, **data}
    if tensors is not None:
        path = ROOT / (label + ".pt")
        assert not path.exists()
        torch.save(cpu(tensors), path)
        result["tensor_file"] = path.name
        result["tensor_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = ROOT / (label + ".json")
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dense(row, column, mix):
    return torch.cat([
        torch.cat([(row[i] * mix[:, i, j]) @ column[j] for j in range(len(column))], dim=1)
        for i in range(len(row))
    ], dim=0)


def tensor_pairs(actual, expected, exact=False):
    result = {}
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        other = expected[name]
        assert torch.isfinite(value).all() and torch.isfinite(other).all()
        assert value.double().norm() > 0 and other.double().norm() > 0
        if exact:
            assert torch.equal(value, other), name
        else:
            torch.testing.assert_close(value, other, rtol=1e-10, atol=1e-11)
        result[name] = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "max_absolute_error": (value.double() - other.double()).abs().max().item(),
            "norm": value.double().norm().item(), "exact": torch.equal(value, other),
        }
    return result


def test_shapes_and_counts():
    counts = {}
    for form in ("gelu", "swiglu"):
        ffn = BlastFFN(form)
        count = sum(p.numel() for p in ffn.parameters())
        assert count == 350208 and 8 * count == 2801664
        assert 8 * count + 6297984 == 9099648
        assert 100 * (1 - 8 * count / 9437184) == 70.3125
        assert len(list(ffn.parameters())) == (6 if form == "gelu" else 9)
        for module in (ffn.up, ffn.down) + ((ffn.gate,) if ffn.gate is not None else ()):
            assert sum(p.numel() for p in module.parameters()) == 48 * (
                module.input_width + module.output_width + 64
            )
        counts[form] = {"hidden": ffn.hidden, "ffn_weights_per_layer": count,
                        "ffn_weights_8_layers": 8 * count, "total_weights_8_layers": 9099648}
    for args in ((0, 8, 2, 2), (8, 13, 2, 2), (8, 12, 0, 2), (8, 12, 2, 5)):
        with pytest.raises(ValueError):
            BlastLinear(*args)
    with pytest.raises(ValueError):
        BlastFFN("relu")
    for scale in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            BlastLinear(8, 12, 2, 2, residual_scale=scale)
    module = BlastLinear(8, 12, 2, 2)
    for x in (torch.empty(()), torch.empty(2, 7)):
        with pytest.raises(ValueError):
            module(x)
    record("counts", "counts", {"counts": counts})


@pytest.mark.parametrize("n,m", DIMS)
@pytest.mark.parametrize("seed", [17, 29, 43])
def test_projection_fp64(n, m, seed):
    residual = 0.25 if n > m else 1.0
    rng_before = torch.get_rng_state()
    module = BlastLinear(n, m, seed=seed, residual_scale=residual, dtype=torch.float64)
    duplicate = BlastLinear(n, m, seed=seed, residual_scale=residual, dtype=torch.float64)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(a, b) for a, b in zip(module.parameters(), duplicate.parameters()))
    generator = torch.Generator().manual_seed(seed + 9000)
    x = torch.randn(2, 7, n, generator=generator, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(2, 7, m, generator=generator, dtype=torch.float64)
    params = tuple(module.parameters())
    y = module(x)
    grads = torch.autograd.grad((y * probe).sum(), (x, *params))
    other_x = x.detach().clone().requires_grad_()
    other_params = tuple(p.detach().clone().requires_grad_() for p in params)
    matrix = dense(*other_params)
    reference = other_x @ matrix.T
    ref_grads = torch.autograd.grad((reference * probe).sum(), (other_x, *other_params))
    names = ["output", "input", "row", "column", "mix"]
    actual = dict(zip(names, (y, *grads)))
    expected = dict(zip(names, (reference, *ref_grads)))
    comparisons = tensor_pairs(actual, expected)
    matrix = matrix.detach()
    gram = matrix.T @ matrix if n < m else matrix @ matrix.T
    target = torch.eye(384, dtype=torch.float64) * module.gain**2
    torch.testing.assert_close(gram, target, rtol=1e-10, atol=1e-11)
    row_energy = matrix.square().sum() / m
    target_energy = 0.02**2 * n * residual**2
    assert math.isclose(row_energy.item(), target_energy, rel_tol=1e-10)
    record(f"projection_{n}_{m}_seed{seed}", "projection", {
        "n": n, "m": m, "seed": seed, "gain": module.gain,
        "mean_row_squared_norm": row_energy.item(), "target_mean_row_squared_norm": target_energy,
        "gram_max_absolute_error": (gram - target).abs().max().item(),
        "comparisons": comparisons, "name_local_rng_preserved": True,
    }, {"input": x, "cotangent": probe, "factors": module.state_dict(),
        "matrix": matrix, "actual": actual, "expected": expected})


def test_small_finite_difference():
    module = BlastLinear(8, 12, 2, 3, dtype=torch.float64)
    gen = torch.Generator().manual_seed(79)
    x = torch.randn(2, 8, dtype=torch.float64, generator=gen, requires_grad=True)
    calls = [0]

    def forward(*args):
        calls[0] += 1
        return contract(*args)

    assert torch.autograd.gradcheck(
        forward, (x, *module.parameters()), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True,
    )
    record("gradcheck", "gradcheck", {"finite_difference_forward_calls": calls[0],
           "eps": 1e-6, "atol": 1e-5, "rtol": 1e-3})


@pytest.mark.parametrize("form", ["gelu", "swiglu"])
@pytest.mark.parametrize("seed", [17, 29, 43])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ffn_checkpoint(form, seed, device):
    shape = (2, 7, 384) if device == "cpu" else (16, 128, 384)
    module = BlastFFN(form, seed).to(device)
    other = copy.deepcopy(module)
    gen = torch.Generator().manual_seed(10000 + seed)
    x = torch.randn(shape, generator=gen).to(device).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    probe = torch.randn(shape, generator=gen).to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        y = module(x)
        loss = (y * probe).sum()
    grads = torch.autograd.grad(loss, (x, *module.parameters()))
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        y2 = checkpoint(other, x2, use_reentrant=False, preserve_rng_state=True)
        loss2 = (y2 * probe).sum()
    other_grads = torch.autograd.grad(loss2, (x2, *other.parameters()))
    names = ["output", "input", *dict(module.named_parameters())]
    actual, expected = dict(zip(names, (y, *grads))), dict(zip(names, (y2, *other_grads)))
    comparisons = tensor_pairs(actual, expected, exact=True)
    assert len(comparisons) == (8 if form == "gelu" else 11)
    record(f"ffn_{device}_{form}_seed{seed}", "ffn_checkpoint", {
        "form": form, "device": device, "seed": seed, "input_shape": list(shape),
        "comparisons": comparisons, "parameter_count": sum(p.numel() for p in module.parameters()),
    }, {"input": x, "cotangent": probe, "state": module.state_dict(),
        "actual": actual, "expected": expected})
    if device == "cuda":
        torch.cuda.synchronize()


@pytest.mark.parametrize("n,m", DIMS)
def test_exact_blockshuffle_embedding(n, m):
    old = BlockShuffleLinear(n, m, 8).double()
    gen = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for p in old.parameters():
            p.copy_(torch.randint(-2, 3, p.shape, generator=gen).double())
    module = BlastLinear(n, m, dtype=torch.float64)
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        for i in range(8):
            for j in range(8):
                k = 6 * ((i + j) % 8)
                module.row[i, :, k:k+6].copy_(old.second.weight[i, :, j::8])
                module.column[j, k:k+6, :].copy_(old.first.weight[j, 6*i:6*(i+1), :])
                module.mix[k:k+6, i, j] = 1
    shuffle = torch.arange(384).reshape(8, 48).T.reshape(-1)
    target = torch.block_diag(*old.second.weight) @ torch.block_diag(*old.first.weight)[shuffle]
    actual = dense(*module.parameters())
    assert torch.equal(target, actual)
    x = torch.randint(-2, 3, (2, 7, n), generator=gen).double()
    y, y2 = old(x), unshuffle_channels(module(x), 8)
    assert torch.equal(y, y2) and torch.isfinite(y).all() and y.norm() > 0
    record(f"embedding_{n}_{m}", "embedding", {
        "n": n, "m": m, "canonical_matrix_exact": True, "permuted_output_exact": True,
        "added_same_width_weights": 3072, "canonical_block_rank_bound": 6,
    }, {"first": old.first.weight, "second": old.second.weight,
        "factors": module.state_dict(), "matrix": actual, "reference_matrix": target,
        "input": x, "output": y2, "reference_output": y})


@pytest.mark.parametrize("n,m", DIMS)
def test_hadamard_witness(n, m):
    h = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(3):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    assert torch.equal(h.T @ h, 8 * torch.eye(8, dtype=torch.float64))
    module = BlastLinear(n, m, dtype=torch.float64)
    eye = torch.eye(48, dtype=torch.float64)
    with torch.no_grad():
        module.row.zero_()
        module.column.zero_()
        module.row[:, :48, :] = eye
        module.column[:, :, :48] = eye
        module.mix.copy_(h.expand(48, 8, 8))
    matrix = dense(*module.parameters())
    gram = matrix.T @ matrix if n < m else matrix @ matrix.T
    assert torch.equal(gram, 8 * torch.eye(384, dtype=torch.float64))
    for i in range(8):
        for j in range(8):
            block = matrix[i*(m//8):(i+1)*(m//8), j*(n//8):(j+1)*(n//8)]
            assert torch.equal(block[:48, :48], h[i, j] * eye)
            assert block.square().sum().item() == 48
    ones = dense(module.row, module.column, torch.ones_like(module.mix))
    # This explicit factorization and its identity minor certify rank exactly 48.
    left = torch.cat(list(module.row), 0)
    right = torch.cat(list(module.column), 1)
    assert torch.equal(ones, left @ right) and torch.equal(ones[:48, :48], eye)
    bound = Fraction(64 * (48 - 6), 64 * 48)
    assert bound == Fraction(7, 8) and matrix.square().sum().item() == 3072
    record(f"witness_{n}_{m}", "witness", {
        "n": n, "m": m, "global_rank": 384, "all_64_block_ranks": 48,
        "nonzero_squared_singular_value": 8, "target_squared_frobenius_norm": 3072,
        "rank6_block_squared_error_lower_bound": 2688, "relative_squared_bound": [7, 8],
        "all_ones_global_rank": 48,
    }, {"hadamard": h, "factors": module.state_dict(), "matrix": matrix,
        "all_ones_matrix": ones, "all_ones_left": left, "all_ones_right": right})
