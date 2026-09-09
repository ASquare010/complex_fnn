"""H086 qualification: fixed thresholds, no training or corpus access."""

import copy
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.blast_operator_recovery_v1.source.storage import record_observation
from results.compact_sparse_operator_v1.source.model import (
    CachedSparseFFN,
    CompactSparseFFN,
    CompactSparseLinear,
)

ROOT = Path("results/compact_sparse_operator_v1/observations")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item) for item in value]
    return value


def record(label, kind, data, tensors=None):
    record_observation(ROOT, label, kind, data, cpu(tensors) if tensors is not None else None)


def independent_dense(values, positions):
    # Four separate scalar coordinate assignments, independent of scatter.
    blocks = values.new_zeros(*values.shape[:-1], 4)
    for coordinate in range(4):
        blocks[..., coordinate] = (values * (positions == coordinate)).sum(-1)
    return blocks.flatten(1)


def pairs(actual, expected, atol, rtol, exact=False):
    assert actual.keys() == expected.keys()
    summary = {}
    for name in actual:
        a, b = actual[name], expected[name]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        if exact:
            assert torch.equal(a, b), name
        else:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
        summary[name] = {"max_absolute_error": (a.double() - b.double()).abs().max().item(),
                         "exact": torch.equal(a, b)}
    return summary


def test_counts_storage():
    rows, tensors = {}, {}
    for form in ("gelu", "swiglu"):
        model = CompactSparseFFN(form)
        count = sum(p.numel() for p in model.parameters())
        buffers = sum(p.numel() * p.element_size() for p in model.buffers())
        assert count == 350208 and buffers == 350208
        assert all(name.endswith("values") for name, _ in model.named_parameters())
        assert all(p.dtype == torch.uint8 for p in model.buffers())
        rows[form] = {"hidden": model.hidden, "learned_values": count,
                      "fp32_parameter_bytes": 4 * count, "mask_bytes": buffers,
                      "two_fp32_adam_moments_bytes": 8 * count,
                      "aggregate_dense_temporary_bytes": 8 * count,
                      "aggregate_int64_index_bytes": 8 * count,
                      "ffn_weights_8_layers": count * 8,
                      "total_model_weights_by_count": count * 8 + 6297984,
                      "ffn_reduction_percent": 100 * (1 - count * 8 / 9437184)}
        tensors[form] = model.state_dict()
    record("counts", "counts", rows, tensors)


def test_shapes_support():
    for dims in ((0, 4), (7, 4), (8, 0), (-4, 8)):
        with pytest.raises(ValueError):
            CompactSparseLinear(*dims)
    with pytest.raises(ValueError):
        CompactSparseFFN("relu")
    model = CompactSparseLinear(384, 608)
    assert torch.equal(model.positions, CompactSparseLinear(384, 608).positions)
    assert not torch.equal(model.positions, CompactSparseLinear(384, 608, 29).positions)
    assert ((model.positions[..., 0] < model.positions[..., 1]) &
            (model.positions[..., 1] < 4)).all()
    for x in (torch.empty(()), torch.empty(2, 383)):
        with pytest.raises(ValueError):
            model(x)
    record("support", "support", {"distinct_and_deterministic": True},
           {"positions": model.positions})


@pytest.mark.parametrize("m,n", [(12, 8), (64, 48), (384, 384), (608, 384)])
def test_projection(m, n):
    model = CompactSparseLinear(n, m).double()
    rng = torch.Generator().manual_seed(9821)
    x = torch.randn(3, n, generator=rng, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(3, m, generator=rng, dtype=torch.float64)
    dense = independent_dense(model.values.detach(), model.positions)
    actual_dense = model.materialize()
    assert torch.equal(actual_dense, dense)
    delta = torch.randn(model.values.shape, generator=rng, dtype=torch.float64)
    lifted = independent_dense(delta, model.positions)
    torch.testing.assert_close(lifted.square().sum(), delta.square().sum(), atol=1e-11, rtol=1e-10)
    y = model(x)
    dx, dv = torch.autograd.grad(y, (x, model.values), cotangent)
    weight_gradient = cotangent.T @ x.detach()
    expected = {"output": x.detach() @ dense.T, "input_gradient": cotangent @ dense,
                "value_gradient": weight_gradient.reshape(m, n // 4, 4).gather(-1, model.positions.long())}
    actual = {"output": y, "input_gradient": dx, "value_gradient": dv}
    summary = pairs(actual, expected, 1e-11, 1e-10)
    record(f"projection_{m}_{n}", "projection", {"pairs": summary,
           "mean_row_energy": dense.square().sum(-1).mean().item()},
           {"actual": actual, "expected": expected, "values": model.values,
            "positions": model.positions, "matrix": dense, "input": x,
            "cotangent": cotangent, "delta": delta, "lifted_delta": lifted})


EXECUTIONS = ["cpu64", "cpu32", "cuda32", "cuda_bf16"]


@pytest.mark.parametrize("form", ["gelu", "swiglu"])
@pytest.mark.parametrize("execution", EXECUTIONS)
def test_ffn(form, execution):
    device = "cuda" if execution.startswith("cuda") else "cpu"
    dtype = torch.float64 if execution == "cpu64" else torch.float32
    atol, rtol = (1e-11, 1e-10) if execution == "cpu64" else (2e-6, 2e-5)
    if execution == "cuda_bf16":
        atol, rtol = 0.02, 0.02

    def context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if execution == "cuda_bf16" else nullcontext()

    model = CompactSparseFFN(form).to(device=device, dtype=dtype)
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict())
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    rng = torch.Generator().manual_seed(9822)
    initial = torch.randn(3, 2, 384, generator=rng, dtype=dtype).transpose(0, 1).to(device)
    x = initial.detach().requires_grad_()
    assert not x.is_contiguous()
    cotangent = torch.randn(2, 3, 384, generator=rng, dtype=dtype).to(device)
    names = [name for name, _ in model.named_parameters()]
    with context():
        y = model(x)
    gradients = torch.autograd.grad(y, (x, *model.parameters()), cotangent)
    actual = dict(zip(["output", "input_gradient", *names], (y, *gradients)))

    x_ref = initial.detach().clone().requires_grad_()
    weights, masks = {}, {}
    for name in ("up", "down", "gate"):
        module = getattr(model, name)
        if module is not None:
            weights[name] = independent_dense(module.values.detach(), module.positions).requires_grad_()
            masks[name] = module.positions
    with context():
        z = F.linear(x_ref, weights["up"])
        z = F.gelu(z) if form == "gelu" else F.silu(z) * F.linear(x_ref, weights["gate"])
        y_ref = F.linear(z, weights["down"])
    grads_ref = torch.autograd.grad(y_ref, (x_ref, *weights.values()), cotangent)
    expected = {"output": y_ref, "input_gradient": grads_ref[0]}
    for (name, weight), grad in zip(weights.items(), grads_ref[1:]):
        expected[name + ".values"] = grad.reshape(weight.shape[0], weight.shape[1] // 4, 4).gather(
            -1, masks[name].long())
    summary = pairs(actual, expected, atol, rtol)
    x_cp = initial.detach().clone().requires_grad_()
    with context():
        y_cp = checkpoint(restored, x_cp, use_reentrant=False)
    grads_cp = torch.autograd.grad(y_cp, (x_cp, *restored.parameters()), cotangent)
    cp = dict(zip(["output", "input_gradient", *names], (y_cp, *grads_cp)))
    cp_summary = pairs(actual, cp, 0, 0, exact=True)
    record(f"ffn_{form}_{execution}", "ffn", {"pairs": summary, "checkpoint_pairs": cp_summary,
           "atol": atol, "rtol": rtol}, {"actual": actual, "expected": expected,
           "checkpoint": cp, "state": model.state_dict(), "input": initial,
           "cotangent": cotangent})


def test_finite_difference():
    model = CompactSparseLinear(8, 4).double()
    x = torch.linspace(-0.7, 0.8, 16, dtype=torch.float64).reshape(2, 8).requires_grad_()
    value = model(x).square().sum()
    dx, dv = torch.autograd.grad(value, (x, model.values))
    rows = []
    for tensor, gradient, indices in ((model.values, dv, [0, 7, 15]), (x, dx, [1, 14])):
        for index in indices:
            with torch.no_grad():
                original = tensor.flatten()[index].item()
                tensor.flatten()[index] = original + 1e-6
                plus = model(x).square().sum().item()
                tensor.flatten()[index] = original - 1e-6
                minus = model(x).square().sum().item()
                tensor.flatten()[index] = original
            numerical = (plus - minus) / 2e-6
            analytic = gradient.flatten()[index].item()
            assert abs(numerical - analytic) <= 1e-7 + 1e-5 * abs(analytic)
            rows.append({"coordinate": index, "numerical": numerical, "analytic": analytic})
    record("finite_difference", "finite_difference", {"central_difference_forwards": 10, "rows": rows})


def test_rank_witness():
    h = torch.ones(1, 1, dtype=torch.int64)
    for _ in range(3):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    witness = torch.kron(h, torch.eye(48, dtype=torch.int64))
    positions = torch.stack((torch.arange(384) % 4, (torch.arange(384) + 1) % 4), -1)
    positions = positions[:, None, :].expand(-1, 96, -1).to(torch.uint8).contiguous()
    values = witness.reshape(384, 96, 4).gather(-1, positions.long())
    assert torch.equal(independent_dense(values, positions), witness)
    assert torch.equal(witness @ witness.T, 8 * torch.eye(384, dtype=torch.int64))
    for i in range(8):
        for j in range(8):
            block = witness[48*i:48*(i+1), 48*j:48*(j+1)]
            assert torch.equal(block @ block.T, torch.eye(48, dtype=torch.int64))
    record("rank_witness", "rank_witness", {"block_rank": 48, "global_rank": 384,
           "squared_relative_error_floor_numerator": 7, "denominator": 8,
           "not_equal_count_projection_comparison": True},
           {"matrix": witness, "positions": positions, "values": values})


def capability_error(exc):
    message = str(exc).lower()
    # Shape errors, unknown errors and device faults are deliberately not accepted.
    return any(token in message for token in (
        "not compiled with", "not available", "not supported on this",
        "not implemented for", "could not run 'aten::_cslt", "could not run 'aten::_sparse",
        "cusparselt is not supported", "cusparselt is not enabled"))


@pytest.mark.parametrize("form", ["gelu", "swiglu"])
@pytest.mark.parametrize("backend", ["cutlass", "cusparselt"])
@pytest.mark.parametrize("batch", [16, 256, 2048])
def test_hardware(form, backend, batch):
    label = f"hardware_{form}_{backend}_{batch}"
    if backend == "cusparselt" and not torch.backends.cusparselt.is_available():
        record(label, "hardware", {"status": "UNAVAILABLE", "backend": backend,
               "form": form, "batch": batch, "reason": "torch.backends.cusparselt.is_available() is False"})
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = CompactSparseFFN(form).cuda().eval()
    rng = torch.Generator().manual_seed(9823)
    x = torch.randn(batch, 384, generator=rng).to(device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            cached = CachedSparseFFN(model, backend)
            torch.cuda.synchronize()
            conversion_seconds = time.perf_counter() - started
            actual = cached(x)
            torch.cuda.synchronize()
        except (RuntimeError, NotImplementedError) as exc:
            if not capability_error(exc):
                raise
            record(label, "hardware", {"status": "UNAVAILABLE", "backend": backend,
                   "form": form, "batch": batch, "reason": str(exc), "exception_type": type(exc).__name__})
            return
        logical, padded = cached(x, "dense"), cached(x, "padded")
        summary = pairs({"logical": actual, "padded": actual},
                        {"logical": logical, "padded": padded}, 0.03, 0.03)
        timings = {}
        for mode in ("sparse", "dense", "native"):
            def run():
                if mode == "native":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return model(x)
                return cached(x, mode)
            for _ in range(10):
                run()
            blocks = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    run()
                torch.cuda.synchronize()
                blocks.append(1000 * (time.perf_counter() - start) / 20)
            timings[mode] = {"block_ms_per_call": blocks, "median_ms": statistics.median(blocks)}
        record(label, "hardware", {"backend": backend, "form": form, "batch": batch,
               "pairs": summary, "conversion_seconds": conversion_seconds,
               "storage": cached.storage, "timing": timings,
               "combined_resident_peak_bytes": torch.cuda.max_memory_allocated(),
               "peak_is_not_per_model_comparison": True},
               {"actual": actual, "logical": logical, "padded": padded})
