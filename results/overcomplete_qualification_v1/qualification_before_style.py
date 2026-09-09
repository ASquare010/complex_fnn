"""H053 constructive representation, covariance calibration and native GPU checks."""

import argparse
import hashlib
import json
from pathlib import Path

PLAN = Path("research/overcomplete_qualification_plan.md")
ROOT = Path("results/overcomplete_qualification_v1")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def witness(module):
    """Exact constrained-factor construction, for the frozen 384/512 geometry."""
    import torch

    assert (module.width, module.expanded, module.groups, module.heads, module.hidden) == (
        384,
        512,
        16,
        8,
        200,
    )
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        first, second = module.input_projection.first.weight, module.input_projection.second.weight
        first.copy_(torch.eye(24, dtype=first.dtype).expand(16, -1, -1))
        for block in range(16):
            second[block, 0, :] = 1  # Sixteen block sums in head zero.
            for head in range(1, 7):
                rows = (2 * head, 2 * head + 1, 16 + 2 * head, 17 + 2 * head)
                for j, row in enumerate(rows):
                    second[block, row, 4 * (head - 1) + j] = 1
        a = module.input_projection(torch.eye(384, dtype=first.dtype)).T
        covered = []
        for head in range(1, 7):
            local = a[64 * head : 64 * (head + 1)]
            assert torch.equal(local.sum(-1), torch.ones(64, dtype=first.dtype))
            indices = local.argmax(-1)
            assert torch.equal(local, torch.eye(384, dtype=first.dtype)[indices])
            covered.extend(indices.tolist())
            for j, index in enumerate(indices.tolist()):
                module.gate[head, j, 2 * j] = module.value[head, j, 2 * j] = 1
                module.gate[head, j, 2 * j + 1] = module.value[head, j, 2 * j + 1] = -1
                module.down[head, 2 * j : 2 * j + 2, 0] = 0.5
                module.down[head, 2 * j : 2 * j + 2, 32] = 0.5 * (index + 1)
        assert sorted(covered) == list(range(384))
        assert torch.equal(a[:32:2].sum(0), torch.ones(384, dtype=first.dtype))
        for name in ("gate", "value"):
            p = getattr(module, name)
            p[0, :32:2, 0] = 1
            p[0, :32:2, 1] = -1
        module.down[0, :2, 1] = 0.5
        first, second = (
            module.output_projection.first.weight,
            module.output_projection.second.weight,
        )
        for head in range(1, 7):
            first[2 * head, 0, 0] = first[2 * head + 1, 0, 0] = 1
            second[0, 0, 2 * head] = 1
            second[0, 16, 2 * head + 1] = 1
        first[0, 2, 1] = 1
        second[1, 8, 8] = 1
    return {
        "covered_coordinate_count": len(covered),
        "coordinate_heads": [1, 2, 3, 4, 5, 6],
        "global_sum_head": 0,
        "unused_head": 7,
        "signed_hidden_units_used": 770,
    }


def dense_reference(seed):
    import torch

    from src.dense_ffn import DenseFFN

    m = DenseFFN(384, 1024, "swiglu")
    with torch.no_grad():
        for name, p in m.named_parameters():
            digest = hashlib.sha256(f"{seed}:blocks.0.ffn.{name}".encode()).digest()
            gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            p.normal_(0, 0.005 if name == "down.weight" else 0.02, generator=gen)
    return m


def cpu():
    import torch

    from src.core.reproducibility import provenance
    from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN

    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False)
    prior = json.loads(Path("results/verification/duration_decay_tests_v1.json").read_text())
    assert prior["returncode"] == 0 and prior["source_unchanged"]
    assert all(sha(n) == h for n, h in prior["source_files"].items())
    protocol = {
        "plan_sha256": sha(PLAN),
        "provenance": provenance(),
        "dimensions": dict(width=384, expanded=512, groups=16, heads=8, hidden=200),
        "seed_cohort": [17, 29, 43],
        "cpu_samples_per_seed": 4096,
        "cuda_updates": 10,
        "language_training_tokens": 0,
        "validation_targets_scored": 0,
    }
    write(ROOT / "protocol.json", protocol)
    model = OvercompleteHeadwiseFFN(**protocol["dimensions"]).double()
    proof = witness(model)
    gen = torch.Generator().manual_seed(713)
    x = torch.randn(31, 384, dtype=torch.float64, generator=gen)
    weights = torch.arange(1, 385, dtype=torch.float64)
    target = torch.zeros_like(x)
    target[:, 0] = 0.5 * x.square().sum(-1)
    target[:, 1] = 0.5 * (x.square() * weights).sum(-1)
    target[:, 2] = 0.5 * x.sum(-1).square()
    with torch.no_grad():
        actual = model(x)
    relative = (actual - target).norm().item() / target.norm().item()
    assert relative <= 1e-11
    point = torch.randn(384, dtype=torch.float64, generator=gen, requires_grad=True)
    output = model(point)
    directions = torch.randn(8, 384, dtype=torch.float64, generator=gen)
    errors = []
    for coordinate in range(3):
        first = torch.autograd.grad(
            output[coordinate], point, create_graph=True, retain_graph=True
        )[0]
        for v in directions:
            hv = torch.autograd.grad((first * v).sum(), point, retain_graph=True)[0]
            expected = (v, weights * v, torch.ones_like(v) * v.sum())[coordinate]
            error = (hv - expected).norm().item() / expected.norm().item()
            assert error <= 1e-11
            errors.append(error)
    proof.update(forward_relative_l2=relative, hessian_vector_relative_l2=errors)
    torch.save(model.state_dict(), ROOT / "witness_weights.pt")
    proof["weights_sha256"] = sha(ROOT / "witness_weights.pt")
    del model, actual, output, first
    calibration = []
    for seed in (17, 29, 43):
        double = OvercompleteHeadwiseFFN(**protocol["dimensions"]).double()
        double.initialize(seed, "blocks.0.ffn", 0.25)
        with torch.no_grad():
            a = double.input_projection(torch.eye(384, dtype=torch.float64)).T
            b = double.output_projection(torch.eye(512, dtype=torch.float64)).T
            input_error = (a.T @ a - torch.eye(384, dtype=torch.float64)).abs().max().item()
            output_error = (
                (b @ b.T - 0.0625 * torch.eye(384, dtype=torch.float64)).abs().max().item()
            )
        assert max(input_error, output_error) <= 1e-11
        model = OvercompleteHeadwiseFFN(**protocol["dimensions"])
        model.initialize(seed, "blocks.0.ffn", 0.25)
        full = dense_reference(seed)
        gen = torch.Generator().manual_seed(1000 + seed)
        energy, reference_energy = 0.0, 0.0
        with torch.no_grad():
            a = model.input_projection(torch.eye(384)).double().T
            covariance = a @ a.T
            heads = []
            for h in range(8):
                c = covariance[64 * h : 64 * (h + 1), 64 * h : 64 * (h + 1)]
                trace = c.trace().item()
                frob2 = c.square().sum().item()
                heads.append(
                    {
                        "head": h,
                        "covariance_trace": trace,
                        "effective_rank": trace**2 / frob2,
                        "gate_variance": 0.02**2 * 8 * trace,
                        "gate_variance_target": 0.02**2 * 384,
                        "squared_conditional_variance_expectation": (0.02**2 * 8) ** 2
                        * (trace**2 + 2 * frob2),
                    }
                )
            for _ in range(8):
                sample = torch.randn(512, 384, generator=gen)
                energy += model(sample).double().square().sum().item()
                reference_energy += full(sample).double().square().sum().item()
        ratio = (energy / reference_energy) ** 0.5
        assert 0.8 <= ratio <= 1.25
        calibration.append(
            {
                "seed": seed,
                "input_isometry_max_error": input_error,
                "output_isometry_max_error": output_error,
                "output_rms": (energy / (4096 * 384)) ** 0.5,
                "reference_rms": (reference_energy / (4096 * 384)) ** 0.5,
                "rms_ratio": ratio,
                "heads": heads,
            }
        )
    result = {
        "status": "PASS",
        "plan_sha256": sha(PLAN),
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "ffn_parameters": 350208,
        "witness": proof,
        "calibration": calibration,
        "all_old_tested_sources_exact": True,
        "earns_cuda_qualification": True,
        "language_training_tokens": 0,
        "validation_targets_scored": 0,
    }
    write(ROOT / "cpu.json", result)
    print(
        json.dumps(
            {
                "status": "CPU_PASS",
                "witness_relative_l2": relative,
                "max_hvp_relative_l2": max(errors),
                "rms_ratios": [r["rms_ratio"] for r in calibration],
            }
        ),
        flush=True,
    )


def cuda():
    import copy
    import gc

    import torch

    from src.core.reproducibility import environment
    from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN

    torch.set_num_threads(4)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    prior = json.loads((ROOT / "cpu.json").read_text())
    assert prior["status"] == "PASS" and prior["earns_cuda_qualification"]
    assert (
        sha(ROOT / "protocol.json") == prior["protocol_sha256"]
        and sha(PLAN) == protocol["plan_sha256"]
    )
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    base = OvercompleteHeadwiseFFN(**protocol["dimensions"])
    base.initialize(17, "blocks.0.ffn", 0.25)
    gen = torch.Generator().manual_seed(811)
    x_cpu = torch.randn(16, 128, 384, generator=gen)
    probe_cpu = torch.randn(16, 128, 384, generator=gen)
    model = copy.deepcopy(base).cuda()
    x = x_cpu.cuda().requires_grad_()
    probe = probe_cpu.cuda()
    variables = (x, *model.parameters())
    actual32 = model(x)
    g32 = torch.autograd.grad((actual32 * probe).mean(), variables)
    actual32 = actual32.detach()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual16 = model(x)
    g16 = torch.autograd.grad((actual16.float() * probe).mean(), variables)
    relative = lambda a, b: (
        (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-30)
    )
    output_error = relative(actual16, actual32)
    errors = [relative(a, b) for a, b in zip(g16, g32, strict=True)]
    assert output_error <= 0.02 and max(errors) <= 0.08
    assert all(torch.isfinite(t).all() and t.norm() > 0 for t in g32)
    names = ["input", *[n for n, _ in model.named_parameters()]]
    numerical = {
        "forward_relative_l2": output_error,
        "gradient_relative_l2": dict(zip(names, errors, strict=True)),
        "all_fp32_gradients_nonzero_and_finite": True,
    }
    del variables, actual32, actual16, g32, g16, model, x, probe
    gc.collect()
    torch.cuda.empty_cache()
    peaks = {}
    for name, source in (("overcomplete", base), ("full_swiglu", dense_reference(17))):
        model = copy.deepcopy(source).cuda()
        x = x_cpu.cuda().requires_grad_()
        probe = probe_cpu.cuda()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x)
        (y.float() * probe).mean().backward()
        torch.cuda.synchronize()
        peaks[name] = {
            "baseline_allocated_bytes": allocated,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "isolated eager FFN forward/backward; no optimizer state",
        }
        del model, x, probe, y
        gc.collect()
        torch.cuda.empty_cache()
    model = copy.deepcopy(base).cuda()
    x = x_cpu.cuda()
    target = 0.1 * torch.tanh(x) + 0.01 * torch.sin(x.roll(1, -1))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0012, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    steps = []
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(x)
        loss = (prediction.float() - target).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 0:
            assert all(p.grad.norm() > 0 for p in model.parameters())
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        steps.append({"step": step + 1, "loss": loss.item(), "gradient_norm_pre_clip": norm.item()})
    result = {
        "status": "PASS",
        "plan_sha256": sha(PLAN),
        "cpu_result_sha256": sha(ROOT / "cpu.json"),
        "numerical": numerical,
        "isolated_peaks": peaks,
        "synthetic_steps": steps,
        "environment": environment(),
        "earns_separate_integration": True,
        "language_training_tokens": 0,
        "validation_targets_scored": 0,
    }
    write(ROOT / "cuda.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cpu", "cuda"))
    args = parser.parse_args()
    {"cpu": cpu, "cuda": cuda}[args.command]()
