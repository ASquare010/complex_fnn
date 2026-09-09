"""Fixed H065 mathematical checks and twelve exact product comparisons."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.rational_staged_v1.source.candidate import StagedRationalActivation, staged_residual
from src.core.native_recompute_audit import digest, finite_tree, read, sha, tensor_record, write_new
from src.core.reproducibility import environment
from src.rational_blockshuffle_ffn import RationalResidualActivation

ROOT = Path("results/rational_staged_v1")
PLAN = Path("research/rational_staged_plan.md")
PLAN_SHA = "eecf4e93cce53c3a24c01f8f0d28f692311d4f2aa69a7751d9489178151319ad"


def compare(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    lhs, rhs = left.contiguous(), right.contiguous()
    mismatches = (
        lhs.reshape(-1).view(torch.uint8).reshape(lhs.numel(), -1)
        != rhs.reshape(-1).view(torch.uint8).reshape(rhs.numel(), -1)
    ).any(-1)
    delta = rhs.double() - lhs.double()
    exact = tensor_record(lhs) == tensor_record(rhs)
    return {
        "exact": exact,
        "finite": finite_tree([lhs, rhs]),
        "elements": lhs.numel(),
        "mismatched_elements": int(mismatches.sum()),
        "max_abs_error": float(delta.abs().max()),
        "relative_l2_error": float(delta.norm() / lhs.double().norm().clamp_min(1e-30)),
        "reference": tensor_record(lhs),
        "candidate": tensor_record(rhs),
    }


def mathematical_checks():
    z = torch.linspace(-1.4, 1.7, 12, dtype=torch.float64).reshape(2, 2, 3).requires_grad_()
    a = torch.linspace(-0.3, 0.2, 8, dtype=torch.float64).reshape(2, 4).requires_grad_()
    b = torch.tensor([-0.2, 0.3], dtype=torch.float64, requires_grad=True)
    actual = staged_residual(z, a, b)
    beta = (1 + 0.75 * b.tanh()).unsqueeze(-1)
    direct = (
        0.25
        * sum(c.unsqueeze(-1) * z.pow(j) for j, c in enumerate(a.tanh().unbind(-1)))
        / (1 + beta * z.square())
    )
    torch.testing.assert_close(actual, direct, rtol=2e-12, atol=2e-12)
    assert torch.autograd.gradcheck(
        staged_residual, (z, a, b), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True
    )
    module = StagedRationalActivation(64, 8)
    native = RationalResidualActivation(64, 8)
    assert set(module.state_dict()) == set(native.state_dict())
    assert sum(p.numel() for p in module.parameters()) == 40
    inputs = torch.linspace(-4, 4, 64).reshape(1, 64)
    assert torch.equal(module(inputs), F.silu(inputs))
    with patch(
        "results.rational_staged_v1.source.candidate.checkpoint",
        side_effect=AssertionError("Checkpoint must be bypassed"),
    ):
        module.eval()(inputs.requires_grad_())
        with torch.no_grad():
            module.train()(inputs)
    return {
        "status": "PASS",
        "fp64_direct_formula": True,
        "independent_gradcheck": True,
        "zero_shape_exact": True,
        "state_keys_exact": True,
        "activation_parameters": 40,
        "eval_no_grad_bypass": True,
    }


def execute(kind, shape, dtype, device, state, raw):
    module = (RationalResidualActivation if kind == "native" else StagedRationalActivation)(
        shape[-1], 8
    ).to(device)
    module.load_state_dict(state, strict=True)
    x, v, incoming = [p.to(device=device, dtype=dtype) for p in raw]
    x = x.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    before = digest(module.state_dict())

    def product(u, gate):
        return module(u) * gate

    output = checkpoint(product, x, v, use_reentrant=False, preserve_rng_state=False)
    gradient_tensors = torch.autograd.grad(
        output, (x, v, *module.parameters()), grad_outputs=incoming
    )
    names = ["u", "v", *dict(module.named_parameters())]
    assert digest(module.state_dict()) == before
    assert finite_tree([output, *gradient_tensors])
    return {
        "output": output.detach().cpu(),
        "gradients": {n: g.detach().cpu() for n, g in zip(names, gradient_tensors)},
    }


def main():
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"] == PLAN_SHA
    assert environment() == protocol["environment"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    math_result = mathematical_checks()
    write_new(ROOT / "mathematical_checks.json", math_result)
    source = Path("results/runs") / protocol["input"]["run"]
    assert all(sha(source / n) == h for n, h in protocol["input"]["files"].items())
    trained = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert trained["step"] == 200
    results = {}
    for seed, layer in ((17, 0), (29, 3), (43, 7)):
        for device, dtype, shape in (
            ("cpu", torch.float32, (3, 11, 64)),
            ("cuda", torch.bfloat16, (16, 128, 2048)),
        ):
            generator = torch.Generator().manual_seed(70000 + seed)
            raw = [torch.randn(shape, generator=generator) for _ in range(3)]
            for state_kind in ("zero", "trained"):
                state = {
                    name: trained["model"][f"blocks.{layer}.ffn.curve.{name}"].clone()
                    for name in ("theta_coefficients", "theta_denominator")
                }
                if state_kind == "zero":
                    state = {n: torch.zeros_like(t) for n, t in state.items()}
                cell = f"{device}_s{seed}_{state_kind}"
                folder = ROOT / "cases" / cell
                folder.mkdir(parents=True, exist_ok=False)
                before_rng = {
                    "cpu": tensor_record(torch.get_rng_state()),
                    "cuda": tensor_record(torch.cuda.get_rng_state()),
                }
                outputs = {
                    kind: execute(kind, shape, dtype, device, state, raw)
                    for kind in ("native", "staged")
                }
                after_rng = {
                    "cpu": tensor_record(torch.get_rng_state()),
                    "cuda": tensor_record(torch.cuda.get_rng_state()),
                }
                assert before_rng == after_rng
                comparison = {
                    "output": compare(outputs["native"]["output"], outputs["staged"]["output"]),
                    **{
                        n: compare(
                            outputs["native"]["gradients"][n], outputs["staged"]["gradients"][n]
                        )
                        for n in outputs["native"]["gradients"]
                    },
                }
                artifact = folder / "tensors.pt"
                torch.save({"input_fp32": raw, "state": state, "outputs": outputs}, artifact)
                row = {
                    "case": cell,
                    "seed": seed,
                    "coefficient_source_layer": layer if state_kind == "trained" else None,
                    "device": device,
                    "dtype": str(dtype),
                    "shape": list(shape),
                    "state": state_kind,
                    "input_fp32_hashes": [tensor_record(t) for t in raw],
                    "input_cast_hashes": [tensor_record(t.to(dtype)) for t in raw],
                    "state_hash": digest(state),
                    "comparisons": comparison,
                    "exact": all(v["exact"] for v in comparison.values()),
                    "finite": all(v["finite"] for v in comparison.values()),
                    "rng_unchanged": True,
                    "parameters_unchanged": True,
                    "optimizer_updates": 0,
                    "tensors_sha256": sha(artifact),
                }
                write_new(folder / "result.json", row)
                results[cell] = row
                print(
                    json.dumps(
                        {
                            "case": cell,
                            "exact": row["exact"],
                            "failed_tensors": [n for n, c in comparison.items() if not c["exact"]],
                        }
                    ),
                    flush=True,
                )
                del outputs, comparison
                if device == "cuda":
                    torch.cuda.empty_cache()
    assert len(results) == 12
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    passed = all(r["exact"] and r["finite"] for r in results.values())
    result = {
        "status": "complete",
        "scientific_verdict": "LOCALLY_QUALIFIED" if passed else "REJECTED_FIDELITY",
        "plan_sha256": PLAN_SHA,
        "cases": results,
        "mathematical_checks": math_result,
        "all_cases_exact": passed,
        "optimizer_updates": 0,
        "corpus_targets": 0,
        "full_model_resource_workers": 0,
        "earns_full_model_qualification": passed,
        "research_goal_achieved": False,
    }
    write_new(ROOT / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("cases", "mathematical_checks")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
