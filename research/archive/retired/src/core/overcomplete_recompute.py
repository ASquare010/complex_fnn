"""H055 execution-only native gate recomputation, reusing the H054 pipeline."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

from src.core.overcomplete_integration import CANDIDATE
from src.core.overcomplete_integration import configuration as eager_configuration
from src.core.overcomplete_integration import gpu as pipeline_gpu

PLAN = Path("research/overcomplete_recompute_plan.md")
ROOT = Path("results/overcomplete_recompute_v1")
PRIOR = Path("results/overcomplete_integration_v1")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration(variant):
    from dataclasses import replace

    if variant != CANDIDATE:
        raise ValueError("Only the frozen recomputed candidate is new")
    mc, tc = eager_configuration(variant)
    return mc, replace(tc, recompute_gate=True, gate_recompute_method="native")


def preflight():
    from dataclasses import asdict

    import torch

    from src.core.benchmark import forward_flops
    from src.core.config import VARIANTS, ModelConfig
    from src.core.optimization import group_summary, parameter_groups
    from src.core.reproducibility import environment, provenance
    from src.core.trainer import configure_gate_recomputation
    from src.core.transformer import Transformer

    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False)
    final = read("results/verification/overcomplete_integration_final_v1.json")
    assert final["status"] == "PASS" and final["full_tests_passed"] == 241
    assert (
        sha(PRIOR / "result.json") == final["result_sha256"] and not final["earns_quality_screen"]
    )
    snap = Path("results/verification/overcomplete_recompute_before_v1.pt")
    meta = read(snap.with_suffix(".json"))
    assert meta["variants"] == 32 and sha(snap) == meta["sha256"]
    changed = {n for n, h in meta["provenance"]["source_files"].items() if sha(n) != h}
    assert changed == {
        "src/multihead_ffn/overcomplete.py",
        "src/core/trainer.py",
        "src/core/overcomplete_integration.py",
    }, changed
    before = torch.load(snap, map_location="cpu", weights_only=True)
    assert set(before) == set(VARIANTS)
    x = torch.arange(16).reshape(2, 8)
    for variant, row in before.items():
        c = ModelConfig(**row["config"])
        m = Transformer(c, 17)
        loss = m.loss(x, x + 1)
        loss.backward()
        assert torch.equal(m(x), row["logits"]) and torch.equal(loss, row["loss"])
        assert all(torch.equal(p, row["state"][n]) for n, p in m.state_dict().items())
        assert all(torch.equal(p.grad, row["gradients"][n]) for n, p in m.named_parameters())
        assert forward_flops(c) == row["flops"]
    references = {}
    prior_result = read(PRIOR / "result.json")
    for variant, trial in prior_result["trials"].items():
        path = PRIOR / variant
        assert sha(path / "result.json") == trial["result_sha256"]
        cell = read(path / "result.json")
        assert all(sha(path / n) == h for n, h in cell["files"].items())
        with zipfile.ZipFile(PRIOR / "source.zip") as z:
            assert all(
                hashlib.sha256(z.read(n)).hexdigest() == h
                for n, h in cell["provenance"]["source_files"].items()
            )
        references[variant] = {
            "path": path.as_posix(),
            "result_sha256": sha(path / "result.json"),
            "files": cell["files"],
        }
    old = read(PRIOR / CANDIDATE / "result.json")
    c, tc = configuration(CANDIDATE)
    assert asdict(c) == old["model"]
    config_diff = {k for k, v in asdict(tc).items() if v != old["training"][k]}
    assert config_diff == {"recompute_gate", "gate_recompute_method"}
    m = Transformer(c, 17)
    initial = {n: p.detach().clone() for n, p in m.named_parameters()}
    configure_gate_recomputation(m, tc)
    assert all(b.ffn.recompute_gate and b.ffn.gate_recompute_method == "native" for b in m.blocks)
    assert all(torch.equal(p, initial[n]) for n, p in m.named_parameters())
    assert group_summary(parameter_groups(m, tc)) == old["optimizer_parameter_groups"]
    assert c.unique_ffn_parameters == 2801664 and c.total_parameters == 9099648
    assert environment() == old["environment"]
    for n, h in old["data_hashes"].items():
        assert sha(Path("data/wikitext2_v1") / n) == h
    initial = {
        "status": "PASS",
        "old_variants_exact": 32,
        "snapshot_sha256": sha(snap),
        "qualified_changed_files": sorted(changed),
        "initial_weights_and_actual_groups_unchanged": True,
        "references": references,
        "data_hashes": old["data_hashes"],
        "environment": environment(),
    }
    write(ROOT / "preflight.json", initial)
    protocol = {
        "plan_sha256": sha(PLAN),
        "preflight_sha256": sha(ROOT / "preflight.json"),
        "provenance": provenance(),
        "configurations": {CANDIDATE: {"model": asdict(c), "training": asdict(tc)}},
        "new_training_token_exposures": 40960,
        "validation_targets_scored": 0,
        "optimizer_updates": 20,
    }
    write(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for n in protocol["provenance"]["source_files"]:
            z.write(n, n)
        z.write(PLAN, PLAN.as_posix())
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS",
                "old_variants_exact": 32,
                "changed_training_fields": sorted(config_diff),
            }
        ),
        flush=True,
    )


def numerical_equivalence(model, x, y, training):
    import torch

    from src.core.benchmark import autocast
    from src.core.trainer import configure_gate_recomputation

    parameters = tuple(model.parameters())
    for block in model.blocks:
        block.ffn.recompute_gate = False
    with autocast("cuda", "bf16"):
        eager_loss = model.loss(x, y)
    eager = [g.detach().cpu() for g in torch.autograd.grad(eager_loss, parameters)]
    configure_gate_recomputation(model, training)
    with autocast("cuda", "bf16"):
        native_loss = model.loss(x, y)
    native = [g.detach().cpu() for g in torch.autograd.grad(native_loss, parameters)]
    assert abs(native_loss.item() - eager_loss.item()) <= 1e-7
    reference = read(PRIOR / CANDIDATE / "result.json")
    assert abs(eager_loss.item() - reference["initial_fixed_batch_loss"]) <= 1e-7
    errors = {}
    for (name, _), a, b in zip(model.named_parameters(), native, eager, strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all() and b.norm() > 0
        error = (a - b).norm().item() / b.norm().item()
        assert math.isfinite(error) and error <= 0.02, (name, error)
        errors[name] = error
    return {
        "status": "PASS",
        "eager_initial_loss": eager_loss.item(),
        "native_initial_loss": native_loss.item(),
        "gradient_relative_l2": errors,
        "maximum_gradient_relative_l2": max(errors.values()),
        "optimizer_updates": 0,
        "additional_sampler_batches": 0,
        "validation_targets_scored": 0,
    }


def finish():
    protocol = read(ROOT / "protocol.json")
    pre = read(ROOT / "preflight.json")
    assert (
        sha(PLAN) == protocol["plan_sha256"]
        and sha(ROOT / "preflight.json") == protocol["preflight_sha256"]
    )
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    path = ROOT / CANDIDATE
    result = read(path / "result.json")
    assert result["status"] == "PASS" and result["execution_check"]["status"] == "PASS"
    assert result["provenance"]["source_files"] == protocol["provenance"]["source_files"]
    assert all(sha(path / n) == h for n, h in result["files"].items())
    refs = {}
    for variant, ref in pre["references"].items():
        p = Path(ref["path"])
        assert sha(p / "result.json") == ref["result_sha256"]
        assert all(sha(p / n) == h for n, h in ref["files"].items())
        refs[variant] = read(p / "result.json")
    old = refs[CANDIDATE]
    assert abs(result["initial_fixed_batch_loss"] - old["initial_fixed_batch_loss"]) <= 1e-7
    assert (
        result["sampling_rng_sha256"] == old["sampling_rng_sha256"]
        and result["data_hashes"] == old["data_hashes"]
    )
    assert result["optimizer_parameter_groups"] == old["optimizer_parameter_groups"]
    peak = result["peak_training_allocated_bytes"]
    gates = {
        "at_least_70_percent_fewer_ffn_weights": result["ffn_parameters"]
        <= 0.3 * refs["swiglu"]["ffn_parameters"],
        "strict_memory_reduction_vs_eager": peak < old["peak_training_allocated_bytes"],
    }
    for variant in ("swiglu", "gelu"):
        gates["memory_within_ten_percent_" + variant] = (
            peak <= 1.1 * refs[variant]["peak_training_allocated_bytes"]
        )
    record = {
        "status": "complete",
        "plan_sha256": sha(PLAN),
        "gates": gates,
        "earns_separate_quality_screen": all(gates.values()),
        "quality_measured": False,
        "old_variants_exact": 32,
        "new_training_token_exposures": 40960,
        "validation_targets_scored": 0,
        "peak_mib": peak / 2**20,
        "eager_peak_mib": old["peak_training_allocated_bytes"] / 2**20,
        "relative_memory_percent": {
            v: 100 * (peak / r["peak_training_allocated_bytes"] - 1) for v, r in refs.items()
        },
        "maximum_native_eager_gradient_relative_l2": result["execution_check"][
            "maximum_gradient_relative_l2"
        ],
        "checkpoint_sha256": result["files"]["checkpoint.pt"],
        "trial_result_sha256": sha(path / "result.json"),
        "numerical_failures": 0,
    }
    write(ROOT / "result.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "gpu", "finish"))
    args = parser.parse_args()
    if args.command == "gpu":
        pipeline_gpu(
            CANDIDATE,
            root=ROOT,
            plan=PLAN,
            config_factory=configuration,
            execution_check=numerical_equivalence,
        )
    else:
        {"preflight": preflight, "finish": finish}[args.command]()
