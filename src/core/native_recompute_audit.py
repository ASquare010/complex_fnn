"""H063: equally checkpointed native resource and update-fidelity qualification."""

import argparse
import ast
import hashlib
import json
import math
import runpy
import statistics
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.benchmark import autocast
from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment, provenance
from src.core.trainer import configure_gate_recomputation
from src.core.transformer import Transformer

ROOT = Path("results/native_recompute_v1")
PLAN = Path("research/native_recompute_plan.md")
PLAN_SHA = "66b437bea6669abd55bc6031fc45252d54135e71412773e8f93954ce48d8c024"
SCOPES = ("none", "ffn", "block")
RECIPES = ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle", "rational")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_record(value):
    data = value.detach().cpu().contiguous()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sha256": hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def digest(value):
    def normalized(item):
        if isinstance(item, torch.Tensor):
            return tensor_record(item)
        if isinstance(item, dict):
            return {str(k): normalized(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalized(v) for v in item]
        return item

    return hashlib.sha256(
        json.dumps(normalized(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def tokens():
    return torch.randint(0, 4096, (20, 16, 129), generator=torch.Generator().manual_seed(60017))


def resource_gates(candidate, native_candidate, references):
    if any(row["scope"] != candidate["scope"] for row in references.values()):
        raise ValueError("Memory gates require the same recomputation scope")
    return {
        "at_least_70_percent_fewer_ffn_weights": candidate["ffn_reduction_percent"] >= 70,
        "at_least_ten_percent_native_memory_reduction": candidate["peak_allocated_bytes"]
        <= 0.9 * native_candidate["peak_allocated_bytes"],
        **{
            f"same_scope_memory_within_ten_percent_{name}": candidate["peak_allocated_bytes"]
            <= 1.1 * references[name]["peak_allocated_bytes"]
            for name in ("full_swiglu", "full_gelu")
        },
    }


def preflight():
    assert sha(PLAN) == PLAN_SHA
    assert not (ROOT / "protocol.json").exists()
    before = read(ROOT / "before.json")
    assert sha(ROOT / "before_source.zip") == before["before_archive_sha256"]
    assert (
        sha("results/verification/additive_retirement_final_v1.json") == before["h062_audit_sha256"]
    )
    assert all(
        sha(n) == h for n, h in before["source_hashes"].items() if n != "src/core/transformer.py"
    )
    with zipfile.ZipFile(ROOT / "before_source.zip") as archive:
        old = ast.parse(archive.read("src/core/transformer.py"))
    current = ast.parse(Path("src/core/transformer.py").read_bytes())
    old_classes = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
    current_classes = {n.name: n for n in current.body if isinstance(n, ast.ClassDef)}
    for name in ("RMSNorm", "Attention"):
        assert ast.dump(old_classes[name]) == ast.dump(current_classes[name])
    for name in ("__init__", "forward", "loss"):
        methods = [
            {n.name: ast.dump(n) for n in c["Transformer"].body if isinstance(n, ast.FunctionDef)}
            for c in (old_classes, current_classes)
        ]
        assert methods[0][name] == methods[1][name]
    suite = read(ROOT / "processes/full_tests.json")
    assert suite["status"] == "PASS" and suite["source_unchanged"] and suite["returncode"] == 0
    assert all(sha(n) == h for n, h in suite["sources"].items())
    torch.set_num_threads(4)
    signatures = runpy.run_path("results/verification/cleanup_signatures_v1.py")["signatures"]()
    assert signatures == read("results/verification/cleanup_before_signatures_v1.json")
    write_new(ROOT / "retained_signatures.json", signatures)
    references = before["inputs"]
    assert tuple(references) == RECIPES and len(VARIANTS) == 6
    data_hashes = None
    for recipe, row in references.items():
        directory = Path("results/runs") / row["run"]
        assert all(sha(directory / n) == h for n, h in row["files"].items())
        metrics = read(directory / "metrics.json")
        mc, tc = ModelConfig(**metrics["model"]), TrainConfig(**metrics["training"])
        mc.validate()
        tc.validate()
        assert (mc.width, mc.layers, mc.context, mc.vocab_size) == (384, 8, 128, 4096)
        assert tc.seed == 17 and tc.steps == 200 and tc.batch_size == 16
        assert tc.activation_backend == "eager" and metrics["environment"] == environment()
        if data_hashes is None:
            data_hashes = metrics["data"]["files"]
        assert metrics["data"]["files"] == data_hashes
        with zipfile.ZipFile(directory / "source.zip") as archive:
            assert all(
                hashlib.sha256(archive.read(n)).hexdigest() == h
                for n, h in metrics["provenance"]["source_files"].items()
            )
        ckpt = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert ckpt["step"] == 200 and ckpt["model_config"] == metrics["model"]
        assert ckpt["training_config"] == metrics["training"] and finite_tree(ckpt)
        del ckpt
    assert all(sha(Path("data/wikitext2_v1") / n) == h for n, h in data_hashes.items())
    prov = provenance()
    protocol = {
        "plan_sha256": PLAN_SHA,
        "provenance": prov,
        "inputs": references,
        "token_record": tensor_record(tokens()),
        "environment": environment(),
        "cells": {f"{r}_{s}": {"recipe": r, "scope": s} for r in RECIPES for s in SCOPES},
        "new_synthetic_updates": 300,
        "synthetic_training_target_exposures": 614400,
        "no_update_probe_target_exposures": 30720,
        "corpus_scoring": False,
    }
    write_new(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    write_new(
        ROOT / "preflight.json",
        {
            "status": "PASS",
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "default_signatures_exact": 12,
            "controls_verified": 5,
            "six_variants_retained": True,
            "only_existing_source_change": "src/core/transformer.py",
        },
    )
    print(
        json.dumps({"status": "PASS", "default_signatures_exact": 12, "workers_qualified": 15}),
        flush=True,
    )


def load_model(recipe, scope, protocol):
    directory = Path("results/runs") / protocol["inputs"][recipe]["run"]
    expected = protocol["inputs"][recipe]["files"]
    assert all(sha(directory / n) == h for n, h in expected.items())
    metrics = read(directory / "metrics.json")
    mc, tc = ModelConfig(**metrics["model"]), TrainConfig(**metrics["training"])
    ckpt = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Transformer(mc, 17)
    initialize_dense_width(model, tc)
    model.load_state_dict(ckpt["model"], strict=True)
    assert digest(model.state_dict()) == digest(ckpt["model"])
    model = model.cuda().train()
    configure_gate_recomputation(model, tc)
    model.set_recompute_scope(scope)
    groups = parameter_groups(model, tc)
    assert group_summary(groups) == metrics["optimizer_parameter_groups"]
    optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    optimizer.load_state_dict(ckpt["optimizer"])
    assert group_summary(optimizer.param_groups) == metrics["optimizer_parameter_groups"]
    return model, optimizer, mc, tc, metrics


def worker(cell):
    pre = read(ROOT / "preflight.json")
    assert pre["status"] == "PASS" and sha(ROOT / "protocol.json") == pre["protocol_sha256"]
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    assert environment() == protocol["environment"]
    spec = protocol["cells"][cell]
    directory = ROOT / "workers" / cell
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, optimizer, mc, tc, metrics = load_model(spec["recipe"], spec["scope"], protocol)
    stream = tokens()
    assert tensor_record(stream) == protocol["token_record"]
    config = {
        "cell": cell,
        **spec,
        "model": asdict(mc),
        "training": asdict(tc),
        "peak_lr_held_constant": tc.learning_rate,
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "input_checkpoint": protocol["inputs"][spec["recipe"]],
        "environment": environment(),
        "synthetic": True,
        "start_step": 200,
        "updates": 20,
        "corpus_scoring": False,
    }
    write_new(directory / "config.json", config)
    raw = stream[0].cuda()
    x, y = raw[:, :-1], raw[:, 1:]
    initial_diagnostics = inspect_layers(model, x, "cuda", "bf16")
    assert all(v["finite"] for v in initial_diagnostics.values())
    write_new(directory / "initial_diagnostics.json", initial_diagnostics)
    model.train()
    initial = {
        "weights": digest(model.state_dict()),
        "optimizer": digest(optimizer.state_dict()),
        "groups": group_summary(optimizer.param_groups),
    }
    cpu_rng, cuda_rng = (
        tensor_record(torch.get_rng_state()),
        tensor_record(torch.cuda.get_rng_state()),
    )
    outputs = []
    handle = model.register_forward_hook(
        lambda module, args, output: outputs.append(output.detach())
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast("cuda", "bf16"):
        loss = model.loss(x, y)
    handle.remove()
    assert len(outputs) == 1
    loss.backward()
    initial.update(
        logits=tensor_record(outputs.pop()),
        loss=loss.item(),
        gradients={n: tensor_record(p.grad) for n, p in model.named_parameters()},
        layer_gradients=gradient_stats(model),
    )
    assert finite_tree([p.grad for p in model.parameters()])
    write_new(directory / "initial_signature.json", initial)
    optimizer.zero_grad(set_to_none=True)
    del loss, raw, x, y
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    history, durations = [], []
    for step, batch in enumerate(stream):
        torch.cuda.synchronize()
        start = time.perf_counter()
        raw = batch.cuda()
        x, y = raw[:, :-1], raw[:, 1:]
        for group in optimizer.param_groups:
            group["lr"] = tc.learning_rate * group["lr_scale"]
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", "bf16"):
            loss = model.loss(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= 10:
            durations.append(elapsed)
        row = {"step": 201 + step, "loss": loss.item(), "norm_pre_clip": norm.item()}
        assert finite_tree(row)
        history.append(row)
        with (directory / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    assert finite_tree(model.state_dict()) and finite_tree(optimizer.state_dict())
    assert tensor_record(torch.get_rng_state()) == cpu_rng
    assert tensor_record(torch.cuda.get_rng_state()) == cuda_rng
    final_diagnostics = inspect_layers(model, x, "cuda", "bf16")
    assert all(v["finite"] for v in final_diagnostics.values())
    write_new(directory / "final_diagnostics.json", final_diagnostics)
    final = {"weights": digest(model.state_dict()), "optimizer": digest(optimizer.state_dict())}
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(mc),
            "training_config": asdict(tc),
            "step": 220,
            "synthetic": True,
            "execution_scope": spec["scope"],
            "protocol_sha256": sha(ROOT / "protocol.json"),
        },
        directory / "checkpoint.pt",
    )
    result = {
        "status": "PASS",
        "cell": cell,
        **spec,
        "initial_signature_sha256": sha(directory / "initial_signature.json"),
        "final": final,
        "token_record": tensor_record(stream),
        "cpu_rng": cpu_rng,
        "cuda_rng": cuda_rng,
        "rng_unchanged": True,
        "ffn_parameters": mc.unique_ffn_parameters,
        "total_parameters": mc.total_parameters,
        "ffn_reduction_percent": metrics["ffn_reduction_percent"],
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "step_seconds": durations,
        "median_step_ms": 1000 * statistics.median(durations),
        "training_tokens_per_second": 20480 / sum(durations),
        "clipped_step_fraction": sum(r["norm_pre_clip"] > 1 for r in history) / 20,
        "checkpoint_sha256": sha(directory / "checkpoint.pt"),
        "history_sha256": sha(directory / "history.jsonl"),
        "updates": 20,
        "synthetic_target_exposures": 40960,
        "all_finite": True,
        "corpus_scoring": False,
    }
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    write_new(directory / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "cell",
                    "peak_allocated_bytes",
                    "median_step_ms",
                    "clipped_step_fraction",
                )
            }
        ),
        flush=True,
    )


def finish():
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == PLAN_SHA and protocol["plan_sha256"] == PLAN_SHA
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    results, comparisons = {}, {}
    for cell in protocol["cells"]:
        directory = ROOT / "workers" / cell
        result = read(directory / "result.json")
        assert result["status"] == "PASS" and result["updates"] == 20 and result["all_finite"]
        process = read(ROOT / "processes" / f"worker_{cell}.json")
        assert (
            process["status"] == "PASS"
            and process["returncode"] == 0
            and process["source_unchanged"]
        )
        assert sha(directory / "checkpoint.pt") == result["checkpoint_sha256"]
        assert sha(directory / "history.jsonl") == result["history_sha256"]
        assert sha(directory / "initial_signature.json") == result["initial_signature_sha256"]
        ckpt = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert ckpt["step"] == 220 and ckpt["synthetic"] and finite_tree(ckpt)
        assert digest(ckpt["model"]) == result["final"]["weights"]
        assert digest(ckpt["optimizer"]) == result["final"]["optimizer"]
        results[cell] = result
        del ckpt
    for recipe in RECIPES:
        native = results[f"{recipe}_none"]
        for scope in SCOPES[1:]:
            cell = f"{recipe}_{scope}"
            other = results[cell]
            equality = {
                "initial_signature": read(
                    ROOT / "workers" / f"{recipe}_none" / "initial_signature.json"
                )
                == read(ROOT / "workers" / cell / "initial_signature.json"),
                "history": native["history_sha256"] == other["history_sha256"],
                "final_weights": native["final"]["weights"] == other["final"]["weights"],
                "final_optimizer": native["final"]["optimizer"] == other["final"]["optimizer"],
                "token_and_rng": all(
                    native[k] == other[k] for k in ("token_record", "cpu_rng", "cuda_rng")
                ),
            }
            comparisons[cell] = equality
    exact = all(all(row.values()) for row in comparisons.values())
    native_rational = results["rational_none"]
    gates = {
        scope: resource_gates(
            results[f"rational_{scope}"],
            native_rational,
            {r: results[f"{r}_{scope}"] for r in ("full_swiglu", "full_gelu")},
        )
        for scope in SCOPES[1:]
    }
    eligible = [scope for scope in SCOPES[1:] if exact and all(gates[scope].values())]
    for recipe, row in protocol["inputs"].items():
        assert all(sha(Path("results/runs") / row["run"] / n) == h for n, h in row["files"].items())
    result = {
        "status": "complete",
        "all_updates_exact_across_scopes": exact,
        "exact_comparisons": comparisons,
        "resource_gates": gates,
        "selected_scope": eligible[0] if eligible else None,
        "earns_full_native_training_repeat": bool(eligible),
        "results": results,
        "plan_sha256": PLAN_SHA,
        "synthetic_updates": 300,
        "synthetic_target_exposures": 614400,
        "no_update_probe_target_exposures": 30720,
        "corpus_scoring": False,
        "research_goal_achieved": False,
    }
    write_new(ROOT / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "all_updates_exact_across_scopes",
                    "resource_gates",
                    "selected_scope",
                    "earns_full_native_training_repeat",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "worker", "finish"))
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.cell)
    else:
        {"preflight": preflight, "finish": finish}[args.command]()
