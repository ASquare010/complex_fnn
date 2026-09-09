"""H054 fixed-batch complete-Transformer integration and memory qualification."""

import argparse
import ast
import hashlib
import json
import math
import zipfile
from pathlib import Path

PLAN = Path("research/overcomplete_integration_plan.md")
ROOT = Path("results/overcomplete_integration_v1")
CANDIDATE = "overcomplete_headwise_swiglu"
ORDER = ("swiglu", CANDIDATE, "gelu")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration(variant):
    from src.core.config import ModelConfig, TrainConfig

    if variant not in ORDER:
        raise ValueError("Variant outside the frozen integration cohort")
    mc = ModelConfig(
        variant=variant,
        width=384,
        layers=8,
        heads=6,
        context=128,
        vocab_size=4096,
        groups=8,
        hidden=200 if variant == CANDIDATE else 0,
    )
    tc = TrainConfig(
        steps=20,
        batch_size=16,
        learning_rate=0.0012,
        weight_decay=0.1,
        seed=17,
        eval_batches=1,
        log_every=1,
        precision="bf16",
        device="cuda",
        ffn_lr_mode="uniform",
        ffn_decay_mode="parameter",
    )
    return mc, tc


def preflight():
    from dataclasses import asdict

    import torch

    from src.core.benchmark import forward_flops
    from src.core.config import VARIANTS, ModelConfig
    from src.core.data import TokenData
    from src.core.optimization import group_summary, parameter_groups
    from src.core.reproducibility import environment, provenance
    from src.core.transformer import Transformer
    from src.multihead_ffn.overcomplete import OvercompleteHeadwiseFFN

    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False)
    previous = read("results/verification/overcomplete_final_v1.json")
    assert (
        previous["status"] == "PASS"
        and previous["earns_separate_integration_and_memory_qualification"]
    )
    assert sha("results/overcomplete_qualification_v1/cpu.json") == previous["cpu_result_sha256"]
    assert sha("results/overcomplete_qualification_v1/cuda.json") == previous["cuda_result_sha256"]
    snap = Path("results/verification/overcomplete_before_v1.pt")
    meta = read(snap.with_suffix(".json"))
    assert sha(snap) == meta["sha256"] and meta["variants"] == 31
    changed = {n for n, h in meta["provenance"]["source_files"].items() if sha(n) != h}
    assert changed == {
        "src/core/config.py",
        "src/core/transformer.py",
        "src/core/benchmark.py",
        "src/multihead_ffn/overcomplete.py",
        "tests/test_baselines.py",
    }, changed
    before = torch.load(snap, map_location="cpu", weights_only=True)
    x = torch.arange(16).reshape(2, 8)
    for variant, row in before.items():
        config = ModelConfig(**row["config"])
        m = Transformer(config, 17)
        loss = m.loss(x, x + 1)
        loss.backward()
        assert torch.equal(m(x), row["logits"]) and torch.equal(loss, row["loss"])
        assert all(torch.equal(p, row["state"][n]) for n, p in m.state_dict().items())
        assert all(torch.equal(p.grad, row["gradients"][n]) for n, p in m.named_parameters())
        assert forward_flops(config) == row["flops"]
    assert set(VARIANTS) - set(before) == {CANDIDATE}
    # The qualified numerical module changes only its class docstring.
    with zipfile.ZipFile("results/overcomplete_qualification_v1/source.zip") as z:
        old_ast = ast.parse(z.read("src/multihead_ffn/overcomplete.py").decode())
    current_ast = ast.parse(Path("src/multihead_ffn/overcomplete.py").read_text())
    for tree in (old_ast, current_ast):
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    assert ast.dump(old_ast) == ast.dump(current_ast)
    c, tc = configuration(CANDIDATE)
    m = Transformer(c, 17)
    full = Transformer(configuration("swiglu")[0], 17)
    common = dict(full.named_parameters())
    assert all(torch.equal(p, common[n]) for n, p in m.named_parameters() if ".ffn." not in n)
    for i, block in enumerate(m.blocks):
        independent = OvercompleteHeadwiseFFN(384, 512, 16, 8, 200)
        independent.initialize(17, f"blocks.{i}.ffn", 0.25)
        assert all(
            torch.equal(p, independent.state_dict()[n]) for n, p in block.ffn.state_dict().items()
        )
    assert sum(p.numel() for p in m.parameters()) == c.total_parameters == 9099648
    assert c.unique_ffn_parameters == 2801664
    assert forward_flops(c)["ffn_matrix_forward_flops_per_token"] == 5603328
    groups = group_summary(parameter_groups(m, tc))
    assert all(g["lr_scale"] == 1 and g["weight_decay"] in (0.0, 0.1) for g in groups)
    data = TokenData(Path("data/wikitext2_v1"), "cpu", 10017)
    data_hashes = read("results/duration_decay_v1/preflight.json")["data_hashes"]
    assert data.manifest["files"] == data_hashes
    assert all(sha(Path("data/wikitext2_v1") / n) == h for n, h in data_hashes.items())
    initial = {
        "status": "PASS",
        "old_variants_exact": 31,
        "snapshot_sha256": sha(snap),
        "qualified_module_numerical_ast_exact": True,
        "all_layers_match_standalone": True,
        "common_non_ffn_parameters_exact": True,
        "qualified_changed_files": sorted(changed),
        "actual_candidate_optimizer_groups": groups,
        "data_hashes": data_hashes,
        "environment": environment(),
    }
    write(ROOT / "preflight.json", initial)
    protocol = {
        "plan_sha256": sha(PLAN),
        "preflight_sha256": sha(ROOT / "preflight.json"),
        "provenance": provenance(),
        "order": ORDER,
        "training_token_exposures": 122880,
        "validation_targets_scored": 0,
        "configurations": {
            v: {"model": asdict(configuration(v)[0]), "training": asdict(configuration(v)[1])}
            for v in ORDER
        },
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
                "old_variants_exact": 31,
                "candidate_total_parameters": 9099648,
            }
        ),
        flush=True,
    )


def gpu(variant, *, root=ROOT, plan=PLAN, config_factory=configuration, execution_check=None):
    import gc
    from dataclasses import asdict

    import torch

    from src.core.benchmark import autocast, forward_flops
    from src.core.data import TokenData
    from src.core.diagnostics import gradient_stats, inspect_layers
    from src.core.optimization import group_summary, parameter_groups
    from src.core.reproducibility import environment, provenance
    from src.core.trainer import configure_gate_recomputation
    from src.core.transformer import Transformer

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = read(root / "protocol.json")
    assert (
        sha(plan) == protocol["plan_sha256"]
        and sha(root / "preflight.json") == protocol["preflight_sha256"]
    )
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    c, tc = config_factory(variant)
    assert {"model": asdict(c), "training": asdict(tc)} == protocol["configurations"][variant]
    output = root / variant
    output.mkdir(exist_ok=False)
    model = Transformer(c, 17).cuda()
    configure_gate_recomputation(model, tc)
    data = TokenData(Path("data/wikitext2_v1"), "cuda", 10017)
    x, y = data.batch(16, 128)
    batch_hash = hashlib.sha256(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest()
    assert batch_hash == "258e6c6132c5ac9d202dd965ee64a530055aa414fd62b396b79c4d1be392362e"
    assert data.manifest["files"] == read(root / "preflight.json")["data_hashes"]
    with torch.no_grad():
        expected = model(x).float().cpu()
        with autocast("cuda", "bf16"):
            actual = model(x).float().cpu()
    relative = (actual - expected).norm().item() / expected.norm().item()
    assert math.isfinite(relative) and relative <= 0.05
    del actual, expected
    initial = inspect_layers(model, x, "cuda", "bf16")
    assert all(r["finite"] for r in initial.values())
    write(output / "initial_diagnostics.json", initial)
    extra_check = execution_check(model, x, y, tc) if execution_check is not None else None
    groups = parameter_groups(model, tc)
    optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    history = []
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", "bf16"):
            loss = model.loss(x, y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 0:
            assert all(p.grad.norm() > 0 for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        row = {
            "step": step + 1,
            "fixed_training_batch_loss": loss.item(),
            "gradient_norm_pre_clip": norm.item(),
            "layer_gradient_norms_post_clip": gradient_stats(model),
        }
        assert math.isfinite(row["fixed_training_batch_loss"])
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        history.append(row)
        with (output / "history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    final = inspect_layers(model, x, "cuda", "bf16")
    assert all(r["finite"] for r in final.values())
    write(output / "final_diagnostics.json", final)
    assert all(
        torch.isfinite(t).all()
        for s in optimizer.state.values()
        for t in s.values()
        if isinstance(t, torch.Tensor)
    )
    optimizer_bytes = sum(
        t.numel() * t.element_size()
        for s in optimizer.state.values()
        for t in s.values()
        if isinstance(t, torch.Tensor)
    )
    checkpoint = {
        "model": {n: t.detach().cpu() for n, t in model.state_dict().items()},
        "model_config": asdict(c),
        "training_config": asdict(tc),
        "step": 20,
        "sampling_rng": data.generator.get_state(),
        "scope": "Twenty constant-rate updates on one repeated training batch; no validation scoring",
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    with torch.no_grad(), autocast("cuda", "bf16"):
        expected = model(x).cpu()
    restored = Transformer(c, 29).cuda()
    configure_gate_recomputation(restored, tc)
    restored.load_state_dict(
        torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    )
    with torch.no_grad(), autocast("cuda", "bf16"):
        assert torch.equal(expected, restored(x).cpu())
    record = {
        "status": "PASS",
        "variant": variant,
        "model": asdict(c),
        "training": asdict(tc),
        "precision_relative_l2": relative,
        "execution_check": extra_check,
        "baseline_allocated_bytes": baseline,
        "peak_training_allocated_bytes": peak,
        "optimizer_state_bytes": optimizer_bytes,
        "initial_fixed_batch_loss": history[0]["fixed_training_batch_loss"],
        "last_preupdate_fixed_batch_loss": history[-1]["fixed_training_batch_loss"],
        "ffn_parameters": c.unique_ffn_parameters,
        "total_parameters": c.total_parameters,
        **forward_flops(c),
        "optimizer_parameter_groups": group_summary(groups),
        "clipped_step_fraction": sum(r["gradient_norm_pre_clip"] > 1 for r in history) / 20,
        "all_weights_gradients_moments_and_layer_diagnostics_finite": True,
        "every_parameter_first_gradient_nonzero": True,
        "checkpoint_logits_roundtrip_exact": True,
        "batch_sha256": batch_hash,
        "sampling_rng_sha256": hashlib.sha256(
            data.generator.get_state().numpy().tobytes()
        ).hexdigest(),
        "training_token_exposures": 40960,
        "unique_training_targets": 2048,
        "validation_targets_scored": 0,
        "data_hashes": data.manifest["files"],
        "plan_sha256": sha(plan),
        "environment": environment(),
        "provenance": provenance(),
        "files": {
            n: sha(output / n)
            for n in (
                "checkpoint.pt",
                "history.jsonl",
                "initial_diagnostics.json",
                "final_diagnostics.json",
            )
        },
        "scope": "Fixed-batch pipeline/memory qualification, not language quality, convergence or a timing comparison",
    }
    write(output / "result.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "variant",
                    "precision_relative_l2",
                    "peak_training_allocated_bytes",
                    "clipped_step_fraction",
                    "last_preupdate_fixed_batch_loss",
                )
            }
        ),
        flush=True,
    )


def finish():
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    rows = {}
    for variant in ORDER:
        p = ROOT / variant
        r = read(p / "result.json")
        assert r["status"] == "PASS" and r["plan_sha256"] == protocol["plan_sha256"]
        assert all(sha(p / n) == h for n, h in r["files"].items())
        assert r["provenance"]["source_files"] == protocol["provenance"]["source_files"]
        assert r["training_token_exposures"] == 40960 and r["validation_targets_scored"] == 0
        rows[variant] = r
    assert len({r["sampling_rng_sha256"] for r in rows.values()}) == 1
    assert all(
        r["data_hashes"] == rows["swiglu"]["data_hashes"]
        and r["environment"] == rows["swiglu"]["environment"]
        for r in rows.values()
    )
    gates = {
        "at_least_70_percent_fewer_ffn_weights": rows[CANDIDATE]["ffn_parameters"]
        <= 0.3 * rows["swiglu"]["ffn_parameters"]
    }
    for variant in ("swiglu", "gelu"):
        gates["memory_within_ten_percent_" + variant] = (
            rows[CANDIDATE]["peak_training_allocated_bytes"]
            <= 1.1 * rows[variant]["peak_training_allocated_bytes"]
        )
    result = {
        "status": "complete",
        "plan_sha256": sha(PLAN),
        "old_variants_exact": 31,
        "gates": gates,
        "earns_separate_quality_screen": all(gates.values()),
        "quality_measured": False,
        "training_token_exposures": 122880,
        "validation_targets_scored": 0,
        "numerical_failures": 0,
        "trials": {
            v: {
                "result_sha256": sha(ROOT / v / "result.json"),
                "peak_mib": r["peak_training_allocated_bytes"] / 2**20,
                "precision_relative_l2": r["precision_relative_l2"],
                "clipped_step_fraction": r["clipped_step_fraction"],
            }
            for v, r in rows.items()
        },
    }
    write(ROOT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "gpu", "finish"))
    parser.add_argument("--variant", choices=ORDER)
    args = parser.parse_args()
    if args.command == "gpu":
        if args.variant is None:
            parser.error("GPU qualification requires a variant")
        gpu(args.variant)
    else:
        {"preflight": preflight, "finish": finish}[args.command]()
