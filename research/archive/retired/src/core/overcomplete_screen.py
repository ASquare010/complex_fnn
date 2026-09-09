"""H056 bounded quality screen; computation uses the existing frozen worker."""

import argparse
import ast
import hashlib
import json
import math
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path("results/overcomplete_screen_v1")
PLAN = Path("research/overcomplete_screen_plan.md")
CACHE = Path("data/wikitext2_v1")
RATES = (0.0003, 0.0006, 0.0012)
RECIPE = "overcomplete_headwise"
CONTROLS = ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle", "headwise")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration(rate):
    from src.core.config import ModelConfig, TrainConfig

    if rate not in RATES:
        raise ValueError("Rate is outside the frozen screen")
    raw = read(f"configs/wikitext2_{RECIPE}_screen.json")
    return ModelConfig(**raw["model"]), replace(TrainConfig(**raw["training"]), learning_rate=rate)


def output_path(rate):
    return Path(f"results/runs/wikitext2_{RECIPE}_lr{round(rate * 1e6)}_s17_200")


def decision(candidate, references):
    from src.core.wikitext_screen import promotion

    gates = promotion({**references, "blockshuffle": candidate})
    return {
        "gates": gates,
        "earns_separate_longer_comparison": all(gates.values()),
        "diagnostic_point_two_percent_narrow_margin": (
            candidate["validation_loss"]
            <= 0.998 * references["calibrated_narrow"]["validation_loss"]
        ),
    }


def archive_check(path, metrics, *, compare_old=False):
    with zipfile.ZipFile(path / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in metrics["provenance"]["source_files"].items()
        )
        if not compare_old:
            return
        for n in (
            "src/core/data.py",
            "src/core/optimization.py",
            "src/dense_ffn/__init__.py",
            "src/blockshuffle_ffn/__init__.py",
            "src/grouped_ffn/__init__.py",
        ):
            assert archive.read(n) == Path(n).read_bytes(), n
        # Compare actual training updates, independently of changed setup branches.
        trees = [
            ast.parse(archive.read("src/core/trainer.py")),
            ast.parse(Path("src/core/trainer.py").read_bytes()),
        ]
        loops = [
            [
                ast.dump(n)
                for n in ast.walk(tree)
                if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name)
                and n.target.id == "step"
            ]
            for tree in trees
        ]
        assert len(loops[0]) == 1 and loops[0] == loops[1]
        functions = []
        for source in (
            archive.read("src/core/benchmark.py"),
            Path("src/core/benchmark.py").read_bytes(),
        ):
            functions.append(
                {
                    n.name: ast.dump(n)
                    for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name != "forward_flops"
                }
            )
        assert all(functions[1][n] == v for n, v in functions[0].items())
        classes = []
        for source in (
            archive.read("src/core/transformer.py"),
            Path("src/core/transformer.py").read_bytes(),
        ):
            classes.append(
                {n.name: n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)}
            )
        for name in ("RMSNorm", "Attention"):
            assert ast.dump(classes[0][name]) == ast.dump(classes[1][name])
        for name in ("Block", "Transformer"):
            methods = [
                {
                    n.name: ast.dump(n)
                    for n in side[name].body
                    if isinstance(n, ast.FunctionDef) and n.name in ("forward", "loss")
                }
                for side in classes
            ]
            assert methods[0] == methods[1]


def checkpoint_check(path, metrics, rng_hash):
    import torch

    ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 200 and ckpt["model_config"] == metrics["model"]
    assert ckpt["training_config"] == metrics["training"]
    assert hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest() == rng_hash
    assert sum(t.numel() for t in ckpt["model"].values()) == metrics["total_parameters"]
    assert all(torch.isfinite(t).all() for t in ckpt["model"].values())
    assert all(
        torch.isfinite(t).all()
        for state in ckpt["optimizer"]["state"].values()
        for t in state.values()
        if isinstance(t, torch.Tensor)
    )


def preflight():
    import torch

    from src.core.config import ModelConfig, TrainConfig
    from src.core.data import TokenData, load_manifest
    from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
    from src.core.reproducibility import environment, provenance
    from src.core.transformer import Transformer

    assert not ROOT.exists()
    torch.set_num_threads(4)
    tested = read("results/verification/overcomplete_recompute_tests_v1.json")
    assert tested["returncode"] == 0 and tested["source_unchanged"]
    assert all(sha(n) == h for n, h in tested["source_files"].items())
    final = read("results/verification/overcomplete_recompute_final_v1.json")
    assert final["status"] == "PASS" and final["earns_separate_quality_screen"]
    assert sha("results/overcomplete_recompute_v1/result.json") == final["result_sha256"]
    assert sha("results/overcomplete_recompute_v1/source.zip") == final["source_archive_sha256"]
    manifest = load_manifest(CACHE)
    prior = read("results/headwise_screen_v1/preflight.json")
    assert manifest["files"] == prior["data_hashes"]
    assert all(sha(CACHE / n) == h for n, h in manifest["files"].items())
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    rows = [*prior["controls"], *read("results/headwise_screen_v1/result.json")["trials"]]
    groups, common = {}, None
    for row in rows:
        path = Path("results/runs") / row["run"]
        assert sha(path / "metrics.json") == row["metrics_sha256"]
        assert sha(path / "checkpoint.pt") == row["checkpoint_sha256"]
        if "source_archive_sha256" in row:
            assert sha(path / "source.zip") == row["source_archive_sha256"]
        row["source_archive_sha256"] = sha(path / "source.zip")
        m = read(path / "metrics.json")
        mc, tc = ModelConfig(**m["model"]), TrainConfig(**m["training"])
        raw = read(f"configs/wikitext2_{row['recipe']}_screen.json")
        assert mc == ModelConfig(**raw["model"])
        assert tc == replace(TrainConfig(**raw["training"]), learning_rate=row["rate"])
        assert m["data"]["files"] == manifest["files"] and m["environment"] == environment()
        assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
        archive_check(path, m, compare_old=True)
        checkpoint_check(path, m, prior["sampling_rng_sha256"])
        model = Transformer(mc, 17)
        initialize_dense_width(model, tc)
        # Later architecture hooks changed this file; qualify retained behavior directly.
        from src.core.diagnostics import gradient_stats, inspect_layers

        legacy = {}
        with zipfile.ZipFile(path / "source.zip") as archive:
            exec(
                compile(archive.read("src/core/diagnostics.py"), "retained_diagnostics", "exec"),
                legacy,
            )
        x, y = batches[0]
        before_diagnostics = {n: p.detach().clone() for n, p in model.named_parameters()}
        rng_before = torch.get_rng_state().clone()
        model.loss(x[:2, :16], y[:2, :16]).backward()
        assert legacy["gradient_stats"](model) == gradient_stats(model)
        assert legacy["inspect_layers"](model, x[:2, :16], "cpu", "fp32") == inspect_layers(
            model, x[:2, :16], "cpu", "fp32"
        )
        assert torch.equal(rng_before, torch.get_rng_state())
        assert all(torch.equal(p, before_diagnostics[n]) for n, p in model.named_parameters())
        model.zero_grad(set_to_none=True)
        actual = group_summary(parameter_groups(model, tc))
        assert actual == m["optimizer_parameter_groups"]
        assert actual == group_summary(parameter_groups(model, replace(tc, ffn_lr_mode="fan_in")))
        groups[row["run"]] = actual
        shared = {n: p.detach().clone() for n, p in model.named_parameters() if ".ffn." not in n}
        if common is None:
            common = shared
        assert all(torch.equal(t, common[n]) for n, t in shared.items())
        del model
    mc, tc = configuration(RATES[0])
    mc.validate()
    tc.validate()
    model = Transformer(mc, 17)
    assert mc.unique_ffn_parameters == 2801664 and mc.total_parameters == 9099648
    assert all(torch.equal(p, common[n]) for n, p in model.named_parameters() if ".ffn." not in n)
    actual = parameter_groups(model, tc)
    scales = {id(p): g["lr_scale"] for g in actual for p in g["params"]}
    named_scales = {n: scales[id(p)] for n, p in model.named_parameters()}
    for n, scale in named_scales.items():
        expected = (
            32 / 3
            if ".output_projection.second." in n
            else 8.0
            if ".input_projection." in n or ".output_projection." in n
            else 1.0
        )
        assert scale == expected, (n, scale, expected)
    configuration_records = {}
    cells = {}
    for rate in RATES:
        c, t = configuration(rate)
        p = output_path(rate)
        assert not p.exists()
        configuration_records[p.name] = {"model": asdict(c), "training": asdict(t)}
        cells[p.name] = p.as_posix()
    prov = provenance()
    record = {
        "status": "PASS",
        "h055_sources_unchanged": True,
        "retained_exact_variant_count": final["old_variants_exact"],
        "controls": rows,
        "data_hashes": manifest["files"],
        "validation_targets": 322688,
        "validation_stream_exact": True,
        "sampling_rng_sha256": prior["sampling_rng_sha256"],
        "environment": environment(),
        "common_non_ffn_initial_weights_exact": True,
        "control_diagnostic_values_exact_on_cpu_probe": True,
        "control_optimizer_groups": groups,
        "candidate_optimizer_groups": group_summary(actual),
        "candidate_lr_scales": named_scales,
    }
    protocol = {
        "plan_sha256": sha(PLAN),
        "configurations": configuration_records,
        "provenance": prov,
        "new_training_tokens": 1228800,
        "rates": RATES,
    }
    ROOT.mkdir()
    write(ROOT / "preflight.json", record)
    write(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for n in prov["source_files"]:
            archive.write(n, n)
        archive.write(PLAN, PLAN.as_posix())
    write(
        ROOT / "worker_qualification.json",
        {
            "status": "PASS",
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "plan_path": PLAN.as_posix(),
            "plan_sha256": sha(PLAN),
            "critical_source_hashes": prov["source_files"],
            "cells": cells,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "retained_controls_verified": len(rows),
                "candidate_optimizer_groups": record["candidate_optimizer_groups"],
            }
        ),
        flush=True,
    )


def finish():
    from src.core.config import ModelConfig, TrainConfig
    from src.core.wikitext_screen import select_best

    assert not (ROOT / "result.json").exists()
    pre = read(ROOT / "preflight.json")
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    assert all(sha(CACHE / n) == h for n, h in pre["data_hashes"].items())
    trials, metrics = [], []
    for rate in RATES:
        path = output_path(rate)
        assert not (path / "failure.json").exists()
        m = read(path / "metrics.json")
        c, t = configuration(rate)
        assert ModelConfig(**m["model"]) == c and TrainConfig(**m["training"]) == t
        assert m["provenance"]["source_files"] == protocol["provenance"]["source_files"]
        assert m["data"]["files"] == pre["data_hashes"] and m["environment"] == pre["environment"]
        assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
        assert m["optimizer_parameter_groups"] == pre["candidate_optimizer_groups"]
        assert m["ffn_parameters"] == 2801664 and m["total_parameters"] == 9099648
        assert m["status"] == "SCREENED" and math.isfinite(m["validation_loss"])
        for name in ("initial_diagnostics.json", "final_diagnostics.json"):
            assert all(v["finite"] for v in read(path / name).values())
        history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
        assert [r["step"] for r in history] == [1, 50, 100, 150, 200]
        assert history[-1]["validation_loss"] == m["validation_loss"]
        assert all(
            math.isfinite(r[k])
            for r in history
            for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
        )
        checkpoint_check(path, m, pre["sampling_rng_sha256"])
        archive_check(path, m)
        metrics.append(m)
        trials.append(
            {
                "recipe": RECIPE,
                "rate": rate,
                "run": path.name,
                "nll": m["validation_loss"],
                "peak_mib": m["peak_allocated_vram_bytes"] / 2**20,
                "clipped_step_fraction": m["clipped_step_fraction"],
                "files": {p.name: sha(p) for p in path.iterdir() if p.is_file()},
            }
        )
    assert len({m["initial_validation_loss"] for m in metrics}) == 1
    controls = {r: [] for r in CONTROLS}
    for row in pre["controls"]:
        path = Path("results/runs") / row["run"]
        assert sha(path / "metrics.json") == row["metrics_sha256"]
        assert sha(path / "checkpoint.pt") == row["checkpoint_sha256"]
        assert sha(path / "source.zip") == row["source_archive_sha256"]
        controls[row["recipe"]].append(read(path / "metrics.json"))
    refs = {r: select_best(rows) for r, rows in controls.items()}
    selected = select_best(metrics)
    outcome = decision(selected, refs)
    result = {
        "status": "complete",
        "trials": trials,
        "selected": selected["run"],
        "references": {r: m["run"] for r, m in refs.items()},
        **outcome,
        "selected_relative_nll_percent": {
            r: 100 * (selected["validation_loss"] / m["validation_loss"] - 1)
            for r, m in refs.items()
        },
        "same_rate_relative_nll_percent": {
            str(round(m["training"]["learning_rate"] * 1e6)): {
                r: 100
                * (
                    m["validation_loss"]
                    / next(
                        c["validation_loss"]
                        for c in rows
                        if c["training"]["learning_rate"] == m["training"]["learning_rate"]
                    )
                    - 1
                )
                for r, rows in controls.items()
            }
            for m in metrics
        },
        "grid_boundary_winner": selected["training"]["learning_rate"] in (RATES[0], RATES[-1]),
        "all_diagnostics_finite": True,
        "all_final_sampling_states_exact": True,
        "plan_sha256": sha(PLAN),
        "new_training_tokens": 1228800,
        "validation_targets_per_evaluation": 322688,
        "official_test_scored": False,
        "research_target_passes": False,
    }
    write(ROOT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "finish"))
    args = parser.parse_args()
    {"preflight": preflight, "finish": finish}[args.command]()
