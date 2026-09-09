"""H061 additive language screen using preserved controls and the frozen trainer."""

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path("results/additive_screen_v2")
PLAN = Path("research/additive_block_lowrank_screen_plan.md")
CACHE = Path("data/wikitext2_v1")
RATES = (0.0003, 0.0006, 0.0012)
RECIPE = "additive_block_lowrank"
CONTROLS = ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")


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


def select_best(rows):
    if not rows or any(not math.isfinite(row["validation_loss"]) for row in rows):
        raise ValueError("Selection requires finite completed trials")
    return min(rows, key=lambda row: (row["validation_loss"], row["training"]["learning_rate"]))


def promotion(selected):
    candidate = selected["blockshuffle"]
    loss = candidate["validation_loss"]
    gates = {
        "at_least_70_percent_fewer_ffn_weights": candidate["ffn_reduction_percent"] >= 70,
        "beats_calibrated_narrow": loss < selected["calibrated_narrow"]["validation_loss"],
    }
    for recipe in ("full_swiglu", "full_gelu"):
        gates[f"within_one_percent_{recipe}"] = loss <= 1.01 * selected[recipe]["validation_loss"]
        gates[f"memory_within_ten_percent_{recipe}"] = (
            candidate["peak_allocated_vram_bytes"]
            <= 1.1 * selected[recipe]["peak_allocated_vram_bytes"]
        )
    return gates


def nodes(source, kind):
    return {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, kind)}


def run_reference_probe(request, output, cwd, label):
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(Path("results/verification/additive_control_probe_v1.py").resolve()),
        str(request.resolve()),
        str(output.resolve()),
    ]
    log_path = ROOT / f"process_reference_{label}.log"
    record_path = ROOT / f"process_reference_{label}.json"
    assert len({output.resolve(), log_path.resolve(), record_path.resolve()}) == 3
    assert not log_path.exists() and not record_path.exists()
    write(record_path, {"status": "RUNNING", "command": command, "cwd": str(cwd)})
    start = time.perf_counter()
    with log_path.open("w") as handle:
        process = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    write(
        record_path,
        {
            "status": "PASS" if process.returncode == 0 else "FAIL",
            "returncode": process.returncode,
            "seconds": time.perf_counter() - start,
            "command": command,
            "cwd": str(cwd),
            "log_sha256": sha(log_path),
        },
    )
    print(json.dumps({"reference_probe": label, "returncode": process.returncode}), flush=True)
    assert process.returncode == 0, log_path.read_text()


def decision(candidate, references):
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
        assert archive.read("src/core/data.py") == Path("src/core/data.py").read_bytes()
        for old_path, new_path, names in (
            ("src/dense_ffn/__init__.py", "src/dense_ffn/__init__.py", ("DenseFFN",)),
            ("src/grouped_ffn/__init__.py", "src/core/structured_linear.py", ("GroupedLinear",)),
            (
                "src/blockshuffle_ffn/__init__.py",
                "src/blockshuffle_ffn/__init__.py",
                ("BlockShuffleLinear", "BlockShuffleFFN", "RecomputedSwiGLU"),
            ),
        ):
            left = nodes(archive.read(old_path), ast.ClassDef)
            right = nodes(Path(new_path).read_bytes(), ast.ClassDef)
            assert all(left[name] == right[name] for name in names), old_path
        left = nodes(archive.read("src/core/trainer.py"), ast.FunctionDef)
        right = nodes(Path("src/core/trainer.py").read_bytes(), ast.FunctionDef)
        assert all(left[name] == right[name] for name in ("learning_rate", "resolve_precision"))
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

    assert sha(PLAN) == "86d3833d3d7fc99e976baf227893193df6715668d7e5ea3a4b2edc5664904763"
    assert not ROOT.exists()
    torch.set_num_threads(4)
    final = read("results/verification/additive_qualification_final_v1.json")
    assert final["status"] == "PASS" and final["quality_screen_earned"]
    assert sha("results/additive_qualification_v1/result.json") == final["result_sha256"]
    assert sha("results/additive_qualification_v1/source.zip") == final["source_archive_sha256"]
    tested = read("results/additive_qualification_v1/process_integration_tests.json")
    assert tested["returncode"] == 0 and tested["source_unchanged"]
    assert all(sha(name) == digest for name, digest in tested["source_hashes"].items())
    prior = read("results/overcomplete_screen_v1/preflight.json")
    rows = [row for row in prior["controls"] if row["recipe"] in CONTROLS]
    assert len(rows) == 12
    assert {(row["recipe"], row["rate"]) for row in rows} == {
        (recipe, rate) for recipe in CONTROLS for rate in RATES
    }
    manifest = load_manifest(CACHE)
    assert manifest["files"] == prior["data_hashes"]
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    ROOT.mkdir()
    groups, common, snapshots = {}, None, {}
    for row in rows:
        path = Path("results/runs") / row["run"]
        assert sha(path / "metrics.json") == row["metrics_sha256"]
        assert sha(path / "checkpoint.pt") == row["checkpoint_sha256"]
        assert sha(path / "source.zip") == row["source_archive_sha256"]
        metrics = read(path / "metrics.json")
        mc, tc = ModelConfig(**metrics["model"]), TrainConfig(**metrics["training"])
        mc.validate()
        tc.validate()
        assert (mc.width, mc.layers, mc.heads, mc.context, mc.vocab_size) == (384, 8, 6, 128, 4096)
        assert (
            mc.ffn_width
            == {
                "full_swiglu": 1024,
                "full_gelu": 1536,
                "calibrated_narrow": 304,
                "blockshuffle": 2048,
            }[row["recipe"]]
        )
        assert (tc.steps, tc.batch_size, tc.eval_batches, tc.log_every, tc.seed, tc.precision) == (
            200,
            16,
            158,
            50,
            17,
            "bf16",
        )
        assert tc.learning_rate == row["rate"]
        assert (
            metrics["data"]["files"] == manifest["files"]
            and metrics["environment"] == environment()
        )
        assert metrics["training_tokens"] == 409600 and metrics["validation_tokens"] == 322688
        archive_check(path, metrics, compare_old=True)
        checkpoint_check(path, metrics, prior["sampling_rng_sha256"])
        model = Transformer(mc, 17)
        initialize_dense_width(model, tc)
        actual = group_summary(parameter_groups(model, tc))
        assert actual == metrics["optimizer_parameter_groups"]
        groups[row["run"]] = actual
        shared = {
            name: sha_tensor(p) for name, p in model.named_parameters() if ".ffn." not in name
        }
        if common is None:
            common = shared
        assert shared == common
        key = metrics["provenance"]["source_hash"]
        entry = snapshots.setdefault(key, {"path": path, "metrics": metrics, "controls": []})
        entry["controls"].append(
            {"run": row["run"], "model": metrics["model"], "training": metrics["training"]}
        )
        del model

    # Execute original src trees in fresh interpreters, never importing them into the live core.
    compatibility = {}
    for index, (key, snapshot) in enumerate(snapshots.items()):
        stage = (ROOT / "legacy_sources" / key).resolve()
        stage.mkdir(parents=True)
        with zipfile.ZipFile(snapshot["path"] / "source.zip") as archive:
            for name in snapshot["metrics"]["provenance"]["source_files"]:
                assert (stage / name).resolve().is_relative_to(stage)
                archive.extract(name, stage)
        request = ROOT / f"reference_request_{index}.json"
        write(
            request,
            {"train_path": str((CACHE / "train.npy").resolve()), "controls": snapshot["controls"]},
        )
        old_output, new_output = (
            ROOT / f"reference_old_{index}.json",
            ROOT / f"reference_current_{index}.json",
        )
        run_reference_probe(request, old_output, stage, f"old_{index}")
        run_reference_probe(request, new_output, Path.cwd(), f"current_{index}")
        assert read(old_output) == read(new_output), f"Full-size compatibility mismatch: {key}"
        compatibility[key] = {
            "controls": [cell["run"] for cell in snapshot["controls"]],
            "original_result_sha256": sha(old_output),
            "current_result_sha256": sha(new_output),
        }

    gpu_data = TokenData(CACHE, "cuda", 10017)
    for _ in range(200):
        gpu_data.batch(16, 128)
    sampler = sha_tensor(gpu_data.generator.get_state())
    assert sampler == prior["sampling_rng_sha256"]
    del gpu_data
    torch.cuda.empty_cache()
    mc, tc = configuration(RATES[0])
    model = Transformer(mc, 17)
    assert mc.unique_ffn_parameters == 2801664 and mc.total_parameters == 9099648
    assert {
        name: sha_tensor(p) for name, p in model.named_parameters() if ".ffn." not in name
    } == common
    actual = parameter_groups(model, tc)
    scales = {id(p): group["lr_scale"] for group in actual for p in group["params"]}
    named_scales = {name: scales[id(p)] for name, p in model.named_parameters()}
    assert (
        len(scales)
        == sum(len(group["params"]) for group in actual)
        == len(list(model.parameters()))
    )
    configuration_records, cells = {}, {}
    for rate in RATES:
        c, t = configuration(rate)
        path = output_path(rate)
        assert not path.exists()
        configuration_records[path.name] = {"model": asdict(c), "training": asdict(t)}
        cells[path.name] = path.as_posix()
    prov = provenance()
    # The reference helper is part of preflight evidence even though the training worker does not import it.
    record = {
        "status": "PASS",
        "h060_existing_sources_unchanged": True,
        "retained_exact_variant_count": 6,
        "full_size_control_probe_count": 12,
        "full_size_control_source_snapshots": compatibility,
        "reference_helper_sha256": sha("results/verification/additive_control_probe_v1.py"),
        "controls": rows,
        "data_hashes": manifest["files"],
        "validation_targets": 322688,
        "validation_stream_exact": True,
        "sampling_rng_sha256": sampler,
        "environment": environment(),
        "common_non_ffn_initial_weights_exact": True,
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
    write(ROOT / "preflight.json", record)
    write(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
        archive.write(
            "results/verification/additive_control_probe_v1.py",
            "results/verification/additive_control_probe_v1.py",
        )
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
                "full_size_control_probes_exact": 12,
                "candidate_optimizer_groups": record["candidate_optimizer_groups"],
            }
        ),
        flush=True,
    )


def sha_tensor(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def finish():
    from src.core.config import ModelConfig, TrainConfig

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
