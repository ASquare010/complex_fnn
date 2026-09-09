"""H052: isolated long-duration product-decay control; no plotting imports."""

import argparse
import hashlib
import json
import math
import zipfile
from copy import deepcopy
from pathlib import Path

PLAN = Path("research/duration_decay_plan.md")
ROOT = Path("results/duration_decay_v1")
RUN = Path("results/runs/wikitext2_blockshuffle_product_lr1200_s17_3200")
RECIPES = ("full_swiglu", "calibrated_narrow", "blockshuffle", "full_gelu")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration(reference):
    raw = {k: deepcopy(reference[k]) for k in ("model", "training")}
    assert raw["training"]["ffn_decay_mode"] == "parameter"
    raw["training"]["ffn_decay_mode"] = "product"
    return raw


def decision(candidate, references, verified):
    if candidate is None or not verified:
        return {"verified_complete": False, "earns_replication": False}
    nll = candidate["validation_loss"]
    assert math.isfinite(nll) and nll > 0
    gates = {
        "at_least_70_percent_fewer_ffn_weights": candidate["ffn_parameters"]
        <= 0.3 * references["full_swiglu"]["ffn_parameters"],
        "beats_calibrated_narrow": nll < references["calibrated_narrow"]["validation_loss"],
    }
    for recipe in ("full_swiglu", "full_gelu"):
        gates[f"within_one_percent_{recipe}"] = nll <= 1.01 * references[recipe]["validation_loss"]
        gates[f"memory_within_ten_percent_{recipe}"] = candidate["peak_allocated_vram_bytes"] <= (
            1.1 * references[recipe]["peak_allocated_vram_bytes"]
        )
    material = nll <= 0.998 * references["blockshuffle"]["validation_loss"]
    return {
        "verified_complete": True,
        "gates": gates,
        "local_gates_pass": all(gates.values()),
        "material_gain_pass": material,
        "earns_replication": all(gates.values()) and material,
        "relative_nll_percent": {
            r: 100 * (nll / m["validation_loss"] - 1) for r, m in references.items()
        },
    }


def archive_verified(path, metrics):
    with zipfile.ZipFile(path / "source.zip") as archive:
        for name, expected in metrics["provenance"]["source_files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name


def prepare():
    from dataclasses import replace

    import torch

    from src.core.config import ModelConfig, TrainConfig
    from src.core.data import TokenData
    from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
    from src.core.reproducibility import environment, provenance
    from src.core.transformer import Transformer

    torch.set_num_threads(4)
    assert not RUN.exists()
    ROOT.mkdir(exist_ok=False)
    final_path = Path("results/verification/long_duration_replication_final_v1.json")
    final = read(final_path)
    tested = read(final["full_test_record"])
    assert final["status"] == "PASS" and final["full_tests_passed"] == 233
    assert not final["research_target_passes"] and tested["returncode"] == 0
    assert tested["source_unchanged"]
    exception = final["post_test_change"]
    for name, expected in tested["source_files"].items():
        if name == exception["file"]:
            assert expected == exception["prior_sha256"]
            assert sha(name) == exception["current_sha256"]
            assert exception["verifier_and_all_nonplot_statements_ast_exact"]
        else:
            assert sha(name) == expected, name
    previous = Path("results/long_duration_replication_v1/result.json")
    assert sha(previous) == final["result_sha256"]
    assert not read(previous)["summary"]["all_seeds_pass_primary_gates"]
    h051 = read("results/long_duration_replication_v1/preflight.json")
    sampler_dir = Path("results/long_duration_v1/sampling")
    sampler = read(sampler_dir / "result.json")
    assert sampler["status"] == "PASS" and sampler["training_batch_calls"] == 3200
    assert sampler["generator_device"] == "cuda"
    assert sampler["optimizer_updates"] == sampler["validation_targets_scored"] == 0
    assert sha(sampler_dir / "states.pt") == sampler["states_sha256"]
    states = torch.load(sampler_dir / "states.pt", map_location="cpu", weights_only=True)
    for count in (800, 3200):
        assert (
            hashlib.sha256(states[f"after_{count}_batches"].numpy().tobytes()).hexdigest()
            == (sampler[f"after_{count}_batches_sha256"])
        )
    assert sampler["data_hashes"] == h051["data_hashes"]
    assert sampler["environment"] == h051["environment"] == environment()
    for name, expected in sampler["data_hashes"].items():
        assert sha(Path("data/wikitext2_v1") / name) == expected
    data = TokenData(Path("data/wikitext2_v1"), "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    refs, treatments, initial_hashes = {}, {}, {}
    raw = None
    for trial in h051["seed17_trials"]:
        recipe = trial["recipe"]
        path = Path("results/runs") / trial["run"]
        for filename in ("metrics.json", "checkpoint.pt"):
            assert sha(path / filename) == trial[filename.split(".")[0] + "_sha256"]
        m = read(path / "metrics.json")
        assert m["environment"] == environment() and m["data"]["files"] == sampler["data_hashes"]
        archive_verified(path, m)
        ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert torch.equal(ckpt["sampling_rng"], states["after_3200_batches"])
        mc, tc = ModelConfig(**m["model"]), TrainConfig(**m["training"])
        assert tc.seed == 17 and tc.steps == 3200 and tc.learning_rate == 0.0012
        model = Transformer(mc, tc.seed)
        initialize_dense_width(model, tc)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        old_groups = parameter_groups(model, tc)
        new_groups = parameter_groups(model, replace(tc, ffn_decay_mode="product"))
        assert group_summary(old_groups) == m["optimizer_parameter_groups"]

        def treatment(groups):
            return {
                id(p): (g["lr_scale"], g["weight_decay"], g["lr"])
                for g in groups
                for p in g["params"]
            }

        old, new = treatment(old_groups), treatment(new_groups)
        changed = []
        for name, p in model.named_parameters():
            assert torch.equal(p, before[name]), name
            a, b = old[id(p)], new[id(p)]
            assert a[0] == b[0] and a[2] == b[2]
            if a != b:
                assert recipe == "blockshuffle" and ".ffn." in name
                assert b[1] == tc.weight_decay / (2 * b[0])
                changed.append(
                    {
                        "name": name,
                        "count": p.numel(),
                        "lr_scale": b[0],
                        "old_decay": a[1],
                        "new_decay": b[1],
                    }
                )
        assert sum(v["count"] for v in changed) == (2801664 if recipe == "blockshuffle" else 0)
        treatments[recipe] = {
            "old_groups": group_summary(old_groups),
            "new_groups": group_summary(new_groups),
            "changed": changed,
        }
        initial_hashes[recipe] = hashlib.sha256(
            b"".join(n.encode() + p.detach().numpy().tobytes() for n, p in model.named_parameters())
        ).hexdigest()
        refs[recipe] = {
            "run": path.name,
            "files": {
                n: sha(path / n)
                for n in (
                    "metrics.json",
                    "checkpoint.pt",
                    "source.zip",
                    "history.jsonl",
                    "initial_diagnostics.json",
                    "final_diagnostics.json",
                )
            },
        }
        if recipe == "blockshuffle":
            raw = configuration(m)
        del model, ckpt, before, old_groups, new_groups
    assert set(refs) == set(RECIPES) and raw is not None
    preflight = {
        "status": "PASS",
        "references": refs,
        "treatments": treatments,
        "initial_weights_unchanged_sha256": initial_hashes,
        "validation_stream_exact": True,
        "data_hashes": sampler["data_hashes"],
        "environment": environment(),
        "sampler_result_sha256": sha(sampler_dir / "result.json"),
        "sampler": sampler,
        "h051_final_audit_sha256": sha(final_path),
        "233_tested_sources_match_except_recorded_palette": True,
    }
    write(ROOT / "preflight.json", preflight)
    protocol = {
        "plan_sha256": sha(PLAN),
        "configurations": {"product_s17": raw},
        "new_training_tokens": 6553600,
        "worker_timeout_seconds": 2400,
        "preflight_sha256": sha(ROOT / "preflight.json"),
        "provenance": provenance(),
    }
    write(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    qualification = {
        "plan_path": PLAN.as_posix(),
        "plan_sha256": sha(PLAN),
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "cells": {"product_s17": RUN.as_posix()},
        "critical_source_hashes": protocol["provenance"]["source_files"],
        "reason": "Frozen isolated product-decay control through unchanged minimal trainer worker.",
    }
    write(ROOT / "qualification.json", qualification)
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS",
                "plan_sha256": sha(PLAN),
                "changed_ffn_weights": 2801664,
                "new_training_tokens": 6553600,
            }
        ),
        flush=True,
    )


def verify():
    import torch

    from src.core.benchmark import forward_flops
    from src.core.config import ModelConfig, TrainConfig
    from src.core.frozen_train_worker import read_cell
    from src.core.trainer import learning_rate

    torch.set_num_threads(4)
    raw, path = read_cell(ROOT / "protocol.json", ROOT / "qualification.json", "product_s17")
    protocol, pre = read(ROOT / "protocol.json"), read(ROOT / "preflight.json")
    assert sha(ROOT / "preflight.json") == protocol["preflight_sha256"]
    refs = {}
    for recipe, reference in pre["references"].items():
        rp = Path("results/runs") / reference["run"]
        assert all(sha(rp / n) == h for n, h in reference["files"].items())
        refs[recipe] = read(rp / "metrics.json")
    for name, expected in pre["data_hashes"].items():
        assert sha(Path("data/wikitext2_v1") / name) == expected
    m = read(path / "metrics.json")
    assert all(m[k] == raw[k] for k in ("model", "training"))
    mc, tc = ModelConfig(**raw["model"]), TrainConfig(**raw["training"])
    assert m["training_tokens"] == 6553600 and m["validation_tokens"] == 322688
    assert m["data"]["files"] == pre["data_hashes"] and m["environment"] == pre["environment"]
    assert m["optimizer_parameter_groups"] == pre["treatments"]["blockshuffle"]["new_groups"]
    assert m["ffn_width_initialization"] == refs["blockshuffle"]["ffn_width_initialization"]
    assert m["activation_compilation"] is None and m["precision"] == "bf16"
    assert m["ffn_parameters"] == mc.unique_ffn_parameters == 2801664
    assert m["total_parameters"] == mc.total_parameters == 9099648
    assert all(m[k] == v for k, v in forward_flops(mc).items())
    assert (
        abs(m["initial_validation_loss"] - refs["blockshuffle"]["initial_validation_loss"]) < 1e-7
    )
    h = [json.loads(s) for s in (path / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in h] == [1, 800, 1600, 2400, 3200]
    old_path = Path("results/runs") / pre["references"]["blockshuffle"]["run"]
    old_first = json.loads((old_path / "history.jsonl").read_text().splitlines()[0])
    assert abs(h[0]["training_loss"] - old_first["training_loss"]) < 1e-7
    assert h[-1]["validation_loss"] == m["validation_loss"]
    for row in h:
        assert all(
            math.isfinite(row[k])
            for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
        )
        assert all(math.isfinite(v) for v in row["layer_gradient_norms_post_clip"].values())
        assert abs(row["learning_rate"] - learning_rate(row["step"] - 1, tc)) < 1e-15
    for stage in ("initial", "final"):
        assert all(r["finite"] for r in read(path / f"{stage}_diagnostics.json").values())
    archive_verified(path, m)
    assert m["provenance"]["source_files"] == protocol["provenance"]["source_files"]
    ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert (
        ckpt["step"] == 3200
        and ckpt["model_config"] == raw["model"]
        and ckpt["training_config"] == raw["training"]
    )
    assert sum(t.numel() for t in ckpt["model"].values()) == mc.total_parameters
    assert all(torch.isfinite(t).all() for t in ckpt["model"].values())
    assert (
        hashlib.sha256(ckpt["sampling_rng"].numpy().tobytes()).hexdigest()
        == pre["sampler"]["after_3200_batches_sha256"]
    )
    change = 100 * (h[-1]["validation_loss"] / h[-2]["validation_loss"] - 1)
    degradation = 100 * (m["validation_loss"] / min(r["validation_loss"] for r in h) - 1)
    result = {
        "status": "complete",
        "run": path.name,
        "plan_sha256": sha(PLAN),
        "decision": decision(m, refs, True),
        "nll": m["validation_loss"],
        "peak_mib": m["peak_allocated_vram_bytes"] / 2**20,
        "new_training_tokens": 6553600,
        "clipped_step_fraction": m["clipped_step_fraction"],
        "numerical_failures": 0,
        "relative_last_800_step_nll_change_percent": change,
        "late_plateau_screen_passes": abs(change) <= 0.2 and degradation <= 0.2,
        "source_data_sampler_initial_losses_and_finite_diagnostics_verified": True,
        "files": {
            n: sha(path / n)
            for n in (
                "metrics.json",
                "checkpoint.pt",
                "source.zip",
                "history.jsonl",
                "initial_diagnostics.json",
                "final_diagnostics.json",
            )
        },
    }
    write(ROOT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify}[args.command]()
