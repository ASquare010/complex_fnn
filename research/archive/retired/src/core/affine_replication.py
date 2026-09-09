"""Two independent seeds for the promoted 800-step affine WikiText recipe."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import traceback
import zipfile
from dataclasses import replace
from pathlib import Path

import torch

from src.core.affine_longer import (
    CRITICAL,
    RECIPES,
    configuration,
    output_path,
    promotion,
    verify_trial,
)
from src.core.optimization import initialize_dense_width
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

SEEDS = (29, 43)
PLAN = Path("research/affine_activation_replication_plan.md")
PREVIOUS = Path("results/affine_longer_v2")


def paired_statistics(differences):
    assert len(differences) == 3 and all(math.isfinite(x) for x in differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half = 4.302652729911275 * sd / math.sqrt(3)
    nonzero = [x for x in differences if x != 0]
    wins = sum(x < 0 for x in nonzero)
    n = len(nonzero)
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
    return {
        "differences": differences,
        "mean": mean,
        "sample_sd": sd,
        "min": min(differences),
        "max": max(differences),
        "student_t_95_interval": [mean - half, mean + half],
        "student_t_df": 2,
        "strict_wins": wins,
        "nonzero_pairs": n,
        "one_sided_sign_p": p,
        "interpretation": "Exploratory paired mean interval assumes approximately normal independent differences; selected seed17 and three samples limit confirmatory interpretation. Sign test concerns the paired median and excludes exact ties.",
    }


def initial_checks(seed):
    x = torch.arange(16).reshape(2, 8)
    common = None
    base = None
    results = {}
    for recipe in RECIPES:
        mc, tc = configuration(recipe, seed)
        tiny = replace(
            mc, width=24, layers=2, heads=3, context=8, vocab_size=32, hidden=48, groups=3
        )
        model = Transformer(tiny, seed)
        initialize_dense_width(model, tc)
        shared = {n: p.detach().clone() for n, p in model.named_parameters() if ".ffn." not in n}
        if common is None:
            common = shared
        assert all(torch.equal(p, common[n]) for n, p in shared.items())
        logits = model(x)
        loss = model.loss(x, x + 1)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if recipe == "blockshuffle":
            base = (
                logits.detach(),
                loss.detach(),
                {n: p.grad.clone() for n, p in model.named_parameters()},
            )
        if recipe == "blockshuffle_affine":
            assert base is not None and torch.equal(logits, base[0]) and torch.equal(loss, base[1])
            assert all(
                torch.equal(p.grad, base[2][n]) for n, p in model.named_parameters() if n in base[2]
            )
        results[recipe] = {
            "common_non_ffn_parameters_exact": True,
            "initial_loss": loss.item(),
            "all_gradients_finite": True,
        }
    return {
        "seed": seed,
        "recipes": results,
        "affine_and_base_initial_logits_loss_common_gradients_exact": True,
        "scope": "Tiny CPU native forwards without recomputation; actual full-model initial NLL is checked within seed after training.",
    }


def preflight():
    torch.set_num_threads(4)
    previous = json.loads((PREVIOUS / "result.json").read_text())
    assert previous["status"] == "complete" and previous["earns_independent_seeds"]
    initial = json.loads((PREVIOUS / "preflight.json").read_text())
    protocol = json.loads((PREVIOUS / "protocol.json").read_text())
    assert initial["environment"] == environment()
    assert all(sha256(Path(n)) == protocol["provenance"]["source_files"][n] for n in CRITICAL)
    checked = []
    for recipe in RECIPES:
        m, _ = verify_trial(recipe, initial)
        row = next(r for r in previous["trials"] if r["recipe"] == recipe)
        assert sha256(output_path(recipe) / "checkpoint.pt") == row["checkpoint_sha256"]
        assert sha256(output_path(recipe) / "metrics.json") == row["metrics_sha256"]
        checked.append(row)
        original_mc, original_tc = configuration(recipe)
        for seed in SEEDS:
            mc, tc = configuration(recipe, seed)
            assert mc == original_mc and tc == replace(original_tc, seed=seed)
    return {
        "previous_trials": checked,
        "initial_checks": [initial_checks(s) for s in SEEDS],
        "prior_result_sha256": sha256(PREVIOUS / "result.json"),
        "all_shared_computation_hashes_unchanged": True,
    }


def load_three_seeds(initial):
    rows = {}
    rng_hashes = {}
    for seed in (17, *SEEDS):
        rows[seed] = {}
        sampling = None
        for recipe in RECIPES:
            m, state = verify_trial(recipe, initial, seed)
            rows[seed][recipe] = m
            if sampling is None:
                sampling = state
            assert torch.equal(sampling, state)
        assert (
            abs(
                rows[seed]["blockshuffle"]["initial_validation_loss"]
                - rows[seed]["blockshuffle_affine"]["initial_validation_loss"]
            )
            < 1e-7
        )
        rng_hashes[seed] = hashlib.sha256(sampling.numpy().tobytes()).hexdigest()
    assert len(set(rng_hashes.values())) == 3
    return rows, rng_hashes


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        checks = preflight()
        write_json(output / "preflight.json", checks)
        initial = json.loads((PREVIOUS / "preflight.json").read_text())
        protocol = {
            "plan_sha256": sha256(PLAN),
            "provenance": provenance(),
            "environment": environment(),
            "seeds": SEEDS,
            "recipes": RECIPES,
            "worker_timeout_seconds": 2400,
            "new_training_tokens": 16384000,
            "config_hashes": {k: sha256(Path(f"configs/wikitext2_{k}_800.json")) for k in RECIPES},
        }
        write_json(output / "protocol.json", protocol)
        with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in protocol["provenance"]["source_files"]:
                archive.write(name, name)
            archive.write(PLAN, PLAN.as_posix())
        for index, seed in enumerate(SEEDS, 1):
            order = RECIPES[index:] + RECIPES[:index]
            for recipe in order:
                path = output_path(recipe, seed)
                assert not path.exists(), f"Do not overwrite {path}"
                print(f"Starting {recipe}, seed {seed}, 800 steps", flush=True)
                with (output / f"{recipe}_s{seed}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-X",
                            "faulthandler",
                            "-m",
                            "src.core.affine_longer",
                            "worker",
                            "--recipe",
                            recipe,
                            "--seed",
                            str(seed),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=2400,
                        check=True,
                    )
                m, _ = verify_trial(recipe, initial, seed)
                assert all(
                    m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                    for n in (*CRITICAL, "src/core/affine_longer.py")
                )
                completed.append(
                    {
                        "seed": seed,
                        "recipe": recipe,
                        "run": path.name,
                        "validation_nll": m["validation_loss"],
                        "metrics_sha256": sha256(path / "metrics.json"),
                        "checkpoint_sha256": sha256(path / "checkpoint.pt"),
                    }
                )
                write_json(output / "progress.json", {"status": "running", "completed": completed})
                print(json.dumps(completed[-1]), flush=True)
        rows, rng_hashes = load_three_seeds(initial)
        gates = {seed: promotion(values) for seed, values in rows.items()}
        differences = {
            recipe: paired_statistics(
                [
                    rows[seed]["blockshuffle_affine"]["validation_loss"]
                    - rows[seed][recipe]["validation_loss"]
                    for seed in rows
                ]
            )
            for recipe in RECIPES
            if recipe != "blockshuffle_affine"
        }
        result = {
            "status": "complete",
            "trials": completed,
            "gates_by_seed": gates,
            "replication_passes": all(all(g.values()) for g in gates.values()),
            "paired_statistics": differences,
            "sampling_rng_hashes": rng_hashes,
            "sampling_rng_matches_within_each_seed": True,
            "affine_base_initial_nll_matches_each_seed": True,
            "all_diagnostics_finite": True,
            "plan_sha256": protocol["plan_sha256"],
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": completed})
        print(json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "completed": completed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure.json", failure)
        write_json(output / "progress.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
