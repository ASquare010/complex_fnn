"""H077 fixed-rate, three-seed longer replication through the unchanged H076 adapter."""

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from results.ungated_lm_screen_v1.source import study as base
from results.ungated_lm_screen_v1.source.study import Observer, operation_counts, trainer
from src.core.native_recompute_audit import digest, finite_tree, read, sha, tensor_record, write_new
from src.core.reproducibility import environment, provenance

ROOT = Path("results/ungated_lm_replication_v1")
PLAN = Path("research/ungated_lm_replication_plan.md")
CACHE = base.CACHE
SEEDS = (17, 29, 43)
FORMS = ("full_swiglu", "narrow_swiglu", "plain", "gelu_matched", "narrow_gelu", "full_gelu")
RATES = {f: 0.0006 if f in ("narrow_gelu", "plain") else 0.0012 for f in FORMS}
MODES = {f: base.MODES[f] for f in FORMS}
ORDERS = {17: FORMS, 29: FORMS[::-1], 43: FORMS[3:] + FORMS[:3]}
CELLS = {
    f"{form}_seed{seed}": {"form": form, "seed": seed, "rate": RATES[form]}
    for seed in SEEDS
    for form in ORDERS[seed]
}
STEPS = 800
TARGETS = STEPS * 16 * 128
T_CRITICAL = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))


def configuration(form, seed):
    if form not in FORMS or seed not in SEEDS:
        raise ValueError("Only the frozen six forms and three seeds are allocated")
    mc, old = base.configuration(form, RATES[form])
    tc = replace(old, steps=STEPS, log_every=200, seed=seed)
    mc.validate()
    tc.validate()
    return mc, tc


def summary(values):
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("Three finite seed values are required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    radius = T_CRITICAL * sd / math.sqrt(3)
    return {"mean": mean, "sample_sd": sd, "exploratory_ci95": [mean - radius, mean + radius]}


def decisions(rows):
    if set(rows) != set(CELLS):
        raise ValueError("Every one of the 18 allocated cells is required")
    complete = all(
        v["status"] == "SCREENED" and v["all_finite"] and finite_tree(v) for v in rows.values()
    )
    gates, paired, margins = {}, {}, {}
    for seed in SEEDS:
        m = rows[f"gelu_matched_seed{seed}"]
        refs = {f: rows[f"{f}_seed{seed}"] for f in FORMS}
        gates[str(seed)] = {
            "all_cells_complete_finite": complete,
            "at_least_70_percent_fewer_ffn_weights": all(
                100 * (1 - m["ffn_parameters"] / refs[f]["ffn_parameters"]) >= 70
                for f in ("full_swiglu", "full_gelu")
            ),
            **{
                f"nll_within_one_percent_{f}": m["validation_loss"]
                <= 1.01 * refs[f]["validation_loss"]
                for f in ("full_swiglu", "full_gelu")
            },
            **{
                f"nll_beats_{f}": m["validation_loss"] < refs[f]["validation_loss"]
                for f in ("narrow_swiglu", "narrow_gelu")
            },
            **{
                f"memory_within_ten_percent_{f}": m["peak_allocated_vram_bytes"]
                <= 1.1 * refs[f]["peak_allocated_vram_bytes"]
                for f in ("full_swiglu", "full_gelu")
            },
        }
        margins[str(seed)] = {
            f: m["validation_loss"] <= 0.998 * refs[f]["validation_loss"]
            for f in ("narrow_swiglu", "narrow_gelu")
        }
    aggregate = {
        f: summary([rows[f"{f}_seed{seed}"]["validation_loss"] for seed in SEEDS]) for f in FORMS
    }
    for ref in FORMS:
        if ref == "gelu_matched":
            continue
        paired[ref] = {
            **summary(
                [
                    rows[f"gelu_matched_seed{seed}"]["validation_loss"]
                    - rows[f"{ref}_seed{seed}"]["validation_loss"]
                    for seed in SEEDS
                ]
            ),
            "relative_mean_nll_percent": 100
            * (aggregate["gelu_matched"]["mean"] / aggregate[ref]["mean"] - 1),
            "per_seed_relative_nll_percent": {
                str(seed): 100
                * (
                    rows[f"gelu_matched_seed{seed}"]["validation_loss"]
                    / rows[f"{ref}_seed{seed}"]["validation_loss"]
                    - 1
                )
                for seed in SEEDS
            },
        }
    plateau = {
        cell: abs(v["late_nll_change_percent"]) <= 0.2 and v["final_above_best_percent"] <= 0.2
        for cell, v in rows.items()
    }
    return {
        "per_seed_gates": gates,
        "replicated_primary_pass": all(all(g.values()) for g in gates.values()),
        "nll_summary": aggregate,
        "paired_candidate_minus_control": paired,
        "descriptive_point_two_percent_narrow_margin": margins,
        "operational_late_plateau": plateau,
        "all_forms_seeds_late_plateau": all(plateau.values()),
    }


def worker(cell):
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    assert environment() == protocol["environment"]
    assert all(sha(CACHE / n) == h for n, h in protocol["data_files"].items())
    spec = CELLS[cell]
    mc, tc = configuration(spec["form"], spec["seed"])
    output = ROOT / "runs" / cell
    assert not output.exists()
    observer = Observer(output, tc, MODES[spec["form"]])

    def full_provenance():
        prov = provenance()
        prov["source_files"] = dict(protocol["sources"])
        prov["source_hash"] = hashlib.sha256(
            json.dumps(prov["source_files"], sort_keys=True).encode()
        ).hexdigest()
        return prov

    with (
        patch.object(trainer, "Transformer", observer.constructor),
        patch.object(trainer, "initialize_dense_width", observer.initialize),
        patch.object(trainer, "provenance", full_provenance),
        patch.object(trainer, "forward_flops", operation_counts),
        patch.object(torch.nn.utils, "clip_grad_norm_", observer.clip),
    ):
        metrics = trainer.train(mc, tc, CACHE, output)
    assert observer.pending is None and len(observer.steps) == STEPS
    assert metrics["validation_tokens"] == 322688 and metrics["training_tokens"] == TARGETS
    assert (
        metrics["clipped_step_fraction"]
        == sum(r["gradient_norm_pre_clip"] > 1 for r in observer.steps) / STEPS
    )
    ckpt = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == STEPS and finite_tree(ckpt)
    assert all(float(v["step"]) == STEPS for v in ckpt["optimizer"]["state"].values())
    for phase in ("initial", "final"):
        diagnostics = read(output / (phase + "_diagnostics.json"))
        assert finite_tree(diagnostics) and all(v["finite"] for v in diagnostics.values())
    history = list(map(json.loads, (output / "history.jsonl").read_text().splitlines()))
    for row in history:
        assert all(
            row[k] == observer.steps[row["step"] - 1][k]
            for k in ("step", "training_loss", "gradient_norm_pre_clip", "learning_rate")
        )
    assert [r["step"] for r in history] == [1, 200, 400, 600, 800]
    initial = read(output / "initial_state_signature.json")
    expected = read(ROOT / "qualification/initial_signatures.json")[cell]
    assert initial["weights"] == expected["weights"]
    assert initial["parameters"] == expected["parameters"]
    sampler = read(ROOT / "qualification/samplers.json")["seeds"][str(spec["seed"])]
    assert tensor_record(ckpt["sampling_rng"]) == sampler["state_after_800"]
    if spec["seed"] == 17:
        prior = read("results/ungated_lm_screen_v1/result.json")
        previous = prior["rows"][prior["selected_cells"][spec["form"]]]
        old_history = list(
            map(
                json.loads,
                (Path("results/ungated_lm_screen_v1/runs") / previous["run"] / "history.jsonl")
                .read_text()
                .splitlines(),
            )
        )
        assert metrics["initial_validation_loss"] == previous["initial_validation_loss"]
        assert history[0]["training_loss"] == old_history[0]["training_loss"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    write_new(
        output / "qualification.json",
        {
            "status": "PASS",
            "all_finite": True,
            "initial_weights_match_qualification": True,
            "seed17_initial_loss_matches_h076": True if spec["seed"] == 17 else None,
            "all_steps_match_shared_history": True,
            "execution_mode": MODES[spec["form"]],
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "metrics_sha256": sha(output / "metrics.json"),
            "checkpoint_sha256": sha(output / "checkpoint.pt"),
            "source_archive_sha256": sha(output / "source.zip"),
            "all_steps_sha256": sha(output / "all_steps.jsonl"),
            "sampling_rng": tensor_record(ckpt["sampling_rng"]),
            "final_weights": digest(ckpt["model"]),
            "final_optimizer": digest(ckpt["optimizer"]),
            "official_test_scored": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "cell": cell,
                "validation_loss": metrics["validation_loss"],
                "peak_mib": metrics["peak_allocated_vram_bytes"] / 2**20,
            }
        ),
        flush=True,
    )


def finish():
    protocol = read(ROOT / "protocol.json")
    rows = {}
    for cell in CELLS:
        folder = ROOT / "runs" / cell
        q, m = read(folder / "qualification.json"), read(folder / "metrics.json")
        event = read(ROOT / "processes" / (cell + ".json"))
        assert q["status"] == event["status"] == "PASS"
        assert event["source_unchanged"] and event["returncode"] == 0
        assert sha(folder / "metrics.json") == q["metrics_sha256"]
        assert sha(folder / "checkpoint.pt") == q["checkpoint_sha256"]
        history = list(map(json.loads, (folder / "history.jsonl").read_text().splitlines()))
        assert m["validation_loss"] == history[-1]["validation_loss"]
        rows[cell] = {
            **m,
            "all_finite": q["all_finite"],
            "execution_mode": q["execution_mode"],
            "late_nll_change_percent": 100
            * (history[-1]["validation_loss"] / history[-2]["validation_loss"] - 1),
            "final_above_best_percent": 100
            * (history[-1]["validation_loss"] / min(h["validation_loss"] for h in history) - 1),
        }
    verdict = decisions(rows)
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    result = {
        "status": "complete",
        "rows": rows,
        **verdict,
        "language_trials": 18,
        "language_optimizer_updates": 14400,
        "language_training_targets": 29491200,
        "shared_validation_target_exposures": 40658688,
        "qualification_optimizer_updates": 0,
        "scientific_attempts": 1,
        "scientific_retries": 0,
        "official_test_scored": False,
        "research_goal_achieved": False,
    }
    write_new(ROOT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    parser.add_argument("--cell", choices=list(CELLS))
    args = parser.parse_args()
    worker(args.cell) if args.command == "worker" else finish()
