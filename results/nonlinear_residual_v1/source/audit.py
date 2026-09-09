"""Independent regeneration, checkpoint rescoring and H099 gate verification."""

import csv
import itertools
import math
import os
import statistics as stats
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.neuron_geometry_v1.source.study import tensor_sha
from results.nonlinear_residual_v1.source.model import CONTROLS, FORMS, TASKS, ResidualTaskFFN

ROOT = Path("results/nonlinear_residual_v1")
SEEDS, RATES = (17, 29, 43), (0.001, 0.003)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def close(a, b, tol=1e-12):
    assert math.isclose(a, b, abs_tol=tol, rel_tol=tol), (a, b)


@torch.inference_mode()
def evaluate(model, x, y):
    return (
        sum(
            F.mse_loss(model(a), b, reduction="sum").item()
            for a, b in zip(x.split(256), y.split(256))
        )
        / y.numel()
    )


def check_data(data):
    x = torch.randn(73728, 384, generator=torch.Generator().manual_seed(9860))
    indices = torch.randperm(384, generator=torch.Generator().manual_seed(9861))[:16]
    matrix = torch.linalg.qr(
        torch.randn(384, 384, generator=torch.Generator().manual_seed(9862), dtype=torch.float64)
    ).Q[:16]
    assert torch.equal(x, data["x"]) and torch.equal(indices, data["indices"])
    assert torch.equal(matrix, data["rotation"])
    q = x[:, indices].double()
    values = {
        "quadratic": (q * q - 1) / math.sqrt(2),
        "cubic": q * (q * q - 3) / math.sqrt(6),
        "product": q * q[:, (torch.arange(16) + 1) % 16],
        "piecewise": (torch.where(q > 0, q, 0).square() - 0.5 - math.sqrt(2 / math.pi) * q)
        / math.sqrt(1.25 - 2 / math.pi),
    }
    for task, latent in values.items():
        raw = latent @ matrix
        scale = raw[:65536].std(dim=0, correction=0)
        assert torch.equal(scale, data["scales"][task])
        assert torch.equal((raw / scale).float(), data["targets"][task])
    for seed in SEEDS:
        indices = torch.randint(
            65536, (300, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        assert torch.equal(indices, data["streams"][seed])


def check_affine(data, rows):
    assert len(rows) == 12
    for row in rows:
        assert sha(row["checkpoint"]) == row["sha256"]
        saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        counts = torch.bincount(data["streams"][row["seed"]].flatten(), minlength=65536)
        assert torch.equal(counts, saved["counts"])
        gram = torch.zeros(385, 385, dtype=torch.float64)
        cross = torch.zeros(385, 384, dtype=torch.float64)
        for start in range(0, 65536, 2048):
            x = data["x"][start : start + 2048].double()
            y = data["targets"][row["task"]][start : start + 2048].double()
            weight = counts[start : start + 2048].double().sqrt()[:, None]
            x = torch.cat((x, torch.ones_like(x[:, :1])), 1) * weight
            gram += x.T @ x
            cross += x.T @ (y * weight)
        for a, b in ((gram, saved["gram"]), (cross, saved["cross"])):
            assert (a - b).norm() / b.norm() < 1e-10
        coefficients = torch.linalg.solve(gram, cross)
        torch.testing.assert_close(coefficients, saved["coefficients"], atol=1e-9, rtol=1e-9)
        x = data["x"][69632:]
        pred = torch.cat((x, torch.ones_like(x[:, :1])), 1) @ saved["coefficients"].float()
        close(
            (pred - data["targets"][row["task"]][69632:]).square().mean().item(),
            row["reporting_mse"],
            1e-6,
        )


def run():
    before = read(ROOT / "before.json")
    assert all(sha(p) == h for p, h in {**before["source_hashes"], **before["anchors"]}.items())
    assert read(ROOT / "study_process.json")["status"] == "PASS"
    raw = read(ROOT / "raw_result.json")
    assert sha(ROOT / "data.pt") == raw["data_sha256"]
    data = torch.load(ROOT / "data.pt", map_location="cpu", weights_only=True)
    check_data(data)
    check_affine(data, raw["affine"])
    observations = list((ROOT / "qualification").glob("*.json"))
    assert len(observations) == 25
    pair_count = 0
    for p in observations:
        row = read(p)
        assert row["status"] == "PASS" and row["optimizer_updates"] == 0
        if "tensor_file" in row:
            tensor = p.with_name(row["tensor_file"])
            assert sha(tensor) == row["tensor_sha256"]
            for item in torch.load(tensor, map_location="cpu", weights_only=True):
                a, b, tol = item["actual"], item["expected"], item["tolerance"]
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                torch.testing.assert_close(a, b, atol=tol, rtol=tol)
                pair_count += 1
    rows = raw["rows"]
    assert len(rows) == 264 and {(r["task"], r["form"], r["seed"], r["rate"]) for r in rows} == set(
        itertools.product(TASKS, FORMS, SEEDS, RATES)
    )
    checked = 0
    for task in TASKS:
        x, y = data["x"].cuda(), data["targets"][task].cuda()
        close(data["targets"][task][69632:].double().square().mean().item(), raw["zero_mse"][task])
        for row in [r for r in rows if r["task"] == task]:
            assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
            saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
            training = read(ROOT / "cells" / row["label"] / "training.json")
            model = ResidualTaskFFN(row["form"], row["seed"], task)
            assert {k: tensor_sha(v) for k, v in model.state_dict().items()} == training[
                "initial_state_sha256"
            ]
            assert sum(p.numel() for p in model.parameters()) == row["parameters"]
            assert sum(p.numel() for p in model.curve.parameters()) == row["activation_parameters"]
            assert (
                2 * sum(p.numel() for p in model.parameters() if p.ndim == 2)
                == row["matrix_forward_flops_per_example"]
            )
            expected_groups = [
                {
                    "name": g["parameter_name"],
                    "lr": g["lr"],
                    "parameters": sum(p.numel() for p in g["params"]),
                }
                for g in model.optimizer_groups(row["rate"])
            ]
            assert expected_groups == training["optimizer_groups"]
            assert saved["step"] == 300 and set(saved["optimizer_steps"]) == {300}
            assert (
                saved["data_sha256"] == raw["data_sha256"]
                and saved["protocol_sha256"] == raw["protocol_sha256"]
            )
            assert saved["stream_sha256"] == tensor_sha(data["streams"][row["seed"]])
            history = training["history"]
            assert [r["step"] for r in history] == list(range(1, 301))
            assert all(math.isfinite(v) for r in history for v in r.values())
            assert all(torch.isfinite(v).all() for v in saved["model"].values())
            for phase in ("update", "forward", "backward"):
                close(
                    stats.median(r[f"{phase}_ms"] for r in history[50:]), row[f"median_{phase}_ms"]
                )
            close(sum(r["preclip_norm"] > 1 for r in history) / 300, row["clipped_fraction"])
            model.load_state_dict(saved["model"])
            model.cuda()
            assert evaluate(model, x[65536:69632], y[65536:69632]) == row["selection_mse"]
            assert evaluate(model, x[69632:], y[69632:]) == row["reporting_mse"]
            checked += 2
            del model, saved
        del x, y
    selected = [r for r in rows if r["selected"]]
    assert len(selected) == 132 and checked == 528
    for task, form, seed in itertools.product(TASKS, FORMS, SEEDS):
        pair = [r for r in rows if (r["task"], r["form"], r["seed"]) == (task, form, seed)]
        chosen = min(pair, key=lambda r: r["selection_mse"])
        assert chosen["selected"] and sum(r["selected"] for r in pair) == 1
        assert (
            read(ROOT / "selections" / f"{task}_{form}_{seed}.json")["selected_label"]
            == chosen["label"]
        )
    lookup = {(r["task"], r["form"], r["seed"]): r for r in selected}
    affine = {(r["task"], r["seed"]): r for r in raw["affine"]}
    summary = raw["summary"]

    def ratio(form, ref, seeds=SEEDS):
        vals = [
            lookup[t, form, s]["reporting_mse"]
            / (
                affine[t, s]["reporting_mse"]
                if ref == "affine"
                else lookup[t, ref, s]["reporting_mse"]
            )
            for t in TASKS
            for s in seeds
        ]
        return math.exp(sum(map(math.log, vals)) / len(vals))

    for form, task in itertools.product(FORMS, TASKS):
        values = [lookup[task, form, s]["reporting_mse"] for s in SEEDS]
        actual = summary["per_task"][form][task]
        assert actual["values"] == values
        for key, fn in (
            ("mean", stats.mean),
            ("median", stats.median),
            ("variance", stats.variance),
            ("sample_sd", stats.stdev),
        ):
            close(actual[key], fn(values))
        close(summary["ratios_vs_affine"][form], ratio(form, "affine"))
    for form in FORMS:
        subset = [r for r in selected if r["form"] == form]
        for key in (
            "median_update_ms",
            "median_forward_ms",
            "median_backward_ms",
            "clipped_fraction",
        ):
            close(summary["resources"][form][key], stats.mean(r[key] for r in subset))
        assert summary["resources"][form]["max_peak_bytes"] == max(r["peak_bytes"] for r in subset)
    for ref, value in summary["candidate_ratios"].items():
        close(value, ratio("learned_mix", ref))
    for ref, values in summary["candidate_seed_ratios"].items():
        for seed, value in zip(SEEDS, values):
            close(value, ratio("learned_mix", ref, (seed,)))
    pc = all(
        lookup[t, "feature_aware", s]["reporting_mse"] <= 0.05 * raw["zero_mse"][t]
        for t in TASKS
        for s in SEEDS
    )
    full_tasks = {
        f: [t for t in TASKS if summary["per_task"][f][t]["mean"] <= 0.5 * raw["zero_mse"][t]]
        for f in ("full_gelu", "full_swiglu", "full_relu2")
    }
    assert full_tasks == summary["full_neural_successful_tasks"]
    neural = any(len(v) >= 3 for v in full_tasks.values())
    caps = []
    for task in TASKS:
        strongest = min(
            *[summary["per_task"][f][task]["mean"] for f in CONTROLS],
            stats.mean(affine[task, s]["reporting_mse"] for s in SEEDS),
        )
        cap = summary["per_task"]["learned_mix"][task]["mean"] / strongest
        close(cap, summary["candidate_task_ratios"][task])
        caps.append(cap)
    gates = {
        "finite_complete_grid": True,
        "feature_aware_positive_control": pc,
        "full_neural_learnability": neural,
        "seventy_percent_reduction": 1 - 351052 / 1181568 >= 0.7,
        "two_percent_over_every_control": all(
            ratio("learned_mix", r) <= 0.98 for r in (*CONTROLS, "affine")
        ),
        "every_seed_better": all(
            ratio("learned_mix", r, (s,)) < 1 for r in (*CONTROLS, "affine") for s in SEEDS
        ),
        "per_task_cap": all(v <= 1.05 for v in caps),
        "full_reference_proxy": max(
            ratio("learned_mix", "full_gelu"), ratio("learned_mix", "full_swiglu")
        )
        <= 1.01,
        "update_time_cap": summary["resources"]["learned_mix"]["median_update_ms"]
        <= 1.25 * summary["resources"]["narrow_gelu"]["median_update_ms"],
    }
    assert summary["gates"] == gates
    verdict = (
        "INCONCLUSIVE_ASSAY"
        if not pc or not neural
        else ("EARNS_CONFIRMATION" if all(gates.values()) else "REJECTED_AT_THIS_BUDGET")
    )
    assert summary["verdict"] == verdict
    assert all(sha(p) == h for p, h in {**before["source_hashes"], **before["anchors"]}.items())
    write_json(
        ROOT / "result.json",
        {
            "status": "PASS",
            "research_goal_achieved": False,
            "runs": 264,
            "updates": 79200,
            "presentations": 20275200,
            "affine_fits": 12,
            "qualification_checks": 25,
            "qualification_pairs_rechecked": pair_count,
            "exact_checkpoint_scores": 528,
            "exact_initializations": 264,
            "rate_selections_verified": 132,
            "summary": summary,
            "affine": raw["affine"],
            "zero_mse": raw["zero_mse"],
            "raw_result_sha256": sha(ROOT / "raw_result.json"),
            "protocol_sha256": raw["protocol_sha256"],
        },
    )
    columns = [
        "task",
        "form",
        "seed",
        "rate",
        "parameters",
        "activation_parameters",
        "selection_mse",
        "reporting_mse",
        "median_update_ms",
        "median_forward_ms",
        "median_backward_ms",
        "peak_bytes",
        "clipped_fraction",
        "selected",
        "checkpoint_sha256",
    ]
    with (ROOT / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "audit_process.json", status)
    start = time.perf_counter()
    try:
        run()
        status["status"] = "PASS"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        status.update(
            elapsed_seconds=time.perf_counter() - start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(ROOT / "audit_process.json", status, exclusive=False)
