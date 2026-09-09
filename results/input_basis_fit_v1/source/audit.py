"""Rescore every saved checkpoint and independently check H094 decisions."""

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
from results.input_basis_fit_v1.source import study

ROOT = Path("results/input_basis_fit_v1")


@torch.inference_mode()
def evaluate(model, x, y):
    totals = [
        F.mse_loss(model(x[i : i + 256]), y[i : i + 256], reduction="sum").item()
        for i in range(0, x.shape[0], 256)
    ]
    return sum(totals) / y.numel()


def run():
    before = read(ROOT / "before.json")
    assert all(sha(p) == h for p, h in before["source_hashes"].items())
    assert read(ROOT / "fitting_process.json")["status"] == "PASS"
    raw = read(ROOT / "raw_result.json")
    assert sha(ROOT / "data.pt") == raw["data_sha256"]
    data = torch.load(ROOT / "data.pt", map_location="cpu", weights_only=True)
    x = (torch.rand(73728, 384, generator=torch.Generator().manual_seed(9844)) * 2 - 1) * math.sqrt(
        3
    )
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(9283))
    rotation = torch.linalg.qr(
        torch.randn(384, 384, generator=torch.Generator().manual_seed(9282))
    ).Q
    assert torch.equal(x, data["x"]) and torch.equal(perm, data["permutation"])
    assert torch.equal(rotation, data["rotation"])
    a, b, c, d = [x[:, perm][:, torch.roll(torch.arange(384), -i)] for i in range(4)]
    targets = {
        "smooth": (2 * a).sin() + b**2 + (0.5 * c).exp() - d,
        "oscillatory": (3 * a).sin() * (2 * b).cos() + 0.5 * (4 * c + d).sin(),
        "multiplicative": a * b + 2 * b * c * d + a**2 * d,
        "piecewise": (a + b).relu() - 0.7 * (c - d).abs() + torch.where(a > 0, b, c),
    }
    for task, target in targets.items():
        target = target @ rotation
        scale = target[:65536].std(dim=0, correction=0)
        assert torch.equal(scale, data["scales"][task])
        assert torch.equal(target / scale, data["targets"][task])
        assert (
            data["targets"][task][69632:].double().square().mean().item() == raw["zero_mse"][task]
        )
    for seed in (17, 29, 43):
        stream = torch.randint(
            65536, (300, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        assert torch.equal(stream, data["streams"][seed])
    del a, b, c, d, targets, target, x
    rows = raw["rows"]
    grid = {(r["task"], r["form"], r["seed"], r["rate"]) for r in rows}
    assert len(rows) == 336 and grid == set(
        itertools.product(study.TASKS, study.FORMS, study.SEEDS, study.RATES)
    )
    selected, checked = [], 0
    for task in study.TASKS:
        x, y = data["x"].cuda(), data["targets"][task].cuda()
        for row in [r for r in rows if r["task"] == task]:
            assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
            state = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
            train = read(ROOT / "cells" / row["label"] / "training.json")
            model = study.make_model(row["form"], row["seed"])
            assert {k: study.shared.tensor_sha(v) for k, v in model.state_dict().items()} == train[
                "initial_state_sha256"
            ]
            assert sum(p.numel() for p in model.parameters()) == row["parameters"]
            assert state["step"] == 300 and set(state["optimizer_steps"]) == {300}
            assert state["data_sha256"] == raw["data_sha256"]
            assert state["protocol_sha256"] == raw["protocol_sha256"]
            assert state["stream_sha256"] == study.shared.tensor_sha(data["streams"][row["seed"]])
            history = train["history"]
            assert [r["step"] for r in history] == list(range(1, 301))
            assert all(math.isfinite(v) for r in history for v in r.values())
            assert all(torch.isfinite(v).all() for v in state["model"].values())
            for kind in ("update", "forward", "backward"):
                assert (
                    stats.median(r[f"{kind}_ms"] for r in history[50:]) == row[f"median_{kind}_ms"]
                )
            assert sum(r["preclip_norm"] > 1 for r in history) / 300 == row["clipped_fraction"]
            model.load_state_dict(state["model"])
            model.cuda()
            assert evaluate(model, x[65536:69632], y[65536:69632]) == row["selection_mse"]
            assert evaluate(model, x[69632:], y[69632:]) == row["reporting_mse"]
            checked += 2
            if row["selected"]:
                selected.append(row)
            del model, state
        del x, y
    assert len(selected) == 168 and checked == 672
    for task, form, seed in itertools.product(study.TASKS, study.FORMS, study.SEEDS):
        pair = [r for r in rows if (r["task"], r["form"], r["seed"]) == (task, form, seed)]
        chosen = min(pair, key=lambda r: r["selection_mse"])
        assert chosen["selected"] and sum(r["selected"] for r in pair) == 1
        assert (
            read(ROOT / "selections" / f"{task}_{form}_s{seed}.json")["selected_label"]
            == chosen["label"]
        )
    lookup = {(r["task"], r["form"], r["seed"]): r for r in selected}
    summary = raw["summary"]

    def ratio(form, ref, seeds=(17, 29, 43)):
        vals = [
            lookup[t, form, s]["reporting_mse"] / lookup[t, ref, s]["reporting_mse"]
            for t in study.TASKS
            for s in seeds
        ]
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    def close(a, b):
        assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (a, b)

    for form, task in itertools.product(study.FORMS, study.TASKS):
        values = [lookup[task, form, s]["reporting_mse"] for s in study.SEEDS]
        row = summary["per_task"][form][task]
        assert row["values"] == values
        for key, fn in (
            ("mean", stats.mean),
            ("median", stats.median),
            ("variance", stats.variance),
            ("sample_sd", stats.stdev),
        ):
            close(row[key], fn(values))
        close(summary["ratios_vs_narrow_gelu"][form], ratio(form, "narrow_gelu"))
    for form in study.FORMS:
        vals = [r for r in selected if r["form"] == form]
        for key in (
            "median_update_ms",
            "median_forward_ms",
            "median_backward_ms",
            "clipped_fraction",
        ):
            close(summary["resources"][form][key], stats.mean(r[key] for r in vals))
        assert summary["resources"][form]["max_peak_bytes"] == max(r["peak_bytes"] for r in vals)
    for form in study.CANDIDATES:
        decision = summary["decisions"][form]
        for ref, value in decision["ratios"].items():
            close(value, ratio(form, ref))
        for ref, values in decision["seed_ratios"].items():
            for seed, value in zip(study.SEEDS, values):
                close(value, ratio(form, ref, (seed,)))
        means = summary["per_task"]
        expected = {
            "complete_finite_grid": True,
            "positive_assay": sum(
                means["full_gelu"][t]["mean"] <= 0.98 * raw["zero_mse"][t] for t in study.TASKS
            )
            >= 2,
            "seventy_percent_reduction": True,
            "two_percent_over_each_control": all(ratio(form, r) <= 0.98 for r in study.CONTROLS),
            "every_seed_beats_each_control": all(
                ratio(form, r, (s,)) < 1 for r in study.CONTROLS for s in study.SEEDS
            ),
            "per_task_regression_cap": all(
                means[form][t]["mean"] <= 1.05 * min(means[r][t]["mean"] for r in study.CONTROLS)
                for t in study.TASKS
            ),
            "full_reference_proxy": max(ratio(form, "full_gelu"), ratio(form, "full_swiglu"))
            <= 1.01,
            "update_time_cap": summary["resources"][form]["median_update_ms"]
            <= 1.25 * summary["resources"]["narrow_gelu"]["median_update_ms"],
        }
        assert decision["gates"] == expected
        assert decision["verdict"] == (
            "EARNS_CONFIRMATION" if all(expected.values()) else "REJECTED_AT_THIS_BUDGET"
        )
    assert all(sha(p) == h for p, h in before["source_hashes"].items())
    write_json(
        ROOT / "result.json",
        {
            "status": "PASS",
            "research_goal_achieved": False,
            "runs": 336,
            "updates": 100800,
            "presentations": 25804800,
            "exact_checkpoint_scores": checked,
            "exact_initializations": 336,
            "selected_rates_verified": 168,
            "data_regenerated_exactly": True,
            "raw_result_sha256": sha(ROOT / "raw_result.json"),
            "summary": summary,
            "zero_mse": raw["zero_mse"],
            "protocol_sha256": raw["protocol_sha256"],
        },
    )
    columns = [
        "task",
        "form",
        "seed",
        "rate",
        "parameters",
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
    started = time.perf_counter()
    try:
        run()
        status["status"] = "PASS"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        status.update(
            elapsed_seconds=time.perf_counter() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(ROOT / "audit_process.json", status, exclusive=False)
