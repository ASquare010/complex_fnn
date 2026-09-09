"""Independent weighted statistics, CPU solutions/scores and parity reconstruction."""

import itertools
import json
import math
import os
import statistics as stats
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.input_basis_fit_v1.source.study import make_model

ROOT = Path("results/affine_falsification_recovery_v1")
OLD = Path("results/input_basis_fit_v1")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS = (17, 29, 43)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def close(a, b, tolerance=1e-12):
    assert math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance), (a, b)


def independent_statistics(x, y, counts):
    gram = torch.zeros(385, 385, dtype=torch.float64)
    cross = torch.zeros(385, 384, dtype=torch.float64)
    # Different chunk size and square-root weighting from the GPU producer.
    for start in range(0, 65536, 2048):
        xx = torch.cat(
            (
                x[start : start + 2048].double(),
                torch.ones(min(2048, 65536 - start), 1, dtype=torch.float64),
            ),
            1,
        )
        weight = counts[start : start + 2048].double().sqrt()[:, None]
        xx = xx * weight
        yy = y[start : start + 2048].double() * weight
        gram += xx.T @ xx
        cross += xx.T @ yy
    return gram, cross


def negative_target(data, task):
    q = -data["x"][69632:, data["permutation"]]
    i = torch.arange(384)
    a, b, c, d = (q[:, (i + offset) % 384] for offset in range(4))
    if task == "smooth":
        raw = (2 * a).sin() + b**2 + (c * 0.5).exp() - d
    elif task == "oscillatory":
        raw = (a * 3).sin() * (b * 2).cos() + 0.5 * (c * 4 + d).sin()
    elif task == "multiplicative":
        raw = a * b + 2 * b * c * d + a**2 * d
    else:
        raw = (a + b).relu() - 0.7 * (c - d).abs() + torch.where(a > 0, b, c)
    return raw @ data["rotation"] / data["scales"][task]


def run():
    before = read(ROOT / "before.json")
    assert all(sha(p) == h for p, h in {**before["source_hashes"], **before["anchors"]}.items())
    assert read(ROOT / "solve_process.json")["status"] == "PASS"
    assert all(
        sha(p) == m["sha256"] and Path(p).stat().st_size == m["bytes"]
        for p, m in before["preserved_h096"].items()
    )
    raw = read(ROOT / "raw_result.json")
    prior = read(OLD / "raw_result.json")
    assert sha(OLD / "data.pt") == raw["data_sha256"]
    data = torch.load(OLD / "data.pt", map_location="cpu", weights_only=True)
    expected_x = (
        2 * torch.rand(73728, 384, generator=torch.Generator().manual_seed(9844)) - 1
    ) * math.sqrt(3)
    assert torch.equal(data["x"], expected_x)
    del expected_x
    rows = raw["baselines"]
    assert len(rows) == 24 and {(r["task"], r["seed"], r["kind"]) for r in rows} == set(
        itertools.product(TASKS, SEEDS, ("linear", "affine"))
    )
    score_checks = 0
    for seed, task in itertools.product(SEEDS, TASKS):
        indices = torch.randint(
            65536, (300, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        assert torch.equal(indices, data["streams"][seed])
        counts = torch.bincount(indices.flatten(), minlength=65536)
        gram, cross = independent_statistics(data["x"], data["targets"][task], counts)
        for row in [r for r in rows if r["task"] == task and r["seed"] == seed]:
            assert sha(row["tensor_file"]) == row["tensor_sha256"]
            payload = torch.load(row["tensor_file"], map_location="cpu", weights_only=True)
            assert torch.equal(payload["counts"], counts)
            assert row["unique_samples"] == int((counts > 0).sum())
            n = 384 if row["kind"] == "linear" else 385
            assert row["parameters"] == n * 384 and row["sample_presentations"] == 76800
            g, c = gram[:n, :n], cross[:n]
            for actual, expected in ((g, payload["gram"]), (c, payload["cross"])):
                assert (actual - expected).norm() / expected.norm() < 1e-10
            eigenvalues = torch.linalg.eigvalsh(g)
            assert eigenvalues[0] > 0 and eigenvalues[-1] / eigenvalues[0] < 100
            coef = torch.linalg.solve(g, c)
            torch.testing.assert_close(coef, payload["coefficients"], atol=1e-9, rtol=1e-9)
            assert (g @ payload["coefficients"] - c).norm() / c.norm() < 1e-10
            for split, start, stop in (("select", 65536, 69632), ("report", 69632, 73728)):
                for precision, dtype, tol in (
                    ("fp64", torch.float64, 1e-10),
                    ("fp32", torch.float32, 1e-6),
                ):
                    x = data["x"][start:stop].to(dtype)
                    if n == 385:
                        x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
                    prediction = x @ payload["coefficients"].to(dtype)
                    y = data["targets"][task][start:stop].to(dtype)
                    mse = (prediction - y).square().sum().item() / y.numel()
                    close(mse, row[f"{split}_{precision}"], tol)
                    saved = payload[f"{split}_{precision}"]
                    close(
                        (saved - y).square().sum().item() / y.numel(),
                        row[f"{split}_{precision}"],
                        tol,
                    )
                    score_checks += 1
        mean = cross[-1] / 76800
        expected = (data["targets"][task][69632:].double() - mean).square().mean().item()
        row = next(r for r in raw["constant_baselines"] if r["task"] == task and r["seed"] == seed)
        close(expected, row["report_mse"], 1e-10)
    selected = [r for r in prior["rows"] if r["selected"]]
    parity_lookup = {(r["task"], r["form"], r["seed"]): r for r in raw["parity"]}
    assert len(raw["parity"]) == len(parity_lookup) == 168
    for task in TASKS:
        x = data["x"][69632:].cuda()
        plus, minus = data["targets"][task][69632:].cuda(), negative_target(data, task).cuda()
        yo, ye = (plus - minus) / 2, (plus + minus) / 2
        for old in [r for r in selected if r["task"] == task]:
            assert sha(old["checkpoint"]) == old["checkpoint_sha256"]
            model = make_model(old["form"], old["seed"]).cuda()
            model.load_state_dict(
                torch.load(old["checkpoint"], map_location="cpu", weights_only=True)["model"]
            )
            with torch.inference_mode():
                p = torch.cat([model(batch) for batch in x.split(256)])
                m = torch.cat([model(-batch) for batch in x.split(256)])
                po, pe = (p - m) / 2, (p + m) / 2
                odd = F.mse_loss(po.double(), yo.double()).item()
                even = F.mse_loss(pe.double(), ye.double()).item()
                total = (
                    F.mse_loss(p.double(), plus.double()).item()
                    + F.mse_loss(m.double(), minus.double()).item()
                ) / 2
            observed = parity_lookup[task, old["form"], old["seed"]]
            for key, value in (("odd_mse", odd), ("even_mse", even), ("antithetic_mse", total)):
                close(value, observed[key], 1e-10)
            close(odd + even, total, 1e-6)
            del model, p, m, po, pe
    pairs = list((ROOT / "parity").glob("*.json"))
    assert len(pairs) == 12
    for path in pairs:
        row = read(path)
        tensor_path = path.with_name(row["tensor_file"])
        assert sha(tensor_path) == row["tensor_sha256"]
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        torch.testing.assert_close(tensors["actual"], tensors["expected"], atol=1e-5, rtol=1e-5)
    lookup = {(r["task"], r["seed"], r["kind"]): r for r in rows}
    summary = raw["summary"]

    def ratio(form, seeds=SEEDS):
        values = [
            r["reporting_mse"] / lookup[r["task"], r["seed"], "affine"]["report_fp32"]
            for r in selected
            if r["form"] == form and r["seed"] in seeds
        ]
        return math.exp(sum(map(math.log, values)) / len(values))

    for form, value in summary["neural_ratios_vs_affine"].items():
        close(value, ratio(form))
    for kind, task in itertools.product(("linear", "affine"), TASKS):
        values = [lookup[task, s, kind]["report_fp32"] for s in SEEDS]
        row = summary["per_task"][kind][task]
        assert row["values"] == values
        for key, fn in (
            ("mean", stats.mean),
            ("median", stats.median),
            ("variance", stats.variance),
            ("sample_sd", stats.stdev),
        ):
            close(row[key], fn(values))
    for seed, value in zip(SEEDS, summary["even_lead_seed_ratios"]):
        close(value, ratio("duplicate", (seed,)))
    for task, value in summary["even_lead_task_ratios"].items():
        mean = stats.mean(
            r["reporting_mse"] for r in selected if r["form"] == "duplicate" and r["task"] == task
        )
        close(value, mean / summary["per_task"]["affine"][task]["mean"])
    expected_gates = {
        "two_percent_over_affine": ratio("duplicate") <= 0.98,
        "every_seed_better": all(ratio("duplicate", (s,)) < 1 for s in SEEDS),
        "per_task_cap": all(v <= 1.05 for v in summary["even_lead_task_ratios"].values()),
    }
    assert expected_gates == summary["gates"]
    assert summary["verdict"] == (
        "SURVIVES_AFFINE_CHECK"
        if all(expected_gates.values())
        else "NO_DEMONSTRATED_MATERIAL_NONLINEAR_BENEFIT_AT_THIS_BUDGET"
    )
    assert score_checks == 96
    assert all(sha(p) == h for p, h in {**before["source_hashes"], **before["anchors"]}.items())
    write_json(
        ROOT / "result.json",
        {
            "status": "PASS",
            "least_squares_fits": 24,
            "neural_optimizer_updates": 0,
            "independent_cpu_score_checks": score_checks,
            "parity_models_recomputed": 168,
            "saved_parity_pairs_rechecked": 12,
            "summary": summary,
            "baselines": rows,
            "constant_baselines": raw["constant_baselines"],
            "parity": raw["parity"],
            "raw_result_sha256": sha(ROOT / "raw_result.json"),
            "research_goal_achieved": False,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


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
