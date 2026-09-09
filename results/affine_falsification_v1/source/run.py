"""Matched-sample affine fits and antithetic parity audit; no neural updates."""

import hashlib
import json
import math
import os
import statistics as stats
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import (
    read,
    record_observation,
    sha,
    write_json,
)
from results.input_basis_fit_v1.source.study import make_model
from results.neuron_geometry_v1.source.study import save_tensor
from src.core.reproducibility import environment

ROOT = Path("results/affine_falsification_v1")
OLD = Path("results/input_basis_fit_v1")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS = (17, 29, 43)
TRAIN, SELECT_END, TOTAL, WIDTH = 65536, 69632, 73728, 384
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def matrix_error(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def stats_matrices(x, y, counts, device="cuda"):
    gram = torch.zeros(385, 385, dtype=torch.float64, device=device)
    cross = torch.zeros(385, 384, dtype=torch.float64, device=device)
    for start in range(0, len(x), 4096):
        xx = x[start : start + 4096].to(device, torch.float64)
        yy = y[start : start + 4096].to(device, torch.float64)
        w = counts[start : start + 4096].to(device, torch.float64).unsqueeze(1)
        xx = torch.cat((xx, torch.ones_like(xx[:, :1])), 1)
        gram += xx.T @ (w * xx)
        cross += xx.T @ (w * yy)
    return (gram + gram.T) / 2, cross


@torch.inference_mode()
def predictions(coefficients, x, precision, device):
    coef = coefficients.to(device, precision)
    xx = x.to(device, precision)
    if coef.shape[0] == 385:
        xx = torch.cat((xx, torch.ones_like(xx[:, :1])), 1)
    return xx @ coef


@torch.inference_mode()
def score(coefficients, x, y, precision, device):
    pred = predictions(coefficients, x, precision, device)
    error = pred - y.to(device, precision)
    return error.square().sum().item() / y.numel(), pred.cpu()


def analytic(x, task, permutation, rotation, scale):
    a, b, c, d = (torch.roll(x[:, permutation], -i, -1) for i in range(4))
    if task == "smooth":
        raw = torch.sin(2 * a) + b.square() + torch.exp(0.5 * c) - d
    elif task == "oscillatory":
        raw = torch.sin(3 * a) * torch.cos(2 * b) + 0.5 * torch.sin(4 * c + d)
    elif task == "multiplicative":
        raw = a * b + 2 * b * c * d + a.square() * d
    elif task == "piecewise":
        raw = F.relu(a + b) - 0.7 * (c - d).abs() + torch.where(a > 0, b, c)
    else:
        raise ValueError(task)
    return (raw @ rotation) / scale


def summarize(baselines, prior_rows):
    lookup = {(r["task"], r["seed"], r["kind"]): r for r in baselines}
    selected = [r for r in prior_rows if r["selected"]]
    forms = sorted({r["form"] for r in selected})

    def ratio(form, seeds=SEEDS):
        values = [
            r["reporting_mse"] / lookup[r["task"], r["seed"], "affine"]["report_fp32"]
            for r in selected
            if r["form"] == form and r["seed"] in seeds
        ]
        return math.exp(stats.mean(math.log(v) for v in values))

    per_task = {}
    for kind in ("linear", "affine"):
        per_task[kind] = {}
        for task in TASKS:
            values = [lookup[task, s, kind]["report_fp32"] for s in SEEDS]
            per_task[kind][task] = {
                "mean": stats.mean(values),
                "median": stats.median(values),
                "variance": stats.variance(values),
                "sample_sd": stats.stdev(values),
                "values": values,
            }
    task_ratios = {}
    for task in TASKS:
        candidate = stats.mean(
            r["reporting_mse"] for r in selected if r["task"] == task and r["form"] == "duplicate"
        )
        task_ratios[task] = candidate / per_task["affine"][task]["mean"]
    seed_ratios = [ratio("duplicate", (s,)) for s in SEEDS]
    gates = {
        "two_percent_over_affine": ratio("duplicate") <= 0.98,
        "every_seed_better": all(v < 1 for v in seed_ratios),
        "per_task_cap": all(v <= 1.05 for v in task_ratios.values()),
    }
    return {
        "per_task": per_task,
        "neural_ratios_vs_affine": {f: ratio(f) for f in forms},
        "even_lead_seed_ratios": seed_ratios,
        "even_lead_task_ratios": task_ratios,
        "gates": gates,
        "verdict": "SURVIVES_AFFINE_CHECK"
        if all(gates.values())
        else "NO_DEMONSTRATED_MATERIAL_NONLINEAR_BENEFIT_AT_THIS_BUDGET",
    }


def run():
    hashes = dict(read(OLD / "before.json")["source_hashes"])
    for p in [
        *sorted((ROOT / "source").glob("*.py")),
        Path("research/affine_falsification_plan.md"),
    ]:
        hashes[p.as_posix()] = sha(p)
    assert all(sha(p) == h for p, h in hashes.items())
    anchors = {
        p: sha(p)
        for p in (
            "results/input_basis_fit_v1/result.json",
            "results/input_basis_fit_v1/raw_result.json",
            "results/input_basis_fold_v1/summary.json",
        )
    }
    before = {
        "source_hashes": hashes,
        "anchors": anchors,
        "neural_optimizer_updates": 0,
        "least_squares_fits": 24,
        "constant_estimates": 12,
        "repetitions": 1,
    }
    write_json(ROOT / "before.json", before)
    write_json(
        ROOT / "protocol.json",
        {**before, "environment": environment(), "before_sha256": sha(ROOT / "before.json")},
    )
    with (ROOT / "source.zip").open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in {**hashes, **anchors}:
                archive.write(name, name)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.testzip() is None
        assert all(
            hashlib.sha256(archive.read(p)).hexdigest() == h
            for p, h in {**hashes, **anchors}.items()
        )
    for folder in ("fits", "parity"):
        (ROOT / folder).mkdir()
    prior = read(OLD / "raw_result.json")
    assert read(OLD / "result.json")["status"] == "PASS"
    assert sha(OLD / "data.pt") == prior["data_sha256"]
    data = torch.load(OLD / "data.pt", map_location="cpu", weights_only=True)
    baselines, constants, parity = [], [], []
    for seed in SEEDS:
        counts = torch.bincount(data["streams"][seed].flatten(), minlength=TRAIN)
        assert counts.sum() == 76800
        for task in TASKS:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            gram, cross = stats_matrices(data["x"][:TRAIN], data["targets"][task][:TRAIN], counts)
            constant = cross[-1] / counts.sum()
            const_mse = (
                (data["targets"][task][SELECT_END:].double() - constant.cpu())
                .square()
                .mean()
                .item()
            )
            constants.append({"task": task, "seed": seed, "report_mse": const_mse})
            for kind, n in (("linear", 384), ("affine", 385)):
                g, c = gram[:n, :n], cross[:n]
                eigenvalues = torch.linalg.eigvalsh(g)
                condition = (eigenvalues[-1] / eigenvalues[0]).item()
                assert eigenvalues[0] > 0 and condition < 100
                lower, info = torch.linalg.cholesky_ex(g)
                assert info.item() == 0
                coef = torch.cholesky_solve(c, lower)
                residual = matrix_error(g @ coef, c)
                assert residual < 1e-10
                independent = torch.linalg.solve(g.cpu(), c.cpu())
                torch.testing.assert_close(coef.cpu(), independent, atol=1e-9, rtol=1e-9)
                row = {
                    "task": task,
                    "seed": seed,
                    "kind": kind,
                    "parameters": n * 384,
                    "unique_samples": int((counts > 0).sum()),
                    "sample_presentations": 76800,
                    "condition_number": condition,
                    "relative_normal_residual": residual,
                }
                payload = {
                    "coefficients": coef.cpu(),
                    "gram": g.cpu(),
                    "cross": c.cpu(),
                    "counts": counts,
                    "independent_coefficients": independent,
                }
                for split, start, stop in (
                    ("select", TRAIN, SELECT_END),
                    ("report", SELECT_END, TOTAL),
                ):
                    xx, yy = data["x"][start:stop], data["targets"][task][start:stop]
                    for name, dtype in (("fp64", torch.float64), ("fp32", torch.float32)):
                        value, prediction = score(coef, xx, yy, dtype, "cuda")
                        row[f"{split}_{name}"] = value
                        payload[f"{split}_{name}"] = prediction
                path = ROOT / "fits" / f"{task}_{seed}_{kind}.pt"
                save_tensor(path, payload)
                row.update(
                    tensor_file=path.as_posix(),
                    tensor_sha256=sha(path),
                    peak_bytes=torch.cuda.max_memory_allocated(),
                    elapsed_since_task_start=time.perf_counter() - started,
                )
                write_json(path.with_suffix(".json"), row)
                baselines.append(row)
            print(task, seed, "least-squares PASS", flush=True)
    selected = [r for r in prior["rows"] if r["selected"]]
    for task in TASKS:
        x = data["x"][SELECT_END:].cuda()
        plus = data["targets"][task][SELECT_END:].cuda()
        # CPU analytic construction preserves the original target arithmetic policy.
        minus = analytic(
            -data["x"][SELECT_END:],
            task,
            data["permutation"],
            data["rotation"],
            data["scales"][task],
        ).cuda()
        target_odd, target_even = (plus - minus) / 2, (plus + minus) / 2
        for row in [r for r in selected if r["task"] == task]:
            assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
            model = make_model(row["form"], row["seed"]).cuda()
            model.load_state_dict(
                torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"]
            )
            with torch.inference_mode():
                p = torch.cat([model(x[i : i + 256]) for i in range(0, 4096, 256)])
                m = torch.cat([model(-x[i : i + 256]) for i in range(0, 4096, 256)])
                odd, even = (p - m) / 2, (p + m) / 2
                odd_error = (odd.double() - target_odd.double()).square().mean().item()
                even_error = (even.double() - target_even.double()).square().mean().item()
                paired_error = (
                    (p.double() - plus.double()).square().mean().item()
                    + (m.double() - minus.double()).square().mean().item()
                ) / 2
                assert math.isclose(
                    odd_error + even_error, paired_error, abs_tol=1e-6, rel_tol=1e-6
                )
                item = {
                    "task": task,
                    "seed": row["seed"],
                    "form": row["form"],
                    "odd_mse": odd_error,
                    "even_mse": even_error,
                    "antithetic_mse": paired_error,
                }
                if row["form"] == "duplicate":
                    linear = F.linear(x, model.down.weight[:, :384])
                    torch.testing.assert_close(odd, linear, atol=1e-5, rtol=1e-5)
                    record_observation(
                        ROOT / "parity",
                        f"{task}_{row['seed']}",
                        "odd_linear",
                        {"atol_rtol": 1e-5},
                        {"actual": odd[:256].cpu(), "expected": linear[:256].cpu()},
                    )
            parity.append(item)
            del model, p, m, odd, even
        del x, plus, minus, target_odd, target_even
    assert len(baselines) == 24 and len(parity) == 168
    assert all(sha(p) == h for p, h in {**hashes, **anchors}.items())
    write_json(
        ROOT / "raw_result.json",
        {
            "baselines": baselines,
            "constant_baselines": constants,
            "parity": parity,
            "summary": summarize(baselines, prior["rows"]),
            "data_sha256": prior["data_sha256"],
            "protocol_sha256": sha(ROOT / "protocol.json"),
        },
    )
    print(json.dumps(summarize(baselines, prior["rows"]), indent=2), flush=True)


if __name__ == "__main__":
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "solve_process.json", status)
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
        write_json(ROOT / "solve_process.json", status, exclusive=False)
