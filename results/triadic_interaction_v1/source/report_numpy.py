"""Separate NumPy-only post-audit diagnostics after preserved Torch import failures."""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results.triadic_interaction_v1.source.tensor_reader import load

ROOT = Path("results/triadic_interaction_v1")
TASKS = ("quadratic", "cubic", "product", "piecewise")


def read(path):
    return json.loads(Path(path).read_text())


def align(weight, basis):
    w = weight.astype(np.float64)
    squared = (w @ basis) ** 2 / np.sum(w * w, axis=1, keepdims=True)
    energy = squared.sum(1)
    assert np.isfinite(energy).all() and (energy >= -1e-12).all() and (energy <= 1 + 1e-12).all()
    return {
        "mean_subspace_fraction": float(energy.mean()),
        "median_subspace_fraction": float(np.median(energy)),
        "top16_mean_subspace_fraction": float(np.sort(energy)[-16:].mean()),
        "max_squared_cosine_by_teacher_direction": squared.max(0).tolist(),
    }


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_reader(data, rows):
    pairs = 0
    for p in sorted((ROOT / "qualification").glob("*.json")):
        r = read(p)
        if "tensor_file" not in r:
            continue
        file = p.with_name(r["tensor_file"])
        assert sha(file) == r["tensor_sha256"]
        for item in load(file):
            a, b, tol = item["actual"], item["expected"], item["tolerance"]
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            assert (np.abs(a - b) <= tol + tol * np.abs(b)).all()
            pairs += 1
    assert pairs == 177
    np.testing.assert_allclose(data["basis"].T @ data["basis"], np.eye(16), atol=1e-12, rtol=1e-12)
    q = data["x"][:257].astype(np.float64) @ data["basis"]
    features = {
        "quadratic": (q * q - 1) / math.sqrt(2),
        "cubic": (q * q * q - 3 * q) / math.sqrt(6),
        "product": q * np.roll(q, -1, axis=1),
        "piecewise": (np.maximum(q, 0) ** 2 - 0.5 - math.sqrt(2 / math.pi) * q)
        / math.sqrt(1.25 - 2 / math.pi),
    }
    for task, value in features.items():
        expected = value @ (data["rotation"] / data["scales"][task])
        np.testing.assert_allclose(expected, data["targets"][task][:257], atol=5e-6, rtol=5e-6)
    for seed, stream in data["streams"].items():
        checkpoint = load(next(r["checkpoint"] for r in rows if r["seed"] == seed))
        assert (
            hashlib.sha256(np.ascontiguousarray(stream).tobytes()).hexdigest()
            == checkpoint["stream_sha256"]
        )
    validation = {
        "status": "PASS",
        "saved_pairs": pairs,
        "oracle_tasks": 4,
        "stream_hashes": 3,
        "basis_orthogonality_tolerance": 1e-12,
        "scope": "Limited dense ZIP tensor decoder; BF16 decoded exactly to FP32",
    }
    with (ROOT / "reader_validation.json").open("x") as out:
        json.dump(validation, out, indent=2)
        out.write("\n")


def main():
    assert read(ROOT / "audit_process.json")["status"] == "PASS"
    result = read(ROOT / "result.json")
    forms_order = list(result["summary"]["per_task"])
    rows = [r for r in read(ROOT / "raw_result.json")["rows"] if r["selected"]]
    assert sha(ROOT / "data.pt") == read(ROOT / "data_manifest.json")["sha256"]
    data = load(ROOT / "data.pt")
    validate_reader(data, rows)
    basis = data["basis"]
    del data
    alignment, trajectories, diagnostics = [], {}, []
    for row in rows:
        training = read(ROOT / "cells" / row["label"] / "training.json")
        diagnostics.append({"label": row["label"], "samples": training["diagnostics"]})
        trajectories[row["label"]] = {
            key: np.array([h[key] for h in training["history"]]).reshape(30, 10).mean(1).tolist()
            for key in ("loss", "preclip_norm")
        }
        if row["form"] == "feature_aware":
            continue
        assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
        saved = load(row["checkpoint"])["model"]
        for name in ("up", "gate", "third"):
            if name + ".weight" in saved:
                alignment.append(
                    {
                        "task": row["task"],
                        "form": row["form"],
                        "seed": row["seed"],
                        "projection": name,
                        "final": align(saved[name + ".weight"], basis),
                    }
                )
    export = {
        "scope": "Post-audit descriptive diagnostic; teacher basis used only after fitting",
        "limitation": "Alignment is not a causal intervention; maxima also depend on width",
        "alignment": alignment,
        "trajectories_10step_means": trajectories,
        "sampled_diagnostics": diagnostics,
    }
    with (ROOT / "diagnostics.json").open("x") as stream:
        json.dump(export, stream, indent=2)
        stream.write("\n")
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    summary = result["summary"]
    forms = [f for f in forms_order if f != "feature_aware"]
    colors = [
        "#b2b9c5"
        if f.startswith("full")
        else "#d76b23"
        if f == "cp3"
        else "#823e96"
        if f == "bounded_cp3"
        else "#3567a8"
        for f in forms
    ]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task in zip(axes.flat, TASKS, strict=True):
        ax.barh(
            forms,
            [summary["per_task"][f][task]["mean"] for f in forms],
            xerr=[summary["per_task"][f][task]["sample_sd"] for f in forms],
            color=colors,
            capsize=2,
        )
        ax.invert_yaxis()
        ax.axvline(result["zero_mse"][task], ls=":", color="black", label="zero predictor")
        ax.set(title=task, xlabel="Reporting MSE (mean +/- sample SD, three seeds)", xlim=(0, None))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("H100: raw and bounded three-way interactions on hidden directions")
    fig.savefig(plots / "comparison.png", dpi=125)
    plt.close(fig)

    chosen = ("narrow_gelu", "starrelu", "cubic_ridge", "cp2", "cp3", "bounded_cp3")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    for ax, task in zip(axes, TASKS, strict=True):
        means = []
        for form in chosen:
            values = [
                a["final"]["mean_subspace_fraction"]
                for a in alignment
                if a["task"] == task and a["form"] == form and a["projection"] == "up"
            ]
            means.append(np.mean(values))
        ax.barh(chosen, means, color="#3567a8")
        ax.invert_yaxis()
        ax.axvline(16 / 384, color="black", ls=":", lw=0.8)
        ax.set(title=task, xlabel="Final weight energy in target subspace", xlim=(0, 1))
    fig.suptitle(
        "First projection only; dotted line is isotropic expectation, not measured initialization"
    )
    fig.savefig(plots / "alignment.png", dpi=125)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for col, task in enumerate(TASKS):
        for form in chosen:
            subset = [r for r in rows if r["task"] == task and r["form"] == form]
            for row_index, key in enumerate(("loss", "preclip_norm")):
                mean = np.mean([trajectories[r["label"]][key] for r in subset], 0)
                axes[row_index, col].plot(np.arange(1, 31) * 10, mean, label=form)
                axes[row_index, col].set(
                    xlabel="Update",
                    yscale="log",
                    ylabel="Training MSE" if key == "loss" else "Preclip gradient norm",
                )
        axes[0, col].set_title(task)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Selected-rate trajectories: three-seed, ten-step means")
    fig.savefig(plots / "training.png", dpi=125)
    plt.close(fig)
    print(
        json.dumps(
            {
                "alignment_records": len(alignment),
                "diagnostic_runs": len(diagnostics),
                "plots": [p.as_posix() for p in sorted(plots.glob("*.png"))],
            }
        )
    )


if __name__ == "__main__":
    state = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    status_path = ROOT / "report_numpy_process.json"
    with status_path.open("x") as stream:
        json.dump(state, stream, indent=2)
    started = time.perf_counter()
    try:
        main()
        state["status"] = "PASS"
    except BaseException as exc:
        state.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        state.update(
            elapsed_seconds=time.perf_counter() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        status_path.write_text(json.dumps(state, indent=2) + "\n")
