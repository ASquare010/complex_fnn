"""H099 freeze, qualify, fit all declared cells, and report provisional decisions."""

import gc
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

from results.affine_falsification_recovery_v1.source.run import stats_matrices
from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.neuron_geometry_v1.source.study import save_tensor, score
from results.nonlinear_residual_v1.source import fitting, qualification
from results.nonlinear_residual_v1.source.data import SELECT_END, TRAIN, make_data
from results.nonlinear_residual_v1.source.model import CONTROLS, FORMS, TASKS, ResidualTaskFFN
from src.core.reproducibility import environment, provenance

ROOT = Path("results/nonlinear_residual_v1")
SEEDS, RATES = (17, 29, 43), (0.001, 0.003)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def affine_controls(data):
    rows = []
    for seed in SEEDS:
        counts = torch.bincount(data["streams"][seed].flatten(), minlength=TRAIN)
        for task in TASKS:
            gram, cross = stats_matrices(data["x"][:TRAIN], data["targets"][task][:TRAIN], counts)
            eig = torch.linalg.eigvalsh(gram)
            assert eig[0] > 0 and eig[-1] / eig[0] < 100
            coef = torch.cholesky_solve(cross, torch.linalg.cholesky(gram))
            assert (gram @ coef - cross).norm() / cross.norm() < 1e-10
            x = data["x"][SELECT_END:].cuda()
            y = data["targets"][task][SELECT_END:].cuda()
            prediction = torch.cat((x, torch.ones_like(x[:, :1])), 1) @ coef.float().cuda()
            mse = (prediction - y).square().mean().item()
            path = ROOT / "affine" / f"{task}_{seed}.pt"
            save_tensor(
                path, {"gram": gram, "cross": cross, "coefficients": coef, "counts": counts}
            )
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "reporting_mse": mse,
                    "parameters": 147840,
                    "checkpoint": path.as_posix(),
                    "sha256": sha(path),
                }
            )
            del x, y, prediction
    return rows


def summarize(rows, affine, zero):
    selected = [r for r in rows if r["selected"]]
    lookup = {(r["task"], r["form"], r["seed"]): r for r in selected}
    alookup = {(r["task"], r["seed"]): r for r in affine}

    def ratio(form, ref, seeds=SEEDS):
        values = []
        for task in TASKS:
            for seed in seeds:
                denominator = (
                    alookup[task, seed]["reporting_mse"]
                    if ref == "affine"
                    else lookup[task, ref, seed]["reporting_mse"]
                )
                values.append(lookup[task, form, seed]["reporting_mse"] / denominator)
        return math.exp(stats.mean(math.log(v) for v in values))

    per_task, resources = {}, {}
    for form in FORMS:
        per_task[form] = {}
        for task in TASKS:
            values = [lookup[task, form, s]["reporting_mse"] for s in SEEDS]
            per_task[form][task] = {
                "mean": stats.mean(values),
                "median": stats.median(values),
                "variance": stats.variance(values),
                "sample_sd": stats.stdev(values),
                "values": values,
            }
        subset = [r for r in selected if r["form"] == form]
        resources[form] = {
            k: stats.mean(r[k] for r in subset)
            for k in (
                "median_update_ms",
                "median_forward_ms",
                "median_backward_ms",
                "clipped_fraction",
            )
        }
        resources[form]["max_peak_bytes"] = max(r["peak_bytes"] for r in subset)
    pc = all(
        lookup[t, "feature_aware", s]["reporting_mse"] <= 0.05 * zero[t]
        for t in TASKS
        for s in SEEDS
    )
    full_success = {
        f: [t for t in TASKS if per_task[f][t]["mean"] <= 0.5 * zero[t]]
        for f in ("full_gelu", "full_swiglu", "full_relu2")
    }
    neural_assay = any(len(v) >= 3 for v in full_success.values())
    refs = (*CONTROLS, "affine")
    ratios = {
        ref: ratio("learned_mix", ref) for ref in (*refs, "full_gelu", "full_swiglu", "full_relu2")
    }
    seed_ratios = {ref: [ratio("learned_mix", ref, (s,)) for s in SEEDS] for ref in refs}
    task_caps = {}
    for task in TASKS:
        strongest = min(
            *[per_task[f][task]["mean"] for f in CONTROLS],
            stats.mean(alookup[task, s]["reporting_mse"] for s in SEEDS),
        )
        task_caps[task] = per_task["learned_mix"][task]["mean"] / strongest
    gates = {
        "finite_complete_grid": len(rows) == 264 and all(r["all_finite"] for r in rows),
        "feature_aware_positive_control": pc,
        "full_neural_learnability": neural_assay,
        "seventy_percent_reduction": all(
            1 - fitting.COUNTS["learned_mix"] / fitting.COUNTS[f] >= 0.7
            for f in ("full_gelu", "full_swiglu")
        ),
        "two_percent_over_every_control": all(ratios[r] <= 0.98 for r in refs),
        "every_seed_better": all(v < 1 for values in seed_ratios.values() for v in values),
        "per_task_cap": all(v <= 1.05 for v in task_caps.values()),
        "full_reference_proxy": all(ratios[f] <= 1.01 for f in ("full_gelu", "full_swiglu")),
        "update_time_cap": resources["learned_mix"]["median_update_ms"]
        <= 1.25 * resources["narrow_gelu"]["median_update_ms"],
    }
    verdict = "EARNS_CONFIRMATION" if all(gates.values()) else "REJECTED_AT_THIS_BUDGET"
    if not pc or not neural_assay:
        verdict = "INCONCLUSIVE_ASSAY"
    return {
        "per_task": per_task,
        "resources": resources,
        "gates": gates,
        "verdict": verdict,
        "full_neural_successful_tasks": full_success,
        "candidate_ratios": ratios,
        "candidate_seed_ratios": seed_ratios,
        "candidate_task_ratios": task_caps,
        "ratios_vs_affine": {f: ratio(f, "affine") for f in FORMS},
    }


def run():
    hashes = dict(read("results/affine_falsification_recovery_v1/before.json")["source_hashes"])
    for p in [
        *sorted((ROOT / "source").glob("*.py")),
        Path("research/nonlinear_residual_plan.md"),
        Path("results/affine_falsification_recovery_v1/source/run.py"),
    ]:
        hashes[p.as_posix()] = sha(p)
    assert all(sha(p) == h for p, h in hashes.items())
    anchors = {
        p: sha(p)
        for p in (
            "results/affine_falsification_recovery_v1/result.json",
            "results/affine_falsification_recovery_v1/audit_process.json",
        )
    }
    write_json(
        ROOT / "before.json",
        {
            "source_hashes": hashes,
            "anchors": anchors,
            "qualification_checks": 25,
            "neural_runs": 264,
            "neural_updates": 79200,
            "affine_fits": 12,
            "repetitions": 1,
        },
    )
    write_json(
        ROOT / "protocol.json",
        {
            **read(ROOT / "before.json"),
            "environment": environment(),
            "provenance": provenance(),
            "before_sha256": sha(ROOT / "before.json"),
        },
    )
    with (ROOT / "source.zip").open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in {**hashes, **anchors}:
                archive.write(p, p)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.testzip() is None
        assert all(
            hashlib.sha256(archive.read(p)).hexdigest() == h
            for p, h in {**hashes, **anchors}.items()
        )
    for name in ("qualification", "cells", "affine", "selections"):
        (ROOT / name).mkdir()
    print("data_generation", flush=True)
    data = make_data()
    save_tensor(ROOT / "data.pt", data)
    data_hash, protocol_hash = sha(ROOT / "data.pt"), sha(ROOT / "protocol.json")
    write_json(
        ROOT / "data_manifest.json",
        {
            "sha256": data_hash,
            "input_seed": 9860,
            "train_rows": TRAIN,
            "selection_rows": 4096,
            "reporting_rows": 4096,
            "input_width": 384,
            "latent_width": 16,
            "batch": 256,
            "updates_per_run": 300,
        },
    )
    print("qualification", flush=True)
    qualified = qualification.run(data)
    write_json(ROOT / "qualification_process.json", qualified)
    print("affine_controls", flush=True)
    affine = affine_controls(data)
    rows, zero = [], {}
    for ti, task in enumerate(TASKS):
        x, y = data["x"].cuda(), data["targets"][task].cuda()
        zero[task] = data["targets"][task][SELECT_END:].double().square().mean().item()
        for si, seed in enumerate(SEEDS):
            stream = data["streams"][seed].cuda()
            offset = (ti * 3 + si) % len(FORMS)
            for form in FORMS[offset:] + FORMS[:offset]:
                runs = [
                    fitting.train_cell(
                        task, form, seed, rate, x, y, stream, data_hash, protocol_hash
                    )
                    for rate in RATES
                ]
                chosen = min(runs, key=lambda r: r["selection_mse"])["label"]
                write_json(
                    ROOT / "selections" / f"{task}_{form}_{seed}.json", {"selected_label": chosen}
                )
                for result in runs:
                    model = ResidualTaskFFN(form, seed, task).cuda()
                    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
                    model.load_state_dict(state["model"])
                    reporting = score(model, x[SELECT_END:], y[SELECT_END:])
                    row = {
                        k: v
                        for k, v in result.items()
                        if k
                        not in (
                            "history",
                            "diagnostics",
                            "initial_state_sha256",
                            "optimizer_groups",
                        )
                    }
                    row.update(reporting_mse=reporting, selected=row["label"] == chosen)
                    write_json(ROOT / "cells" / row["label"] / "reporting.json", row)
                    rows.append(row)
                    del model, state
            del stream
        del x, y
        gc.collect()
        torch.cuda.empty_cache()
    assert len(rows) == 264
    assert all(sha(p) == h for p, h in {**hashes, **anchors}.items())
    summary = summarize(rows, affine, zero)
    write_json(
        ROOT / "raw_result.json",
        {
            "rows": rows,
            "affine": affine,
            "zero_mse": zero,
            "summary": summary,
            "data_sha256": data_hash,
            "protocol_sha256": protocol_hash,
        },
    )
    print(
        json.dumps({"verdict": summary["verdict"], "gates": summary["gates"]}, indent=2), flush=True
    )


if __name__ == "__main__":
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "study_process.json", status)
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
        write_json(ROOT / "study_process.json", status, exclusive=False)
