"""H104 fresh-data fitting; common trainer, equal spectral controls and full accounting."""

import gc
import json
import time
import zipfile
from pathlib import Path

import torch

from results.nonlinear_residual_v1.source.model import TASKS, target_features
from results.spectral_discovery_v1.source.moments import (
    centered_moment,
    leading_subspace,
    weights_from_labels,
)
from results.spectral_discovery_v1.source.study import tensor_sha
from results.spectral_fitting_v1.source.model import FORMS, FeatureFFN
from results.spectral_fitting_v1.source.qualification import run_checks
from src.core.function_fitting import fit_regression, regression_score
from src.core.reproducibility import environment, provenance, sha256, write_json

ROOT = Path("results/spectral_fitting_v1")
PLAN = Path("research/spectral_fitting_plan.md")
TRAIN, SELECT = 65536, 69632


def gen(seed):
    return torch.Generator().manual_seed(seed)


def make_data(seed):
    x = torch.randn(73728, 384, generator=gen(10800 + seed))
    teacher = torch.linalg.qr(
        torch.randn(384, 384, dtype=torch.float64, generator=gen(10900 + seed))
    ).Q[:, :16]
    rotation = torch.linalg.qr(
        torch.randn(384, 384, dtype=torch.float64, generator=gen(11000 + seed))
    ).Q[:16]
    q = x.double() @ teacher
    targets, scales = {}, {}
    for task in TASKS:
        raw = target_features(q, task) @ rotation
        scales[task] = raw[:TRAIN].std(0, correction=0)
        targets[task] = (raw / scales[task]).float()
    return {
        "x": x,
        "targets": targets,
        "teacher": teacher,
        "rotation": rotation,
        "scales": scales,
        "stream": torch.randint(TRAIN, (300, 256), generator=gen(12000 + seed)),
    }


def infer_bases(x, y, seed, task):
    bases = {"random": torch.linalg.qr(torch.randn(384, 32, generator=gen(13000 + seed))).Q}
    records = {}
    for mode in ("raw", "bounded"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        weights = weights_from_labels(y[:TRAIN], mode)
        matrix = centered_moment(x[:TRAIN], weights)
        basis, eigenvalues = leading_subspace(matrix, 32)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        bases[mode] = basis.float()
        records[mode] = {
            "seconds": seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "matrix_sha256": tensor_sha(matrix),
            "basis_sha256": tensor_sha(bases[mode]),
            "training_label_rows": TRAIN,
            "input_sha256": tensor_sha(x[:TRAIN]),
            "label_sha256": tensor_sha(y[:TRAIN]),
        }
    folder = ROOT / "initializers" / f"{task}_s{seed}"
    folder.mkdir(parents=True)
    torch.save(bases, folder / "bases.pt")
    write_json(folder / "metrics.json", records)
    return bases, records


def run():
    assert not (ROOT / "protocol.json").exists(), "Use fresh output directory"
    prior = json.loads(Path("results/spectral_discovery_v1/result.json").read_text())
    assert all(r["earns_fitting"] for r in prior["decisions"].values())
    paths = [
        *(ROOT / "source").glob("*.py"),
        PLAN,
        Path("src/core/function_fitting.py"),
        Path("results/spectral_discovery_v1/source/moments.py"),
        Path("results/spectral_discovery_v1/source/study.py"),
        Path("results/nonlinear_residual_v1/source/model.py"),
    ]
    sources = {p.as_posix(): sha256(p) for p in paths}
    write_json(
        ROOT / "protocol.json",
        {
            "sources": sources,
            "environment": environment(),
            "provenance": provenance(),
            "forms": FORMS,
            "seeds": [17, 29, 43],
            "rates": [0.001, 0.003],
            "tasks": TASKS,
            "steps": 300,
            "batch": 256,
            "plan_sha256": sha256(PLAN),
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path)
    checks = run_checks()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification passed: {len(checks)} checks", flush=True)
    rows = []
    started = time.perf_counter()
    for seed_index, seed in enumerate((17, 29, 43)):
        data = make_data(seed)
        (ROOT / "data").mkdir(exist_ok=True)
        torch.save(data, ROOT / "data" / f"s{seed}.pt")
        x = data["x"]
        for task in TASKS:
            y = data["targets"][task]
            bases, preprocessing = infer_bases(x, y, seed, task)
            forms = FORMS[seed_index:] + FORMS[:seed_index]
            for form in forms:
                for rate in (0.001, 0.003):
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = FeatureFFN(form, seed, bases).cuda()
                    initial = {n: tensor_sha(p) for n, p in model.state_dict().items()}
                    result = fit_regression(model, x[:TRAIN], y[:TRAIN], data["stream"], rate)
                    result["selection_mse"] = regression_score(
                        model, x[TRAIN:SELECT], y[TRAIN:SELECT]
                    )
                    result["reporting_mse"] = regression_score(model, x[SELECT:], y[SELECT:])
                    label = f"{task}_{form}_s{seed}_lr{int(rate * 1e6)}"
                    folder = ROOT / "cells" / label
                    folder.mkdir(parents=True)
                    checkpoint = {
                        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
                        "form": form,
                        "seed": seed,
                        "task": task,
                        "rate": rate,
                        "step": 300,
                    }
                    torch.save(checkpoint, folder / "checkpoint.pt")
                    history = result.pop("history")
                    with (folder / "history.jsonl").open("w") as stream:
                        for step in history:
                            stream.write(json.dumps(step) + "\n")
                    mode = form.split("_")[-1]
                    result.update(
                        {
                            "form": form,
                            "seed": seed,
                            "task": task,
                            "rate": rate,
                            "label": label,
                            **model.counts(),
                            "initial_hashes": initial,
                            "state_hashes": {
                                n: tensor_sha(p) for n, p in checkpoint["model"].items()
                            },
                            "stream_sha256": tensor_sha(data["stream"]),
                            "preprocessing": preprocessing.get(
                                mode, {"seconds": 0, "peak_cuda_bytes": 0, "training_label_rows": 0}
                            ),
                            "checkpoint_sha256": sha256(folder / "checkpoint.pt"),
                            "history_sha256": sha256(folder / "history.jsonl"),
                            "checkpoint": str(folder / "checkpoint.pt"),
                            "training_presentations": 300 * 256,
                            "matrix_forward_flops_per_example": 2
                            * model.counts()["matrix_parameters"],
                        }
                    )
                    result["selected"] = False
                    write_json(folder / "metrics.json", result)
                    rows.append(result)
                    print(
                        json.dumps(
                            {
                                "completed": label,
                                "selection": result["selection_mse"],
                                "reporting": result["reporting_mse"],
                            }
                        ),
                        flush=True,
                    )
                    del model, checkpoint, result
            for form in FORMS:
                peers = [r for r in rows if (r["form"], r["seed"], r["task"]) == (form, seed, task)]
                min(peers, key=lambda r: (r["selection_mse"], r["rate"]))["selected"] = True
        del data, x, y, bases
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "elapsed_seconds": time.perf_counter() - started,
            "runs": len(rows),
            "optimizer_updates": 300 * len(rows),
            "breakthrough": False,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        raise
