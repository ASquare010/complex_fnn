"""H107 frozen paired post-training FFN compression experiment."""

import copy
import gc
import json
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from results.affine_residual_fit_v1.source.model import (
    SPECS,
    Student,
    fold_raw,
    initialize_hidden,
    normalization,
    normalized_teacher,
    ridge_readout,
)
from results.affine_residual_fit_v1.source.qualification import run as qualify
from src.core.config import ModelConfig
from src.core.data import load_manifest
from src.core.ffn_capture import capture_ffn_pairs
from src.core.function_fitting import fit_regression, regression_score
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/affine_residual_fit_v1")
SEEDS, LAYERS, RATES = (71, 83, 97), (0, 3, 7), (0.001, 0.003)
TRAIN, SELECT = 32768, 36864


def read(path):
    return json.loads(Path(path).read_text())


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def scores(model, x, y):
    return {
        "selection_mse": regression_score(model, x[TRAIN:SELECT], y[TRAIN:SELECT]),
        "reporting_mse": regression_score(model, x[SELECT:], y[SELECT:]),
    }


@torch.no_grad()
def inference(model, x):
    model.cuda().eval()
    fixed = x[:256].cuda()
    for _ in range(10):
        model(fixed)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            model(fixed)
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start) / 20)
    model.cpu()
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def save_state(model, path):
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)
    return {"path": path.as_posix(), "sha256": sha256(path)}


def run():
    assert not (ROOT / "protocol.json").exists(), "No silent reruns of a frozen experiment"
    capacity = Path("results/affine_residual_capacity_v1")
    assert (capacity / "audit_exit.txt").read_text() == "0"
    prior = read("results/real_subspace_v1/result.json")
    manifest = load_manifest(Path("data/wikitext2_v1"))
    teacher_paths = {
        kind: Path(f"results/ungated_duration_v1/runs/full_{kind}_seed17/checkpoint.pt")
        for kind in ("gelu", "swiglu")
    }
    files = [
        *(ROOT / "source").glob("*.py"),
        Path("research/affine_residual_fit_plan.md"),
        *[
            Path("src/core") / name
            for name in (
                "ffn_capture.py",
                "function_fitting.py",
                "transformer.py",
                "config.py",
                "data.py",
                "reproducibility.py",
            )
        ],
        Path("src/dense_ffn/__init__.py"),
    ]
    hashes = {p.as_posix(): sha256(p) for p in files}
    write_json(
        ROOT / "protocol.json",
        {
            "sources": hashes,
            "environment": environment(),
            "provenance": provenance(),
            "teachers": {
                k: {"path": p.as_posix(), "sha256": sha256(p)} for k, p in teacher_paths.items()
            },
            "data_files": manifest["files"],
            "prerequisites": {
                str(capacity / f): sha256(capacity / f) for f in ("result.json", "audit.json")
            },
            "seeds": SEEDS,
            "layers": LAYERS,
            "rates": RATES,
            "forms": SPECS,
            "steps": 600,
            "batch": 256,
            "neural_updates": 172800,
            "resource_profile_updates": 540,
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(path, path)
    checks = qualify()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification: {len(checks)} checks passed", flush=True)
    tokens = torch.from_numpy(np.load("data/wikitext2_v1/train.npy").astype("int64"))
    windows = tokens[: len(tokens) // 128 * 128].reshape(-1, 128)
    excluded = {v for c in prior["collections"] for v in c["window_ids"]}
    available = torch.tensor([i for i in range(len(windows)) if i not in excluded])
    chosen = available[
        torch.randperm(len(available), generator=torch.Generator().manual_seed(17001))[:960]
    ]
    assert len(set(chosen.tolist())) == 960 and not excluded.intersection(chosen.tolist())
    rows, collections, diagnostics = [], [], []
    begin = time.perf_counter()
    for kind, teacher_path in teacher_paths.items():
        checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
        model = Transformer(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        for seed_index, seed in enumerate(SEEDS):
            ids = chosen[seed_index * 320 : (seed_index + 1) * 320]
            model.cuda().eval()
            clean()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            pairs = capture_ffn_pairs(model, windows[ids], LAYERS)
            torch.cuda.synchronize()
            collection = {
                "teacher": kind,
                "seed": seed,
                "window_ids": ids.tolist(),
                "seconds": time.perf_counter() - start,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            collections.append(collection)
            model.cpu()
            clean()
            for layer in LAYERS:
                label = f"{kind}_l{layer}_s{seed}"
                folder = ROOT / "pairs" / label
                folder.mkdir(parents=True)
                pair = pairs.pop(layer)
                norm = normalization(pair["x"][:TRAIN], pair["y"][:TRAIN])
                x = ((pair["x"].double() - norm["mx"]) / norm["sx"]).float()
                y = ((pair["y"].double() - norm["my"]) / norm["sy"]).float()
                stream = torch.randint(
                    TRAIN, (600, 256), generator=torch.Generator().manual_seed(19000 + seed)
                )
                torch.save(
                    {**pair, "window_ids": ids, "normalization": norm, "stream": stream},
                    folder / "data.pt",
                )
                teacher = copy.deepcopy(model.blocks[layer].ffn).cpu().eval()
                full = normalized_teacher(teacher, kind, norm).cuda()
                clean()
                full_record = fit_regression(full, x[:TRAIN], y[:TRAIN], stream[:30], 0, warmup=10)
                full_record.update(scores(full, x, y))
                full.cpu()
                full_record["raw_inference"] = inference(copy.deepcopy(teacher), pair["x"])
                del full
                clean()
                # Affine-only diagnostic, fitted on training rows with an intercept.
                design = torch.cat(
                    (x[:TRAIN].double(), torch.ones(TRAIN, 1, dtype=torch.float64)), 1
                )
                affine = torch.linalg.lstsq(
                    design, y[:TRAIN].double(), driver="gelsd", rcond=1e-12
                ).solution
                report_design = torch.cat(
                    (x[SELECT:].double(), torch.ones(len(x) - SELECT, 1, dtype=torch.float64)), 1
                )
                diagnostic = {
                    "teacher": kind,
                    "layer": layer,
                    "seed": seed,
                    "data_sha256": sha256(folder / "data.pt"),
                    "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
                    "full_profile": full_record,
                    "affine_only_reporting_mse": (report_design @ affine - y[SELECT:].double())
                    .square()
                    .mean()
                    .item(),
                }
                diagnostics.append(diagnostic)
                write_json(folder / "metrics.json", diagnostic)
                forms = list(SPECS)
                forms = forms[seed_index:] + forms[:seed_index]
                for form in forms:
                    run_dir = folder / form
                    run_dir.mkdir()
                    torch.manual_seed(20000 + seed)
                    initial = Student(form)
                    start = time.perf_counter()
                    bank_rows = initialize_hidden(initial, teacher, norm, seed)
                    initial, ridge = ridge_readout(initial, x[:TRAIN], y[:TRAIN])
                    init_seconds = time.perf_counter() - start
                    initial_file = save_state(initial, run_dir / "initial.pt")
                    initial.cuda()
                    endpoints = [
                        {"rate": 0, "steps": 0, "checkpoint": initial_file, **scores(initial, x, y)}
                    ]
                    initial.cpu()
                    clean()
                    costs = []
                    for rate in RATES:
                        fitted = copy.deepcopy(initial).cuda()
                        clean()
                        record = fit_regression(fitted, x[:TRAIN], y[:TRAIN], stream, rate)
                        endpoint = {
                            "rate": rate,
                            "steps": 600,
                            **scores(fitted, x, y),
                            "checkpoint": save_state(fitted, run_dir / f"rate{rate}.pt"),
                        }
                        endpoints.append(endpoint)
                        costs.append(record)
                        write_json(run_dir / f"rate{rate}.json", {**endpoint, **record})
                        del fitted
                        clean()
                    selected = min(range(3), key=lambda i: (endpoints[i]["selection_mse"], i))
                    chosen_state = torch.load(
                        endpoints[selected]["checkpoint"]["path"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    initial.load_state_dict(chosen_state)
                    raw = fold_raw(initial, norm)
                    with torch.no_grad():
                        reference = (
                            initial(x[SELECT : SELECT + 256]) * norm["sy"].float()
                            + norm["my"].float()
                        )
                        exported = raw(pair["x"][SELECT : SELECT + 256])
                        torch.testing.assert_close(
                            exported, reference, atol=2e-4 * norm["sy"].item(), rtol=2e-4
                        )
                    raw_file = save_state(raw, run_dir / "raw.pt")
                    latency = inference(raw, pair["x"])
                    count = sum(p.numel() for p in initial.parameters())
                    row = {
                        "teacher": kind,
                        "layer": layer,
                        "seed": seed,
                        "form": form,
                        "parameters": count,
                        "parameter_reduction": 1 - count / diagnostic["teacher_parameters"],
                        "activation_parameters": int(form == "prelu"),
                        "initialization_seconds": init_seconds,
                        "ridge": ridge,
                        "teacher_rows": bank_rows.tolist(),
                        "endpoints": endpoints,
                        "selected_index": selected,
                        "reporting_mse": endpoints[selected]["reporting_mse"],
                        "raw_checkpoint": raw_file,
                        "raw_inference": latency,
                        "raw_fold_max_abs": (exported - reference).abs().max().item(),
                        "peak_allocated_bytes": max(c["peak_allocated_bytes"] for c in costs),
                        "peak_reserved_bytes": max(c["peak_reserved_bytes"] for c in costs),
                        "pipeline_peak_cuda_bytes": max(
                            collection["peak_cuda_bytes"],
                            *(c["peak_allocated_bytes"] for c in costs),
                        ),
                        "optimizer_bytes": costs[0]["optimizer_bytes"],
                        "parameter_bytes": costs[0]["parameter_bytes"],
                        "median_update_ms": statistics.mean(c["median_update_ms"] for c in costs),
                        "median_forward_ms": statistics.mean(c["median_forward_ms"] for c in costs),
                        "median_backward_ms": statistics.mean(
                            c["median_backward_ms"] for c in costs
                        ),
                        "all_finite": all(c["all_finite"] for c in costs),
                    }
                    rows.append(row)
                    write_json(run_dir / "summary.json", row)
                    print(
                        json.dumps(
                            {
                                "completed": len(rows),
                                "pair": label,
                                "form": form,
                                "mse": row["reporting_mse"],
                                "selected": selected,
                                "elapsed_seconds": time.perf_counter() - begin,
                            }
                        ),
                        flush=True,
                    )
                    del initial, raw, chosen_state, costs, reference, exported
                    clean()
                del x, y, pair, teacher, design, report_design, affine, stream
            del pairs
        del model
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "collections": collections,
            "diagnostics": diagnostics,
            "elapsed_seconds": time.perf_counter() - begin,
            "neural_updates": 172800,
            "resource_profile_updates": 540,
            "breakthrough": False,
        },
    )
    print("All 288 fits and 144 selections completed", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        raise
