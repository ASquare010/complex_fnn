"""H105 frozen local projection screen, without any neural optimization."""

import copy
import gc
import json
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from results.real_subspace_v1.source.model import METHODS, RANKS, ProjectedFFN, estimate, maps
from results.real_subspace_v1.source.qualification import run as qualify
from results.spectral_discovery_v1.source.study import tensor_sha
from src.core.config import ModelConfig
from src.core.data import load_manifest
from src.core.ffn_capture import capture_ffn_pairs
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/real_subspace_v1")
LAYERS = (0, 3, 7)
TRAIN = 32768


def read(path):
    return json.loads(Path(path).read_text())


@torch.no_grad()
def evaluate(model, x, reference=None):
    model.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    for part in x.split(512):
        value = model(part.cuda())
        assert torch.isfinite(value).all()
        outputs.append(value.cpu())
    output = torch.cat(outputs)
    peak = torch.cuda.max_memory_allocated()
    fixed = x[:256].cuda()
    for _ in range(10):
        model(fixed)
    torch.cuda.synchronize()
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            model(fixed)
        torch.cuda.synchronize()
        timings.append(1000 * (time.perf_counter() - start) / 20)
    record = {
        "inference_ms": statistics.median(timings),
        "inference_samples_ms": timings,
        "peak_allocated_bytes": peak,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "finite": True,
    }
    if reference is not None:
        record["mse"] = (output.double() - reference.double()).square().mean().item()
    model.cpu()
    return output, record


def run():
    assert not (ROOT / "protocol.json").exists(), "Completed/partial output is immutable"
    cache = Path("data/wikitext2_v1")
    manifest = load_manifest(cache)
    teacher_paths = {
        a: Path(f"results/ungated_duration_v1/runs/full_{a}_seed17/checkpoint.pt")
        for a in ("gelu", "swiglu")
    }
    sources = [
        *(ROOT / "source").glob("*.py"),
        Path("research/real_subspace_plan.md"),
        Path("src/core/ffn_capture.py"),
        Path("src/core/transformer.py"),
        Path("src/dense_ffn/__init__.py"),
        Path("src/core/config.py"),
        Path("results/spectral_discovery_v1/source/study.py"),
    ]
    hashes = {p.as_posix(): sha256(p) for p in sources}
    write_json(
        ROOT / "protocol.json",
        {
            "sources": hashes,
            "environment": environment(),
            "provenance": provenance(),
            "teachers": {
                k: {"path": str(p), "sha256": sha256(p)} for k, p in teacher_paths.items()
            },
            "data_files": manifest["files"],
            "seeds": [17, 29, 43],
            "layers": LAYERS,
            "ranks": RANKS,
            "methods": METHODS,
            "neural_updates": 0,
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(path, path)
    checks = qualify()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification passed: {len(checks)}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    tokens = torch.from_numpy(np.load(cache / "train.npy").astype("int64"))
    windows = tokens[: len(tokens) // 128 * 128].reshape(-1, 128)
    all_rows, collections, diagnostics = [], [], []
    start_study = time.perf_counter()
    for activation, path in teacher_paths.items():
        settings = read(path.parent / "metrics.json")["model"]
        model = Transformer(ModelConfig(**settings))
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        for seed in (17, 29, 43):
            ids = torch.randperm(
                len(windows), generator=torch.Generator().manual_seed(16000 + seed)
            )[:320]
            sampled = windows[ids]
            assert len(torch.unique(ids)) == 320
            model.cuda().eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            pairs = capture_ffn_pairs(model, sampled, LAYERS)
            torch.cuda.synchronize()
            collection = {
                "teacher": activation,
                "seed": seed,
                "seconds": time.perf_counter() - start,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "window_ids": ids.tolist(),
                "window_sha256": tensor_sha(sampled),
                "calibration_tokens": TRAIN,
                "reporting_tokens": 8192,
            }
            collections.append(collection)
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            for layer in LAYERS:
                folder = ROOT / "pairs" / f"{activation}_l{layer}_s{seed}"
                folder.mkdir(parents=True)
                pair = pairs.pop(layer)
                torch.save({**pair, "window_ids": ids}, folder / "data.pt")
                teacher = copy.deepcopy(model.blocks[layer].ffn).cpu().eval()
                before = time.perf_counter()
                state = estimate(pair["x"][:TRAIN], pair["y"][:TRAIN], teacher, seed)
                estimation_seconds = time.perf_counter() - before
                torch.save(state, folder / "statistics.pt")
                reference, teacher_resources = evaluate(teacher, pair["x"][TRAIN:])
                centered = reference.double() - state["my"]
                variance = centered.square().mean().item()
                affine_prediction = (
                    state["my"] + (pair["x"][TRAIN:].double() - state["mx"]) @ state["affine"]
                )
                diagnostic = {
                    "teacher": activation,
                    "layer": layer,
                    "seed": seed,
                    "estimation_seconds_all_methods": estimation_seconds,
                    "training_pair_sha256": {k: tensor_sha(v[:TRAIN]) for k, v in pair.items()},
                    "reporting_pair_sha256": {k: tensor_sha(v[TRAIN:]) for k, v in pair.items()},
                    "data_sha256": sha256(folder / "data.pt"),
                    "statistics_sha256": sha256(folder / "statistics.pt"),
                    "variance": variance,
                    "covariance_floored": state["floored"],
                    "condition_ratio": (
                        state["eigenvalues"][-1] / state["eigenvalues"][0].clamp_min(1e-30)
                    ).item(),
                    "affine_normalized_mse": (affine_prediction - reference.double())
                    .square()
                    .mean()
                    .item()
                    / variance,
                    "bf16_target_normalized_drift": (
                        pair["y"][TRAIN:].double() - reference.double()
                    )
                    .square()
                    .mean()
                    .item()
                    / variance,
                    "teacher_resources": teacher_resources,
                }
                diagnostics.append(diagnostic)
                write_json(folder / "metrics.json", diagnostic)
                teacher.double()
                for rank in RANKS:
                    for method in METHODS:
                        a, e, c = maps(state, method, rank)
                        student = ProjectedFFN(teacher, a, e, c, state["mx"], state["my"]).float()
                        output, resources = evaluate(student, pair["x"][TRAIN:], reference)
                        # Small CPU FP64 formula witness at each real fitted projector.
                        probe = pair["x"][TRAIN : TRAIN + 5].double()
                        explicit = (
                            state["my"]
                            + (teacher(state["mx"] + (probe - state["mx"]) @ a @ e) - state["my"])
                            @ c
                            @ c.T
                        )
                        exact = ProjectedFFN(teacher, a, e, c, state["mx"], state["my"])(probe)
                        torch.testing.assert_close(explicit, exact, atol=1e-9, rtol=1e-9)
                        row = {
                            "teacher": activation,
                            "layer": layer,
                            "seed": seed,
                            "rank": rank,
                            "method": method,
                            **resources,
                            "normalized_mse": resources["mse"] / variance,
                            "pipeline_peak_cuda_bytes": max(
                                collection["peak_cuda_bytes"], resources["peak_allocated_bytes"]
                            ),
                            "fold_error_fp64": (explicit - exact).abs().max().item(),
                            "output_sha256": tensor_sha(output),
                            "parameter_reduction": 1
                            - resources["parameters"] / teacher_resources["parameters"],
                        }
                        all_rows.append(row)
                        del student, output, explicit, exact
                print(
                    json.dumps(
                        {
                            "completed": str(folder),
                            "comparisons": len(all_rows),
                            "affine_nmse": diagnostic["affine_normalized_mse"],
                        }
                    ),
                    flush=True,
                )
                del pair, state, teacher, reference
            del pairs
        del model
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": all_rows,
            "collections": collections,
            "diagnostics": diagnostics,
            "elapsed_seconds": time.perf_counter() - start_study,
            "neural_updates": 0,
            "breakthrough": False,
        },
    )
    print("All 378 comparisons complete", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        raise
