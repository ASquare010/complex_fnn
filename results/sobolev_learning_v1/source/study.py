"""H109: paired fresh-input value learning from four frozen H108 initializers."""

import copy
import gc
import json
import statistics as st
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from results.sobolev_learning_v1.source.model import (
    LAYERS,
    METHODS,
    RATES,
    SEEDS,
    SELECT,
    TRAIN,
    TrainingAdapter,
    endpoint_scores,
    from_checkpoint,
    teacher_targets,
)
from results.sobolev_learning_v1.source.qualification import run as qualify
from results.sobolev_selection_v1.source.study import profile
from src.core.config import ModelConfig
from src.core.data import load_manifest
from src.core.ffn_capture import capture_ffn_pairs
from src.core.function_fitting import fit_regression
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/sobolev_learning_v1")
PRIOR = Path("results/sobolev_selection_v1")


def read(path):
    return json.loads(Path(path).read_text())


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def checkpoint(model, path):
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "width": 384,
            "hidden": model.up.out_features,
            "activation": model.activation,
        },
        path,
    )
    return {"path": path.as_posix(), "sha256": sha256(Path(path))}


def run():
    assert not (ROOT / "protocol.json").exists(), "Do not replace a frozen or partial experiment"
    assert read("results/verification/sobolev_selection_final_v1.json")["status"] == "PASS"
    prior = read(PRIOR / "result.json")
    previous = read("results/affine_residual_fit_v1/result.json")
    older = read("results/real_subspace_v1/result.json")
    old_protocol = read(PRIOR / "protocol.json")
    sources = {p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")}
    for path in [
        "research/sobolev_learning_plan.md",
        "src/core/ffn_capture.py",
        "src/core/function_fitting.py",
        "src/core/transformer.py",
        "src/core/config.py",
        "src/core/data.py",
    ]:
        sources[path] = sha256(Path(path))
    for path, digest in old_protocol["sources"].items():
        assert sha256(Path(path)) == digest
        sources[path] = digest
    write_json(
        ROOT / "protocol.json",
        {
            "sources": sources,
            "environment": environment(),
            "provenance": provenance(),
            "teachers": old_protocol["teachers"],
            "prior_result_sha256": sha256(PRIOR / "result.json"),
            "data_files": load_manifest(Path("data/wikitext2_v1"))["files"],
            "methods": METHODS,
            "rates": RATES,
            "seeds": SEEDS,
            "layers": LAYERS,
            "steps": 600,
            "batch": 256,
            "neural_updates": 86400,
            "profile_updates": 540,
            "window_seed": 18001,
            "training_stream_offset": 31000,
            "reporting_probe_offsets": [32000, 33000],
            "allocator": "malloc",
            "pythonhashseed": 107,
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path)
    checks = qualify()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification passed: {len(checks)} groups", flush=True)
    tokens = torch.from_numpy(np.load("data/wikitext2_v1/train.npy").astype("int64"))
    windows = tokens[: len(tokens) // 128 * 128].reshape(-1, 128)
    excluded = {
        i for source in (older, previous) for c in source["collections"] for i in c["window_ids"]
    }
    available = torch.tensor([i for i in range(len(windows)) if i not in excluded])
    chosen = available[
        torch.randperm(len(available), generator=torch.Generator().manual_seed(18001))[:960]
    ]
    assert len(set(chosen.tolist())) == 960 and not excluded.intersection(chosen.tolist())
    write_json(
        ROOT / "data_manifest.json",
        {
            "window_ids": chosen.tolist(),
            "excluded_window_ids": sorted(excluded),
            "fresh_disjoint_windows": True,
        },
    )
    rows, cases, collections = [], [], []
    beginning = time.perf_counter()
    for kind, teacher_record in old_protocol["teachers"].items():
        assert sha256(Path(teacher_record["path"])) == teacher_record["sha256"]
        saved = torch.load(teacher_record["path"], map_location="cpu", weights_only=True)
        decoder = Transformer(ModelConfig(**saved["model_config"]))
        decoder.load_state_dict(saved["model"])
        del saved
        for seed_index, seed in enumerate(SEEDS):
            ids = chosen[seed_index * 320 : (seed_index + 1) * 320]
            decoder.cuda().eval()
            clean()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            pairs = capture_ffn_pairs(decoder, windows[ids], LAYERS)
            torch.cuda.synchronize()
            collection = {
                "teacher": kind,
                "seed": seed,
                "window_ids": ids.tolist(),
                "seconds": time.perf_counter() - begin,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            collections.append(collection)
            decoder.cpu()
            clean()
            for layer in LAYERS:
                label = f"{kind}_l{layer}_s{seed}"
                folder = ROOT / "pairs" / label
                folder.mkdir(parents=True)
                x = pairs.pop(layer)["x"]
                prior_case = next(
                    c
                    for c in prior["cases"]
                    if (c["teacher"], c["layer"], c["seed"]) == (kind, layer, seed)
                )
                stats_path = PRIOR / "cases" / label / "statistics.pt"
                assert sha256(stats_path) == prior_case["statistics_sha256"]
                norm = torch.load(stats_path, map_location="cpu", weights_only=True)["statistics"]
                sy, sx = norm["scale_y"], norm["sx"]
                del norm
                teacher = copy.deepcopy(decoder.blocks[layer].ffn).cpu().eval()
                begin = time.perf_counter()
                y, target_j, probes, target_peak = teacher_targets(teacher, x, sx, seed)
                target_seconds = time.perf_counter() - begin
                stream = torch.randint(
                    TRAIN, (600, 256), generator=torch.Generator().manual_seed(31000 + seed)
                )
                torch.save(
                    {
                        "x": x,
                        "y": y,
                        "target_j": target_j,
                        "sx": sx,
                        "sy": sy,
                        "window_ids": ids,
                        "stream": stream,
                    },
                    folder / "data.pt",
                )
                normalized = y / sy
                full = TrainingAdapter(copy.deepcopy(teacher), sy).cuda()
                clean()
                full_profile = fit_regression(
                    full, x[:TRAIN], normalized[:TRAIN], stream[:30], 0, warmup=10
                )
                for name, value in teacher.state_dict().items():
                    torch.testing.assert_close(
                        value, full.raw.state_dict()[name].cpu(), atol=0, rtol=0
                    )
                del full
                clean()
                full_inference = profile(teacher, x[SELECT:])
                case = {
                    "teacher": kind,
                    "layer": layer,
                    "seed": seed,
                    "data_sha256": sha256(folder / "data.pt"),
                    "sy": sy,
                    "target_seconds": target_seconds,
                    "target_peak_cuda_bytes": target_peak,
                    "fresh_capture_peak_cuda_bytes": collection["peak_cuda_bytes"],
                    "prior_capture_peak_cuda_bytes": prior_case[
                        "prerequisite_capture_peak_cuda_bytes"
                    ],
                    "prior_preprocessing_peak_cuda_bytes": prior_case[
                        "preprocessing_peak_cuda_bytes"
                    ],
                    "prior_statistics_seconds": prior_case["statistic_seconds"],
                    "prior_all_selectors_seconds": prior_case["all_selectors_seconds"],
                    "full_profile": full_profile,
                    "full_inference": full_inference,
                }
                cases.append(case)
                write_json(folder / "metrics.json", case)
                methods = METHODS[seed_index:] + METHODS[:seed_index]
                for method in methods:
                    run_dir = folder / method
                    run_dir.mkdir()
                    prior_row = next(
                        r
                        for r in prior["rows"]
                        if (r["teacher"], r["layer"], r["seed"], r["budget"], r["method"])
                        == (kind, layer, seed, "primary", method)
                    )
                    original = prior_row["checkpoint"]
                    assert sha256(Path(original["path"])) == original["sha256"]
                    initial = from_checkpoint(
                        torch.load(original["path"], map_location="cpu", weights_only=True)
                    )
                    init_file = checkpoint(initial, run_dir / "initial.pt")
                    endpoints = [
                        {
                            "rate": 0,
                            "steps": 0,
                            "checkpoint": init_file,
                            **endpoint_scores(initial, x, y, target_j, probes, sy),
                        }
                    ]
                    costs = []
                    for rate in RATES:
                        fitted = TrainingAdapter(copy.deepcopy(initial), sy).cuda()
                        clean()
                        record = fit_regression(fitted, x[:TRAIN], normalized[:TRAIN], stream, rate)
                        endpoint = {
                            "rate": rate,
                            "steps": 600,
                            "checkpoint": checkpoint(fitted.raw, run_dir / f"rate{rate}.pt"),
                            **endpoint_scores(fitted.raw, x, y, target_j, probes, sy),
                        }
                        endpoints.append(endpoint)
                        costs.append(record)
                        write_json(run_dir / f"rate{rate}.json", {**endpoint, **record})
                        del fitted
                        clean()
                    selected = min(range(3), key=lambda i: (endpoints[i]["selection_mse"], i))
                    deployed = from_checkpoint(
                        torch.load(
                            endpoints[selected]["checkpoint"]["path"],
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    inference = profile(deployed, x[SELECT:])
                    row = {
                        "teacher": kind,
                        "layer": layer,
                        "seed": seed,
                        "method": method,
                        "initialization": original,
                        "prior_readout_seconds": prior_row["readout_seconds"],
                        "endpoints": endpoints,
                        "selected_index": selected,
                        **{
                            k: endpoints[selected][k]
                            for k in ("selection_mse", "reporting_mse", "derivative_relative_mse")
                        },
                        "inference": inference,
                        "parameters": inference["parameters"],
                        "parameter_reduction": 1
                        - inference["parameters"] / full_inference["parameters"],
                        "peak_allocated_bytes": max(c["peak_allocated_bytes"] for c in costs),
                        "peak_reserved_bytes": max(c["peak_reserved_bytes"] for c in costs),
                        "pipeline_peak_cuda_bytes": max(
                            collection["peak_cuda_bytes"],
                            target_peak,
                            prior_row["pipeline_peak_cuda_bytes"],
                            *(c["peak_allocated_bytes"] for c in costs),
                            inference["peak_allocated_bytes"],
                        ),
                        "optimizer_bytes": costs[0]["optimizer_bytes"],
                        "parameter_bytes": costs[0]["parameter_bytes"],
                        "all_finite": all(c["all_finite"] for c in costs),
                        **{
                            f"median_{k}": st.mean(c[f"median_{k}"] for c in costs)
                            for k in ("update_ms", "forward_ms", "backward_ms")
                        },
                    }
                    rows.append(row)
                    write_json(run_dir / "summary.json", row)
                    print(
                        json.dumps(
                            {
                                "completed": len(rows),
                                "pair": label,
                                "method": method,
                                "mse": row["reporting_mse"],
                                "jvp": row["derivative_relative_mse"],
                                "selected": selected,
                                "elapsed_seconds": time.perf_counter() - beginning,
                            }
                        ),
                        flush=True,
                    )
                    del initial, deployed, costs
                    clean()
                del x, y, target_j, probes, teacher, normalized, stream
            del pairs
        del decoder
    assert len(rows) == 72
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "cases": cases,
            "collections": collections,
            "elapsed_seconds": time.perf_counter() - beginning,
            "neural_updates": 86400,
            "profile_updates": 540,
            "breakthrough": False,
        },
    )
    print("All 144 fits and 72 selections completed", flush=True)


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
