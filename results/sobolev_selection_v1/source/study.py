"""H108 frozen zero-SGD neuron selection and directional-sensitivity screen."""

import gc
import json
import statistics as st
import time
import zipfile
from pathlib import Path

import torch

from results.sobolev_selection_v1.source.features import directions, export, statistics, value_jvp
from results.sobolev_selection_v1.source.qualification import run as qualify
from results.sobolev_selection_v1.source.selection import METHODS, RIDGE, WIDTHS, all_orders, solve
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.dense_ffn import DenseFFN

ROOT = Path("results/sobolev_selection_v1")
PRIOR = Path("results/affine_residual_fit_v1")


def read(path):
    return json.loads(Path(path).read_text())


def clean():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def predictions(model, x, probes):
    model.cuda().eval()
    values, derivatives = [], []
    for direction in probes:
        y_rows, j_rows = [], []
        for start in range(0, len(x), 512):
            y, j = value_jvp(
                model, x[start : start + 512].cuda(), direction[start : start + 512].cuda()
            )
            assert bool(torch.isfinite(y).all()) and bool(torch.isfinite(j).all())
            y_rows.append(y.cpu())
            j_rows.append(j.cpu())
        values.append(torch.cat(y_rows))
        derivatives.append(torch.cat(j_rows))
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
    model.cpu()
    return values[0], torch.stack(derivatives)


@torch.no_grad()
def profile(model, x):
    model.cuda().eval()
    fixed = x[:256].cuda()
    for _ in range(10):
        model(fixed)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            model(fixed)
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start) / 20)
    result = {
        "inference_ms": st.median(samples),
        "inference_samples_ms": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }
    model.cpu()
    return result


def errors(value, derivative, reference, target_derivative, scale_y):
    error = (value.double() - reference.double()).square().mean(1)
    error_j = (derivative.double() - target_derivative.double()).square().mean(-1)
    target_energy = target_derivative.double().square().mean(-1)
    return {
        "value_nmse": error.mean().item() / scale_y**2,
        "derivative_relative_mse": error_j.mean().item() / target_energy.mean().item(),
        "max_value_sample_nmse": error.max().item() / scale_y**2,
        "max_derivative_sample_relative_mse": (error_j / target_energy.clamp_min(1e-12))
        .max()
        .item(),
        "derivative_target_mean_square": target_energy.mean().item(),
        "all_finite": True,
    }


def run():
    assert not (ROOT / "protocol.json").exists(), "Preserve existing/partial experiments"
    assert read("results/verification/affine_residual_final_v1.json")["status"] == "PASS"
    prior = read(PRIOR / "result.json")
    assert read(PRIOR / "audit.json")["passed"]
    sources = [
        *(ROOT / "source").glob("*.py"),
        Path("research/sobolev_selection_plan.md"),
        Path("src/dense_ffn/__init__.py"),
        Path("src/core/reproducibility.py"),
    ]
    hashes = {p.as_posix(): sha256(p) for p in sources}
    teachers = {
        kind: Path(f"results/ungated_duration_v1/runs/full_{kind}_seed17/checkpoint.pt")
        for kind in ("gelu", "swiglu")
    }
    write_json(
        ROOT / "protocol.json",
        {
            "sources": hashes,
            "environment": environment(),
            "provenance": provenance(),
            "prior_result_sha256": sha256(PRIOR / "result.json"),
            "prior_audit_sha256": sha256(PRIOR / "audit.json"),
            "teachers": {
                k: {"path": p.as_posix(), "sha256": sha256(p)} for k, p in teachers.items()
            },
            "methods": METHODS,
            "widths": WIDTHS,
            "ridge": RIDGE,
            "derivative_weight_fraction": 0.1,
            "calibration_rows": 8192,
            "reporting_rows": 4096,
            "neural_updates": 0,
            "allocator": "malloc",
            "pythonhashseed": 107,
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(path, path)
    checks = qualify()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification passed: {len(checks)} groups", flush=True)
    rows, cases = [], []
    beginning = time.perf_counter()
    for kind, teacher_path in teachers.items():
        full_state = torch.load(teacher_path, map_location="cpu", weights_only=True)["model"]
        for seed in (71, 83, 97):
            collection = next(
                c for c in prior["collections"] if (c["teacher"], c["seed"]) == (kind, seed)
            )
            for layer in (0, 3, 7):
                label = f"{kind}_l{layer}_s{seed}"
                original = PRIOR / "pairs" / label / "data.pt"
                prior_case = next(
                    c
                    for c in prior["diagnostics"]
                    if (c["teacher"], c["layer"], c["seed"]) == (kind, layer, seed)
                )
                assert sha256(original) == prior_case["data_sha256"]
                source = torch.load(original, map_location="cpu", weights_only=True)
                calibration, reporting = source["x"][:8192].clone(), source["x"][-4096:].clone()
                del source
                prefix = f"blocks.{layer}.ffn."
                weights = {
                    k.removeprefix(prefix): v for k, v in full_state.items() if k.startswith(prefix)
                }
                teacher = DenseFFN(
                    weights["up.weight"].shape[1], weights["up.weight"].shape[0], kind
                )
                teacher.load_state_dict(weights, strict=True)
                folder = ROOT / "cases" / label
                folder.mkdir(parents=True)
                clean()
                start = time.perf_counter()
                state = statistics(teacher, calibration, seed)
                statistic_seconds = time.perf_counter() - start
                start = time.perf_counter()
                orders, traces = all_orders(state, teacher, seed, max(WIDTHS[kind].values()))
                selection_seconds = time.perf_counter() - start
                torch.save(
                    {"statistics": state, "orders": orders, "traces": traces},
                    folder / "statistics.pt",
                )
                probes = [
                    directions(len(reporting), reporting.shape[1], offset + seed, state["sx"])
                    for offset in (29000, 30000)
                ]
                reference, target_j = predictions(teacher, reporting, probes)
                resources = profile(teacher, reporting)
                torch.save({"value": reference, "derivative": target_j}, folder / "reference.pt")
                case = {
                    "teacher": kind,
                    "layer": layer,
                    "seed": seed,
                    "input_data_sha256": sha256(original),
                    "statistics_sha256": sha256(folder / "statistics.pt"),
                    "reference_sha256": sha256(folder / "reference.pt"),
                    "statistic_seconds": statistic_seconds,
                    "all_selectors_seconds": selection_seconds,
                    "preprocessing_peak_cuda_bytes": state["peak_cuda_bytes"],
                    "prerequisite_capture_peak_cuda_bytes": collection["peak_cuda_bytes"],
                    "scale_y": state["scale_y"],
                    "beta": state["beta"],
                    "teacher_resources": resources,
                }
                cases.append(case)
                write_json(folder / "metrics.json", case)
                print(
                    json.dumps(
                        {
                            "statistics": label,
                            "seconds": statistic_seconds,
                            "selection_seconds": selection_seconds,
                            "beta": state["beta"],
                        }
                    ),
                    flush=True,
                )
                for budget, hidden in WIDTHS[kind].items():
                    for method, (order_name, use_sobolev) in METHODS.items():
                        selected = orders[order_name][:hidden]
                        start = time.perf_counter()
                        coefficient, solve_diagnostic = solve(state, selected, use_sobolev)
                        readout_seconds = time.perf_counter() - start
                        student = export(teacher, state, selected, coefficient).float()
                        path = folder / f"{budget}_{method}.pt"
                        torch.save(
                            {
                                "state_dict": student.state_dict(),
                                "selected": selected,
                                "coefficient": coefficient,
                                "activation": kind,
                                "hidden": hidden,
                                "width": reporting.shape[1],
                            },
                            path,
                        )
                        value, derivative = predictions(student, reporting, probes)
                        measurement = errors(
                            value, derivative, reference, target_j, state["scale_y"]
                        )
                        resource = profile(student, reporting)
                        row = {
                            "teacher": kind,
                            "layer": layer,
                            "seed": seed,
                            "budget": budget,
                            "method": method,
                            "hidden": hidden,
                            "selected": selected.tolist(),
                            "readout_seconds": readout_seconds,
                            "solve": solve_diagnostic,
                            "checkpoint": {"path": path.as_posix(), "sha256": sha256(path)},
                            **measurement,
                            **resource,
                            "parameter_reduction": 1
                            - resource["parameters"] / resources["parameters"],
                            "pipeline_peak_cuda_bytes": max(
                                state["peak_cuda_bytes"],
                                collection["peak_cuda_bytes"],
                                resource["peak_allocated_bytes"],
                            ),
                        }
                        rows.append(row)
                        write_json(folder / f"{budget}_{method}.json", row)
                        del student, value, derivative, coefficient
                        clean()
                    print(
                        json.dumps(
                            {
                                "completed": label,
                                "budget": budget,
                                "endpoints": len(rows),
                                "elapsed_seconds": time.perf_counter() - beginning,
                            }
                        ),
                        flush=True,
                    )
                del (
                    calibration,
                    reporting,
                    teacher,
                    state,
                    orders,
                    traces,
                    reference,
                    target_j,
                    probes,
                )
        del full_state
    assert len(rows) == 432
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "cases": cases,
            "elapsed_seconds": time.perf_counter() - beginning,
            "neural_updates": 0,
            "breakthrough": False,
        },
    )
    print("All 432 compressed endpoints complete, zero SGD updates", flush=True)


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
