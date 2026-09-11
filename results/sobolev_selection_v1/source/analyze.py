"""Post-screen decisions and complete compact endpoint export; no Torch dependency."""

import csv
import gzip
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/sobolev_selection_v1")
METHODS = (
    "random",
    "weight_norm",
    "fluctuation",
    "variance_pivot",
    "value_greedy",
    "sobolev_greedy",
    "value_select_sobolev_fit",
    "sobolev_select_value_fit",
)


def describe(values):
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "sample_variance": st.variance(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def geomean(values):
    return math.exp(st.mean(math.log(v) for v in values))


def run():
    result = json.loads((ROOT / "result.json").read_text())
    rows = result["rows"]
    assert result["status"] == "COMPLETE" and len(rows) == 432
    lookup = {(r["teacher"], r["layer"], r["seed"], r["budget"], r["method"]): r for r in rows}
    groups = []
    for teacher in ("gelu", "swiglu"):
        for budget in ("primary", "half", "three_quarter"):
            for method in METHODS:
                peers = [
                    r
                    for r in rows
                    if (r["teacher"], r["budget"], r["method"]) == (teacher, budget, method)
                ]
                groups.append(
                    {
                        "teacher": teacher,
                        "budget": budget,
                        "method": method,
                        "parameters": peers[0]["parameters"],
                        "parameter_reduction": peers[0]["parameter_reduction"],
                        **{
                            key: describe([r[key] for r in peers])
                            for key in (
                                "value_nmse",
                                "derivative_relative_mse",
                                "inference_ms",
                                "peak_allocated_bytes",
                                "peak_reserved_bytes",
                                "pipeline_peak_cuda_bytes",
                                "max_value_sample_nmse",
                                "max_derivative_sample_relative_mse",
                            )
                        },
                        "seed_means": {
                            str(seed): {
                                key: st.mean(r[key] for r in peers if r["seed"] == seed)
                                for key in ("value_nmse", "derivative_relative_mse")
                            }
                            for seed in (71, 83, 97)
                        },
                        "depth_means": {
                            str(layer): {
                                key: st.mean(r[key] for r in peers if r["layer"] == layer)
                                for key in ("value_nmse", "derivative_relative_mse")
                            }
                            for layer in (0, 3, 7)
                        },
                    }
                )
    group_index = {(r["teacher"], r["budget"], r["method"]): r for r in groups}
    components, allocation = {}, {}
    for budget in ("primary", "half", "three_quarter"):
        component_teachers, allocation_methods = {}, {}
        for teacher in ("gelu", "swiglu"):
            peers = [
                r
                for r in rows
                if (r["teacher"], r["budget"], r["method"]) == (teacher, budget, "sobolev_greedy")
            ]
            ratios = {
                key: geomean(
                    [
                        r[key] / lookup[teacher, r["layer"], r["seed"], budget, "value_greedy"][key]
                        for r in peers
                    ]
                )
                for key in ("value_nmse", "derivative_relative_mse")
            }
            seeds = {
                str(seed): {
                    key: geomean(
                        [
                            r[key] / lookup[teacher, r["layer"], seed, budget, "value_greedy"][key]
                            for r in peers
                            if r["seed"] == seed
                        ]
                    )
                    for key in ratios
                }
                for seed in (71, 83, 97)
            }
            candidate = group_index[teacher, budget, "sobolev_greedy"]
            dominators = [
                method
                for method in ("random", "weight_norm", "fluctuation")
                if all(
                    group_index[teacher, budget, method][key]["mean"] <= candidate[key]["mean"]
                    for key in ratios
                )
                and any(
                    group_index[teacher, budget, method][key]["mean"] < candidate[key]["mean"]
                    for key in ratios
                )
            ]
            gates = {
                "five_percent_derivative_gain": ratios["derivative_relative_mse"] <= 0.95,
                "value_regression_at_most_one_percent": ratios["value_nmse"] <= 1.01,
                "every_seed_meets_both": all(
                    v["value_nmse"] <= 1.01 and v["derivative_relative_mse"] <= 0.95
                    for v in seeds.values()
                ),
                "not_dominated_by_static_selector": not dominators,
                "finite": all(r["all_finite"] for r in peers),
            }
            component_teachers[teacher] = {
                "ratios_to_value_greedy": ratios,
                "seed_ratios": seeds,
                "static_dominators": dominators,
                "gates": gates,
                "passes": all(gates.values()),
            }
        components[budget] = {
            "teachers": component_teachers,
            "verdict": "COMPONENT_QUALIFIES"
            if all(v["passes"] for v in component_teachers.values())
            else "FIXED_COMPONENT_REJECTED",
        }
        for method in ("value_greedy", "sobolev_greedy"):
            teachers = {}
            for teacher in ("gelu", "swiglu"):
                candidate = group_index[teacher, budget, method]
                reference = [
                    c["teacher_resources"] for c in result["cases"] if c["teacher"] == teacher
                ]
                memory_ratio = candidate["peak_allocated_bytes"]["mean"] / st.mean(
                    r["peak_allocated_bytes"] for r in reference
                )
                latency_ratio = candidate["inference_ms"]["mean"] / st.mean(
                    r["inference_ms"] for r in reference
                )
                gates = {
                    "mean_value_error_below_five_percent": candidate["value_nmse"]["mean"] <= 0.05,
                    "mean_derivative_error_below_ten_percent": candidate["derivative_relative_mse"][
                        "mean"
                    ]
                    <= 0.1,
                    "every_seed_meets_both_absolute_errors": all(
                        s["value_nmse"] <= 0.05 and s["derivative_relative_mse"] <= 0.1
                        for s in candidate["seed_means"].values()
                    ),
                    "ten_percent_inference_memory_saving": memory_ratio <= 0.9,
                    "latency_within_ten_percent": latency_ratio <= 1.1,
                    "at_least_twenty_percent_fewer_parameters": candidate["parameter_reduction"]
                    >= 0.2,
                }
                teachers[teacher] = {
                    "gates": gates,
                    "memory_ratio_to_full": memory_ratio,
                    "latency_ratio_to_full": latency_ratio,
                    "passes": all(gates.values()),
                }
            component_ok = (
                method == "value_greedy" or components[budget]["verdict"] == "COMPONENT_QUALIFIES"
            )
            passed = component_ok and all(t["passes"] for t in teachers.values())
            allocation_methods[method] = {
                "teachers": teachers,
                "component_eligible": component_ok,
                "verdict": "EARNS_FRESH_LANGUAGE_INSERTION" if passed else "NO_LANGUAGE_ALLOCATION",
                "original_seventy_percent_parameter_target": group_index["gelu", budget, method][
                    "parameter_reduction"
                ]
                >= 0.7
                and group_index["swiglu", budget, method]["parameter_reduction"] >= 0.7,
            }
        allocation[budget] = allocation_methods
    summary = {
        "groups": groups,
        "component_decisions": components,
        "allocation": allocation,
        "elapsed_seconds": result["elapsed_seconds"],
        "neural_updates": 0,
        "preprocessing_seconds": sum(c["statistic_seconds"] for c in result["cases"]),
        "all_selection_seconds": sum(c["all_selectors_seconds"] for c in result["cases"]),
        "all_readout_seconds": sum(r["readout_seconds"] for r in rows),
        "preprocessing_peak_cuda_bytes": max(
            c["preprocessing_peak_cuda_bytes"] for c in result["cases"]
        ),
        "pipeline_peak_cuda_bytes": max(r["pipeline_peak_cuda_bytes"] for r in rows),
        "unit": "Nine depth-by-sampling cells per teacher; reused development inputs, one teacher training seed",
        "breakthrough": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = (
        "teacher",
        "layer",
        "seed",
        "budget",
        "method",
        "hidden",
        "parameters",
        "parameter_reduction",
        "value_nmse",
        "derivative_relative_mse",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "pipeline_peak_cuda_bytes",
        "inference_ms",
        "readout_seconds",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: r[k] for k in fields} for r in rows)
    (ROOT / "metrics.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    for group in groups:
        print(
            group["teacher"],
            group["budget"],
            group["method"],
            "value",
            round(group["value_nmse"]["mean"], 6),
            "jvp",
            round(group["derivative_relative_mse"]["mean"], 6),
        )
    print(json.dumps({"components": components, "allocation": allocation}, indent=2))


if __name__ == "__main__":
    run()
