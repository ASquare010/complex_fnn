"""All endpoints, paired decisions, learning effects and diagnostic summaries."""

import csv
import gzip
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/sobolev_learning_v1")
METHODS = ("value_greedy", "value_select_sobolev_fit", "sobolev_greedy", "random")
METRICS = ("reporting_mse", "derivative_relative_mse")


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "sample_variance": st.variance(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def geometric(values):
    return math.exp(st.mean(math.log(x) for x in values))


def run():
    result = read(ROOT / "result.json")
    rows, cases = result["rows"], result["cases"]
    assert len(rows) == 72 and result["neural_updates"] == 86400
    lookup = {(r["teacher"], r["layer"], r["seed"], r["method"]): r for r in rows}
    groups = []
    for teacher in ("gelu", "swiglu"):
        for method in METHODS:
            peers = [r for r in rows if (r["teacher"], r["method"]) == (teacher, method)]
            records = [
                read(
                    ROOT
                    / "pairs"
                    / f"{teacher}_l{r['layer']}_s{r['seed']}"
                    / method
                    / f"rate{rate}.json"
                )
                for r in peers
                for rate in (0.001, 0.003)
            ]
            history = [h for record in records for h in record["history"]]
            group = {
                "teacher": teacher,
                "method": method,
                "parameters": peers[0]["parameters"],
                "parameter_reduction": peers[0]["parameter_reduction"],
                **{
                    k: describe([r[k] for r in peers])
                    for k in (
                        *METRICS,
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "pipeline_peak_cuda_bytes",
                        "median_update_ms",
                        "median_forward_ms",
                        "median_backward_ms",
                    )
                },
                "inference": {
                    k: describe([r["inference"][k] for r in peers])
                    for k in ("inference_ms", "peak_allocated_bytes", "peak_reserved_bytes")
                },
                "initial": {k: describe([r["endpoints"][0][k] for r in peers]) for k in METRICS},
                "selected_over_initial_ratios": {
                    k: geometric([r[k] / r["endpoints"][0][k] for r in peers]) for k in METRICS
                },
                "selected_indices": [r["selected_index"] for r in peers],
                "seed_means": {
                    str(seed): {
                        k: st.mean(r[k] for r in peers if r["seed"] == seed) for k in METRICS
                    }
                    for seed in (71, 83, 97)
                },
                "depth_means": {
                    str(layer): {
                        k: st.mean(r[k] for r in peers if r["layer"] == layer) for k in METRICS
                    }
                    for layer in (0, 3, 7)
                },
                "training": {
                    "clip_fraction": sum(h["preclip_norm"] > 1 for h in history) / len(history),
                    "max_preclip_gradient": max(h["preclip_norm"] for h in history),
                    "min_preclip_gradient": min(h["preclip_norm"] for h in history),
                    "all_finite": all(r["all_finite"] for r in records),
                    "steps": len(history),
                },
                "training_curves": {
                    str(rate): [
                        {
                            "step": step,
                            "mean_loss": st.mean(
                                record["history"][step - 1]["loss"]
                                for record in records
                                if record["rate"] == rate
                            ),
                        }
                        for step in (1, 50, 150, 300, 600)
                    ]
                    for rate in (0.001, 0.003)
                },
            }
            groups.append(group)
    index = {(g["teacher"], g["method"]): g for g in groups}
    components = {}
    for method in METHODS[1:3]:
        teachers = {}
        for teacher in ("gelu", "swiglu"):
            peers = [r for r in rows if (r["teacher"], r["method"]) == (teacher, method)]
            comparisons = {}
            for endpoint_index in (None, 0, 1, 2):
                ratios = {}
                for metric in METRICS:
                    values = []
                    for r in peers:
                        control = lookup[teacher, r["layer"], r["seed"], "value_greedy"]
                        a = r if endpoint_index is None else r["endpoints"][endpoint_index]
                        b = (
                            control
                            if endpoint_index is None
                            else control["endpoints"][endpoint_index]
                        )
                        values.append(a[metric] / b[metric])
                    ratios[metric] = geometric(values)
                comparisons["selected" if endpoint_index is None else str(endpoint_index)] = ratios
            seed_ratios = {
                str(seed): {
                    k: geometric(
                        [
                            r[k] / lookup[teacher, r["layer"], r["seed"], "value_greedy"][k]
                            for r in peers
                            if r["seed"] == seed
                        ]
                    )
                    for k in METRICS
                }
                for seed in (71, 83, 97)
            }
            g = index[teacher, method]
            dominators = [
                m
                for m in ("value_greedy", "random")
                if all(index[teacher, m][k]["mean"] <= g[k]["mean"] for k in METRICS)
                and any(index[teacher, m][k]["mean"] < g[k]["mean"] for k in METRICS)
            ]
            time_ratio = (
                g["median_update_ms"]["mean"]
                / index[teacher, "value_greedy"]["median_update_ms"]["mean"]
            )
            gates = {
                "five_percent_derivative_gain": comparisons["selected"]["derivative_relative_mse"]
                <= 0.95,
                "at_most_one_percent_value_regression": comparisons["selected"]["reporting_mse"]
                <= 1.01,
                "every_seed_meets_both": all(
                    s["reporting_mse"] <= 1.01 and s["derivative_relative_mse"] <= 0.95
                    for s in seed_ratios.values()
                ),
                "not_pareto_dominated": not dominators,
                "update_within_ten_percent": time_ratio <= 1.1,
                "finite": all(r["all_finite"] for r in peers),
            }
            teachers[teacher] = {
                "paired_ratios": comparisons,
                "seed_ratios": seed_ratios,
                "dominators": dominators,
                "update_ratio": time_ratio,
                "gates": gates,
                "passes": all(gates.values()),
            }
        components[method] = {
            "teachers": teachers,
            "verdict": "QUALIFIES_FOR_LEARNING_INVESTIGATION"
            if all(t["passes"] for t in teachers.values())
            else "FIXED_INITIALIZER_LEARNING_RECIPE_CLOSED",
        }
    allocation = {}
    for method in METHODS[:3]:
        teachers = {}
        for teacher in ("gelu", "swiglu"):
            g = index[teacher, method]
            references = [c for c in cases if c["teacher"] == teacher]
            ratios = {
                "training_memory": g["peak_allocated_bytes"]["mean"]
                / st.mean(c["full_profile"]["peak_allocated_bytes"] for c in references),
                "inference_memory": g["inference"]["peak_allocated_bytes"]["mean"]
                / st.mean(c["full_inference"]["peak_allocated_bytes"] for c in references),
                "inference_time": g["inference"]["inference_ms"]["mean"]
                / st.mean(c["full_inference"]["inference_ms"] for c in references),
            }
            gates = {
                "mean_value_error": g["reporting_mse"]["mean"] <= 0.05,
                "mean_derivative_error": g["derivative_relative_mse"]["mean"] <= 0.1,
                "all_seed_absolute_errors": all(
                    s["reporting_mse"] <= 0.05 and s["derivative_relative_mse"] <= 0.1
                    for s in g["seed_means"].values()
                ),
                "ten_percent_training_saving": ratios["training_memory"] <= 0.9,
                "ten_percent_inference_saving": ratios["inference_memory"] <= 0.9,
                "inference_within_ten_percent": ratios["inference_time"] <= 1.1,
                "seventy_percent_parameters": g["parameter_reduction"] >= 0.7,
            }
            teachers[teacher] = {"ratios": ratios, "gates": gates, "passes": all(gates.values())}
        eligible = (
            method == "value_greedy"
            or components[method]["verdict"] == "QUALIFIES_FOR_LEARNING_INVESTIGATION"
        )
        allocation[method] = {
            "teachers": teachers,
            "component_eligible": eligible,
            "verdict": "EARNS_LANGUAGE_DIAGNOSTIC"
            if eligible and all(t["passes"] for t in teachers.values())
            else "NO_LANGUAGE_ALLOCATION",
        }
    summary = {
        "groups": groups,
        "components": components,
        "allocation": allocation,
        "elapsed_seconds": result["elapsed_seconds"],
        "neural_updates": 86400,
        "profile_updates": 540,
        "pipeline_peak_cuda_bytes": max(r["pipeline_peak_cuda_bytes"] for r in rows),
        "fresh_capture_seconds": sum(c["seconds"] for c in result["collections"]),
        "fresh_target_seconds": sum(c["target_seconds"] for c in cases),
        "prior_statistics_and_all_selectors_seconds": sum(
            c["prior_statistics_seconds"] + c["prior_all_selectors_seconds"] for c in cases
        ),
        "full_references": {
            teacher: {
                k: st.mean(c[part][field] for c in cases if c["teacher"] == teacher)
                for k, part, field in (
                    ("training_peak_bytes", "full_profile", "peak_allocated_bytes"),
                    ("inference_peak_bytes", "full_inference", "peak_allocated_bytes"),
                    ("inference_ms", "full_inference", "inference_ms"),
                )
            }
            for teacher in ("gelu", "swiglu")
        },
        "breakthrough": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = []
    for row in rows:
        for i, endpoint in enumerate(row["endpoints"]):
            table.append(
                {
                    **{
                        k: row[k]
                        for k in (
                            "teacher",
                            "layer",
                            "seed",
                            "method",
                            "parameters",
                            "peak_allocated_bytes",
                            "pipeline_peak_cuda_bytes",
                        )
                    },
                    "selected": i == row["selected_index"],
                    **{k: endpoint[k] for k in ("rate", "steps", "selection_mse", *METRICS)},
                    "checkpoint_sha256": endpoint["checkpoint"]["sha256"],
                }
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    (ROOT / "metrics.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    diagnostics = []
    for row in rows:
        for rate in (0.001, 0.003):
            record = read(
                ROOT
                / "pairs"
                / f"{row['teacher']}_l{row['layer']}_s{row['seed']}"
                / row["method"]
                / f"rate{rate}.json"
            )
            diagnostics.append(
                {
                    **{k: row[k] for k in ("teacher", "layer", "seed", "method")},
                    "rate": rate,
                    "diagnostics": record["diagnostics"],
                }
            )
    (ROOT / "diagnostics.json.gz").write_bytes(
        gzip.compress(json.dumps(diagnostics).encode(), mtime=0)
    )
    print(json.dumps({"components": components, "allocation": allocation}, indent=2))


if __name__ == "__main__":
    run()
