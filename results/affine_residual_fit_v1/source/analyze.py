"""Post-run descriptive statistics and frozen H107 decisions; standard library only."""

import csv
import gzip
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/affine_residual_fit_v1")
CONTROLS = ("gelu", "relu", "leaky_relu", "prelu", "silu", "swiglu")
CANDIDATES = ("affine_gelu", "affine_swiglu")


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "sample_variance": st.variance(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def geometric(values):
    return math.exp(st.mean(math.log(v) for v in values))


def run():
    result = read(ROOT / "result.json")
    assert result["status"] == "COMPLETE"
    rows = result["rows"]
    lookup = {(r["teacher"], r["layer"], r["seed"], r["form"]): r for r in rows}
    descriptions = []
    for teacher in ("gelu", "swiglu"):
        for form in (*CANDIDATES, *CONTROLS):
            peers = [r for r in rows if (r["teacher"], r["form"]) == (teacher, form)]
            descriptions.append(
                {
                    "teacher": teacher,
                    "form": form,
                    "parameters": peers[0]["parameters"],
                    "parameter_reduction": peers[0]["parameter_reduction"],
                    **{
                        key: describe([r[key] for r in peers])
                        for key in (
                            "reporting_mse",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                            "pipeline_peak_cuda_bytes",
                            "median_update_ms",
                            "median_forward_ms",
                            "median_backward_ms",
                            "initialization_seconds",
                        )
                    },
                    "raw_inference_ms": describe([r["raw_inference"]["median_ms"] for r in peers]),
                    "seed_mse": {
                        str(s): st.mean(r["reporting_mse"] for r in peers if r["seed"] == s)
                        for s in (71, 83, 97)
                    },
                    "depth_mse": {
                        str(layer): st.mean(
                            r["reporting_mse"] for r in peers if r["layer"] == layer
                        )
                        for layer in (0, 3, 7)
                    },
                    "selected_initializations": sum(r["selected_index"] == 0 for r in peers),
                    "mean_initial_mse": st.mean(r["endpoints"][0]["reporting_mse"] for r in peers),
                    "mean_relative_training_gain": st.mean(
                        1 - r["reporting_mse"] / r["endpoints"][0]["reporting_mse"] for r in peers
                    ),
                }
            )
    decisions = {}
    for candidate in CANDIDATES:
        teachers = {}
        for teacher in ("gelu", "swiglu"):
            peers = [r for r in rows if (r["teacher"], r["form"]) == (teacher, candidate)]
            diagnostics = [d for d in result["diagnostics"] if d["teacher"] == teacher]
            ratios = {
                control: geometric(
                    [
                        r["reporting_mse"]
                        / lookup[teacher, r["layer"], r["seed"], control]["reporting_mse"]
                        for r in peers
                    ]
                )
                for control in CONTROLS
            }
            seed_ratios = {
                str(seed): {
                    control: geometric(
                        [
                            r["reporting_mse"]
                            / lookup[teacher, r["layer"], seed, control]["reporting_mse"]
                            for r in peers
                            if r["seed"] == seed
                        ]
                    )
                    for control in CONTROLS
                }
                for seed in (71, 83, 97)
            }
            depth_ratios = {
                str(layer): st.mean(r["reporting_mse"] for r in peers if r["layer"] == layer)
                / min(
                    st.mean(
                        lookup[teacher, layer, seed, control]["reporting_mse"]
                        for seed in (71, 83, 97)
                    )
                    for control in CONTROLS
                )
                for layer in (0, 3, 7)
            }
            ordinary = [lookup[teacher, r["layer"], r["seed"], "gelu"] for r in peers]
            memory_ratio = st.mean(r["peak_allocated_bytes"] for r in peers) / st.mean(
                d["full_profile"]["peak_allocated_bytes"] for d in diagnostics
            )
            update_ratio = st.mean(r["median_update_ms"] for r in peers) / st.mean(
                r["median_update_ms"] for r in ordinary
            )
            inference_ratio = st.mean(r["raw_inference"]["median_ms"] for r in peers) / st.mean(
                r["raw_inference"]["median_ms"] for r in ordinary
            )
            gates = {
                "mean_error_at_most_five_percent": st.mean(r["reporting_mse"] for r in peers)
                <= 0.05,
                "five_percent_better_than_every_control": all(r <= 0.95 for r in ratios.values()),
                "every_seed_beats_every_control": all(
                    r < 1 for values in seed_ratios.values() for r in values.values()
                ),
                "no_depth_regression_above_five_percent": all(
                    r <= 1.05 for r in depth_ratios.values()
                ),
                "finite": all(r["all_finite"] for r in peers),
                "at_least_seventy_percent_fewer_parameters": all(
                    r["parameter_reduction"] >= 0.7 for r in peers
                ),
                "at_least_twenty_five_percent_lower_training_allocation": memory_ratio <= 0.75,
                "update_within_twenty_five_percent_of_gelu": update_ratio <= 1.25,
                "raw_inference_within_twenty_five_percent_of_gelu": inference_ratio <= 1.25,
            }
            teachers[teacher] = {
                "gates": gates,
                "paired_geomean_ratios": ratios,
                "seed_ratios": seed_ratios,
                "depth_ratios_vs_best_control": depth_ratios,
                "training_memory_ratio_to_full": memory_ratio,
                "update_ratio_to_gelu": update_ratio,
                "raw_inference_ratio_to_gelu": inference_ratio,
                "passes": all(gates.values()),
            }
        decisions[candidate] = {
            "teachers": teachers,
            "verdict": "EARNS_LANGUAGE_TEST"
            if all(t["passes"] for t in teachers.values())
            else "CLOSED_AT_THIS_RECIPE",
        }
    capacity = read("results/affine_residual_capacity_v1/result.json")
    capacity_summary = [
        {
            "teacher": t,
            "rank": rank,
            "floor": describe(
                [
                    r["fp32_lower_bound"]
                    for r in capacity["rows"]
                    if r["teacher"] == t and r["rank"] == rank
                ]
            ),
            "training_basis_oracle_upper": describe(
                [
                    r["train_basis_oracle_fp32_upper"]
                    for r in capacity["rows"]
                    if r["teacher"] == t and r["rank"] == rank
                ]
            ),
        }
        for t in ("gelu", "swiglu")
        for rank in (32, 64, 85, 128, 170, 192, 256)
    ]
    summary = {
        "descriptive_units": "Nine depth-by-sampling cells per teacher/form; not nine independent teacher training seeds",
        "statistics": descriptions,
        "decisions": decisions,
        "capacity": capacity_summary,
        "capacity_decisions": capacity["decisions"],
        "elapsed_seconds": result["elapsed_seconds"],
        "capacity_seconds": capacity["elapsed_seconds"],
        "collection_seconds": sum(c["seconds"] for c in result["collections"]),
        "initialization_seconds": sum(r["initialization_seconds"] for r in rows),
        "neural_updates": result["neural_updates"],
        "reference_profile_updates": result["resource_profile_updates"],
        "pipeline_peak_cuda_bytes": max(r["pipeline_peak_cuda_bytes"] for r in rows),
        "full_profiles": {
            t: {
                k: describe(
                    [d["full_profile"][k] for d in result["diagnostics"] if d["teacher"] == t]
                )
                for k in ("peak_allocated_bytes", "median_update_ms", "reporting_mse")
            }
            for t in ("gelu", "swiglu")
        },
        "affine_only_mse": {
            t: describe(
                [d["affine_only_reporting_mse"] for d in result["diagnostics"] if d["teacher"] == t]
            )
            for t in ("gelu", "swiglu")
        },
        "breakthrough": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    flat = []
    for row in rows:
        for i, endpoint in enumerate(row["endpoints"]):
            flat.append(
                {
                    **{
                        k: row[k]
                        for k in (
                            "teacher",
                            "layer",
                            "seed",
                            "form",
                            "parameters",
                            "parameter_reduction",
                            "peak_allocated_bytes",
                            "median_update_ms",
                        )
                    },
                    "inference_ms": row["raw_inference"]["median_ms"],
                    "rate": endpoint["rate"],
                    "steps": endpoint["steps"],
                    "selection_mse": endpoint["selection_mse"],
                    "reporting_mse": endpoint["reporting_mse"],
                    "selected": i == row["selected_index"],
                    "checkpoint_sha256": endpoint["checkpoint"]["sha256"],
                }
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    (ROOT / "metrics.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    for item in descriptions:
        print(
            item["teacher"],
            item["form"],
            "MSE",
            round(item["reporting_mse"]["mean"], 6),
            "MiB",
            round(item["peak_allocated_bytes"]["mean"] / 2**20, 3),
            "ms",
            round(item["median_update_ms"]["mean"], 3),
        )
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    run()
