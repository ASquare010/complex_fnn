"""Summarize all H110 trials and enforce the frozen fidelity/memory/runtime gates."""

import csv
import gzip
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/token_memory_duration_v1")


def describe(values):
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "sample_variance": st.variance(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def run():
    result = json.loads((ROOT / "result.json").read_text())
    rows = result["cases"]
    lookup = {(r["context"], r["seed"], r["policy"]): r for r in rows}
    comparisons = []
    for recorded in result["pairs"]:
        context, seed = recorded["context"], recorded["seed"]
        ref, candidate = [lookup[context, seed, p] for p in ("block", "loss_chunks")]
        memory = candidate["peak_training_allocated_bytes"] / ref["peak_training_allocated_bytes"]
        job = candidate["peak_job_allocated_bytes"] / ref["peak_job_allocated_bytes"]
        timing = candidate["timing"]["update_ms"]["median"] / ref["timing"]["update_ms"]["median"]
        nll = candidate["full_validation"]["nll"] / ref["full_validation"]["nll"] - 1
        trajectory = {
            str(a["step"]): b["nll"] / a["nll"] - 1
            for a, b in zip(ref["evaluations"], candidate["evaluations"], strict=True)
        }
        gates = {
            "training_memory_saving_at_least_15_percent": memory <= 0.85,
            "median_update_cost_at_most_25_percent": timing <= 1.25,
            "full_nll_within_one_percent": abs(nll) <= 0.01,
            "trajectory_nll_within_one_percent": all(
                abs(v) <= 0.01 for step, v in trajectory.items() if step != "0"
            ),
            "finite": ref["weights_and_moments_finite"] and candidate["weights_and_moments_finite"],
        }
        comparisons.append(
            {
                "context": context,
                "batch": recorded["batch"],
                "seed": seed,
                "training_memory_ratio": memory,
                "job_memory_ratio": job,
                "update_time_ratio": timing,
                "full_relative_nll_change": nll,
                "trajectory_relative_nll_changes": trajectory,
                "gates": gates,
                "passes": all(gates.values()),
            }
        )
    groups = []
    for context in (128, 512):
        pairs = [c for c in comparisons if c["context"] == context]
        if not pairs:
            continue
        groups.append(
            {
                "context": context,
                "paired_seeds": [c["seed"] for c in pairs],
                **{
                    key: describe([p[key] for p in pairs])
                    for key in (
                        "training_memory_ratio",
                        "job_memory_ratio",
                        "update_time_ratio",
                        "full_relative_nll_change",
                    )
                },
                "policies": {
                    p: {
                        "full_nll": describe(
                            [
                                r["full_validation"]["nll"]
                                for r in rows
                                if r["context"] == context and r["policy"] == p
                            ]
                        ),
                        "training_peak_bytes": describe(
                            [
                                r["peak_training_allocated_bytes"]
                                for r in rows
                                if r["context"] == context and r["policy"] == p
                            ]
                        ),
                        "job_peak_bytes": describe(
                            [
                                r["peak_job_allocated_bytes"]
                                for r in rows
                                if r["context"] == context and r["policy"] == p
                            ]
                        ),
                        "validation_peak_bytes": describe(
                            [
                                r["peak_validation_allocated_bytes"]
                                for r in rows
                                if r["context"] == context and r["policy"] == p
                            ]
                        ),
                        "median_update_ms": describe(
                            [
                                r["timing"]["update_ms"]["median"]
                                for r in rows
                                if r["context"] == context and r["policy"] == p
                            ]
                        ),
                    }
                    for p in ("block", "loss_chunks")
                },
                "verdict": "QUALIFIED_800_STEP_EXECUTION_RESULT"
                if len(pairs) == 3 and all(p["passes"] for p in pairs)
                else "FIXED_DURATION_RECIPE_NOT_QUALIFIED",
                "whole_job_saving_at_least_15_percent_each_seed": len(pairs) == 3
                and all(p["job_memory_ratio"] <= 0.85 for p in pairs),
            }
        )
    summary = {
        "experiment_status": result["status"],
        "completed_trials": len(rows),
        "updates": result["updates"],
        "training_targets": sum(r["training_targets"] for r in rows),
        "comparisons": comparisons,
        "groups": groups,
        "elapsed_seconds": result["elapsed_seconds"],
        "all_weights_and_moments_finite": all(r["weights_and_moments_finite"] for r in rows),
        "breakthrough": False,
        "new_activation": False,
        "parameters_changed": False,
        "architectural_goal_achieved": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = [
        {
            **{
                k: r[k]
                for k in (
                    "label",
                    "context",
                    "batch",
                    "seed",
                    "policy",
                    "steps",
                    "training_targets",
                    "parameter_count",
                    "peak_training_allocated_bytes",
                    "peak_validation_allocated_bytes",
                    "peak_job_allocated_bytes",
                    "clip_fraction",
                )
            },
            "full_validation_nll": r["full_validation"]["nll"],
            "median_update_ms": r["timing"]["update_ms"]["median"],
            "mean_update_ms": r["timing"]["update_ms"]["mean"],
            "training_tokens_per_second": 1000
            * r["batch"]
            * r["context"]
            / r["timing"]["update_ms"]["mean"],
        }
        for r in rows
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    (ROOT / "metrics.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    (ROOT / "diagnostics.json.gz").write_bytes(
        gzip.compress(json.dumps({r["label"]: r["diagnostics"] for r in rows}).encode(), mtime=0)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
