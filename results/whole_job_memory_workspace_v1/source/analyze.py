"""H112 decisions use native audited quality and actual integrated job peaks."""

import csv
import gzip
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")


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
    result, audit = [json.loads((ROOT / n).read_text()) for n in ("result.json", "audit.json")]
    assert result["status"] == "COMPLETE" and audit["passed"]
    rows = result["cases"]
    lookup = {(r["dataset"], r["seed"], r["policy"]): r for r in rows}
    primary = {
        (r["dataset"], r["seed"], r["policy"]): r["native_full_nll"] for r in audit["trials"]
    }
    pairs, groups = [], []
    for dataset in ("wikitext2", "tinystories"):
        for seed in (61, 73, 89):
            ref, candidate = [lookup[dataset, seed, p] for p in ("block", "loss_chunks")]
            nll = primary[dataset, seed, "loss_chunks"] / primary[dataset, seed, "block"]
            memory = candidate["peak_job_allocated_bytes"] / ref["peak_job_allocated_bytes"]
            time = candidate["timing"]["update_ms"]["median"] / ref["timing"]["update_ms"]["median"]
            gates = {
                "native_nll_at_most_one_percent_worse": nll <= 1.01,
                "whole_job_saving_at_least_15_percent": memory <= 0.85,
                "median_update_cost_at_most_25_percent": time <= 1.25,
                "finite": candidate["weights_and_moments_finite"]
                and ref["weights_and_moments_finite"],
            }
            pairs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "native_nll_ratio": nll,
                    "job_memory_ratio": memory,
                    "training_memory_ratio": candidate["peak_training_allocated_bytes"]
                    / ref["peak_training_allocated_bytes"],
                    "update_time_ratio": time,
                    "gates": gates,
                    "passes": all(gates.values()),
                }
            )
        peers = [p for p in pairs if p["dataset"] == dataset]
        policies = {}
        for policy in ("block", "loss_chunks"):
            subset = [r for r in rows if r["dataset"] == dataset and r["policy"] == policy]
            policies[policy] = {
                "native_full_nll": describe([primary[dataset, r["seed"], policy] for r in subset]),
                **{
                    key: describe([r[key] for r in subset])
                    for key in (
                        "peak_training_allocated_bytes",
                        "peak_validation_allocated_bytes",
                        "peak_job_allocated_bytes",
                        "clip_fraction",
                    )
                },
                "median_update_ms": describe([r["timing"]["update_ms"]["median"] for r in subset]),
            }
        groups.append(
            {
                "dataset": dataset,
                "policies": policies,
                **{
                    key: describe([p[key] for p in peers])
                    for key in (
                        "native_nll_ratio",
                        "job_memory_ratio",
                        "training_memory_ratio",
                        "update_time_ratio",
                    )
                },
                "all_seeds_pass": all(p["passes"] for p in peers),
                "verdict": "SCOPED_WHOLE_JOB_COMPONENT_QUALIFIED"
                if all(p["passes"] for p in peers)
                else "FIXED_RECIPE_NOT_QUALIFIED",
            }
        )
    summary = {
        "status": "COMPLETE",
        "completed_trials": 12,
        "updates": 9600,
        "training_targets": result["training_targets"],
        "pairs": pairs,
        "groups": groups,
        "two_corpus_component_qualifies": all(g["all_seeds_pass"] for g in groups),
        "single_process_fresh_models": True,
        "zero_allocation_boundaries": len(result["boundaries"]),
        "workspace_bytes_subtracted": 0,
        "elapsed_seconds": result["elapsed_seconds"],
        "parameters_changed": False,
        "new_architecture": False,
        "broad_goal_achieved": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = []
    for r in rows:
        table.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "dataset",
                        "label",
                        "seed",
                        "policy",
                        "evaluation_policy",
                        "steps",
                        "training_targets",
                        "parameter_count",
                        "ffn_parameter_count",
                        "peak_training_allocated_bytes",
                        "peak_validation_allocated_bytes",
                        "peak_job_allocated_bytes",
                        "clip_fraction",
                    )
                },
                "native_full_nll": primary[r["dataset"], r["seed"], r["policy"]],
                "streamed_full_nll": r["full_validation"]["nll"],
                "median_update_ms": r["timing"]["update_ms"]["median"],
                "mean_update_ms": r["timing"]["update_ms"]["mean"],
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    (ROOT / "metrics.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    (ROOT / "diagnostics.json.gz").write_bytes(
        gzip.compress(
            json.dumps({f"{r['dataset']}_{r['label']}": r["diagnostics"] for r in rows}).encode(),
            mtime=0,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
