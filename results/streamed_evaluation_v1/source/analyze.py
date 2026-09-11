"""Apply H111's fixed score, allocation, time and policy-selection gates."""

import json
import statistics as st
from pathlib import Path

ROOT = Path("results/streamed_evaluation_v1")


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
    assert result["status"] == "COMPLETE" and len(result["rows"]) == 24
    groups = []
    for context in (128, 512):
        states = [r for r in result["rows"] if r["context"] == context]
        policies = {}
        for policy in ("native", "classifier_chunks", "sequence_chunks"):
            rows = [p for r in states for p in r["policies"] if p["policy"] == policy]
            gates = {
                "all_scores_within_0_01_percent": all(
                    p["relative_nll_difference_abs"] <= 0.0001 for p in rows
                ),
                "all_evaluation_memory_savings_at_least_10_percent": all(
                    p["evaluation_memory_ratio"] <= 0.9 for p in rows
                ),
                "median_evaluation_time_ratio_at_most_1_5": st.median(
                    p["evaluation_time_ratio"] for p in rows
                )
                <= 1.5,
            }
            policies[policy] = {
                **{
                    key: describe([p[key] for p in rows])
                    for key in (
                        "relative_nll_difference_abs",
                        "peak_allocated_bytes",
                        "evaluation_memory_ratio",
                        "evaluation_time_ratio",
                        "estimated_job_peak_bytes",
                    )
                },
                "gates": gates,
                "eligible": policy != "native" and all(gates.values()),
            }
        eligible = [p for p in ("classifier_chunks", "sequence_chunks") if policies[p]["eligible"]]
        selected = (
            min(
                eligible,
                key=lambda p: (
                    policies[p]["estimated_job_peak_bytes"]["median"],
                    st.median(
                        q["timing_ms"]["median"]
                        for r in states
                        for q in r["policies"]
                        if q["policy"] == p
                    ),
                    0 if p == "sequence_chunks" else 1,
                ),
            )
            if eligible
            else None
        )
        groups.append(
            {
                "context": context,
                "states": len(states),
                "policies": policies,
                "selected_policy": selected,
                "verdict": "EVALUATION_COMPONENT_QUALIFIED"
                if selected
                else "NO_TRAINING_ALLOCATION",
            }
        )
    summary = {
        "status": "COMPLETE",
        "states": 24,
        "scores": 72,
        "optimizer_updates": 0,
        "groups": groups,
        "actual_training_job": False,
        "peak_process_allocated_bytes": result["peak_process_allocated_bytes"],
        "wall_seconds": result["wall_seconds"],
        "broad_goal_achieved": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
