"""Apply H113's prospective gates to every saved fixture without selection."""

import csv
import gzip
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/classifier_precision_v1")


def describe(values):
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "sample_variance": st.variance(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def run():
    result = json.loads((ROOT / "result.json").read_text())
    assert result["status"] == "COMPLETE" and result["fixtures"] == 24
    entries, topology = [], []
    for row in result["rows"]:
        records = {r["policy"]: r for r in row["records"]}
        cached, uncached, native = [
            records[p] for p in ("chunks_cached", "chunks_uncached", "native_bf16")
        ]
        topology.append(
            {
                "label": row["label"],
                "cached_cast_nodes": cached["cast_graph"]["weight_cast_nodes"],
                "uncached_cast_nodes": uncached["cast_graph"]["weight_cast_nodes"],
                "cached_cast_gradient_dtypes": [e["dtype"] for e in cached["cast_graph"]["events"]],
                "uncached_cast_gradient_dtypes": [
                    e["dtype"] for e in uncached["cast_graph"]["events"]
                ],
                **row["comparison"],
            }
        )
        for record in row["records"]:
            entries.append(
                {
                    "label": row["label"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "step": row["step"],
                    "training_policy": row["training_policy"],
                    "policy": record["policy"],
                    "loss": record["loss"],
                    "loss_relative_error": record["relative_loss_error"],
                    "hidden_error": record["hidden_gradient"]["relative_l2"],
                    "weight_error": record["weight_gradient"]["relative_l2"],
                    "hidden_error_ratio_to_cached": record["hidden_gradient"]["relative_l2"]
                    / cached["hidden_gradient"]["relative_l2"],
                    "weight_error_ratio_to_cached": record["weight_gradient"]["relative_l2"]
                    / cached["weight_gradient"]["relative_l2"],
                    "allocated_bytes": record["peak_allocated_bytes"],
                    "memory_ratio_to_native_bf16": record["peak_allocated_bytes"]
                    / native["peak_allocated_bytes"],
                    "median_total_ms": record["median_total_ms"],
                    "time_ratio_to_native_bf16": record["median_total_ms"]
                    / native["median_total_ms"],
                }
            )
    metric_keys = [
        k
        for k in entries[0]
        if k not in ("label", "dataset", "seed", "step", "training_policy", "policy")
    ]
    groups, by_step, decisions = [], [], []
    for dataset in ("wikitext2", "tinystories"):
        for policy in (
            "native_bf16",
            "chunks_cached",
            "chunks_uncached",
            "native_fp32",
            "chunks_fp32",
        ):
            peers = [e for e in entries if e["dataset"] == dataset and e["policy"] == policy]
            assert len(peers) == 12
            groups.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "metrics": {k: describe([e[k] for e in peers]) for k in metric_keys},
                }
            )
            for step in (100, 800):
                subset = [e for e in peers if e["step"] == step]
                by_step.append(
                    {
                        "dataset": dataset,
                        "policy": policy,
                        "step": step,
                        "metrics": {k: describe([e[k] for e in subset]) for k in metric_keys},
                    }
                )
            if policy in ("chunks_uncached", "chunks_fp32"):
                gates = {
                    "median_weight_error_ratio_at_most_0_75": st.median(
                        e["weight_error_ratio_to_cached"] for e in peers
                    )
                    <= 0.75,
                    "all_hidden_error_ratios_at_most_1_01": all(
                        e["hidden_error_ratio_to_cached"] <= 1.01 for e in peers
                    ),
                    "all_memory_ratios_at_most_0_85": all(
                        e["memory_ratio_to_native_bf16"] <= 0.85 for e in peers
                    ),
                    "median_time_ratio_at_most_1_5": st.median(
                        e["time_ratio_to_native_bf16"] for e in peers
                    )
                    <= 1.5,
                }
                decisions.append(
                    {
                        "dataset": dataset,
                        "policy": policy,
                        "gates": gates,
                        "earns_whole_model_resource_screen": all(gates.values()),
                    }
                )
    mechanism = all(
        t["cached_cast_nodes"] == 1
        and t["uncached_cast_nodes"] == 8
        and t["loss_equal"]
        and t["hidden_gradient_equal"]
        and t["cached_cast_gradient_dtypes"] == ["torch.bfloat16"]
        and t["uncached_cast_gradient_dtypes"] == ["torch.bfloat16"] * 8
        for t in topology
    ) and any(not t["weight_gradient_equal"] for t in topology)
    summary = {
        "status": "COMPLETE",
        "fixtures": 24,
        "policy_comparisons": 120,
        "classifier_backward_passes": 504,
        "optimizer_updates": 0,
        "elapsed_seconds": result["elapsed_seconds"],
        "cache_mechanism_supported": mechanism,
        "topology": topology,
        "groups": groups,
        "by_step": by_step,
        "decisions": decisions,
        "whole_model_screen_candidates": [
            p
            for p in ("chunks_uncached", "chunks_fp32")
            if all(d["earns_whole_model_resource_screen"] for d in decisions if d["policy"] == p)
        ],
        "capture_peak_allocated_bytes": max(
            r["capture_peak_allocated_bytes"] for r in result["rows"]
        ),
        "maximum_logged_phase_peak_bytes": max(
            max(
                r["reference_measurement"]["peak_allocated_bytes"],
                r["capture_peak_allocated_bytes"],
                *[m["peak_allocated_bytes"] for m in r["records"]],
            )
            for r in result["rows"]
        ),
        "whole_diagnostic_process_peak_measured": False,
        "accounting_limitation": "Peak counters reset per repetition; warmup peaks were not serialized, so maximum logged phase peak is not a whole-process maximum.",
        "language_quality_proved": False,
        "new_architecture": False,
        "broad_goal_achieved": False,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(entries[0]))
    writer.writeheader()
    writer.writerows(entries)
    (ROOT / "metrics.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("topology", "groups", "by_step")},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
