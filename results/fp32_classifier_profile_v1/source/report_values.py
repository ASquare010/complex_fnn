"""Read-only bounded report values from the complete H114 evidence."""

import json
from pathlib import Path

root = Path("results/fp32_classifier_profile_v1")
result = json.loads((root / "result.json").read_text())
print("TOTAL", {k: v for k, v in result.items() if k not in ("cases", "boundaries")})
for case in result["cases"]:
    if case["policy"] in ("block_bf16", "chunks_fp32"):
        phases = [
            p["phase"]
            for p in case["memory_phases"]
            if p["peak_allocated_bytes"] == case["peak_job_allocated_bytes"]
        ]
        print(
            "PEAK",
            case["label"],
            case["peak_job_allocated_bytes"] / 2**20,
            phases[:4],
            "count",
            len(phases),
            "reserved",
            case["peak_job_reserved_bytes"] / 2**20,
            "state_MiB",
            {k: case[k] / 2**20 for k in ("parameter_bytes", "gradient_bytes", "optimizer_bytes")},
            "sourceLR",
            case["source_learning_rates"],
        )
summary_path = root / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    for group in summary["groups"]:
        print(
            "GROUP",
            group["scope"],
            group["policy"],
            {
                k: dict(median=v["median"], min=v["min"], max=v["max"])
                for k, v in group["metrics"].items()
            },
        )
    print("DECISIONS", summary["decisions"])
    print("AUDIT", summary["independent_audit"])
