"""Read-only compact preview; no allocation decision from an incomplete grid."""

import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("results/streamed_evaluation_v1/progress.jsonl").read_text().splitlines()
]
for row in rows:
    print(
        json.dumps(
            {
                "label": row["label"],
                "step": row["checkpoint"]["step"],
                "policies": {
                    p["policy"]: {
                        "nll_difference": p["relative_nll_difference_abs"],
                        "mib": p["peak_allocated_bytes"] / 2**20,
                        "memory_ratio": p["evaluation_memory_ratio"],
                        "time_ratio": p["evaluation_time_ratio"],
                    }
                    for p in row["policies"]
                },
            }
        )
    )
