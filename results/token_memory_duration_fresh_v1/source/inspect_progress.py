"""Read complete trial files only; print compact diagnostics without a verdict."""

import json
from pathlib import Path

roots = [Path("results/token_memory_duration_v1"), Path("results/token_memory_duration_fresh_v1")]
for root in roots:
    for path in sorted(root.glob("runs/*/metrics.json")):
        row = json.loads(path.read_text())
        print(
            json.dumps(
                {
                    "label": row["label"],
                    "nll": row["full_validation"]["nll"],
                    "training_mib": row["peak_training_allocated_bytes"] / 2**20,
                    "validation_mib": row["peak_validation_allocated_bytes"] / 2**20,
                    "job_mib": row["peak_job_allocated_bytes"] / 2**20,
                    "median_update_ms": row["timing"]["update_ms"]["median"],
                    "clip_fraction": row["clip_fraction"],
                    "targets": row["full_validation"]["targets"],
                    "early_late_ms": row["early_late_update_ms"],
                }
            )
        )
