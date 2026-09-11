"""Read completed metadata without importing the training runtime."""

import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")
protocol = json.loads((ROOT / "protocol.json").read_text())
print("Protocol keys:", list(protocol))
print("Source count:", len(protocol["sources"]))
for key in (
    "datasets",
    "previous_evidence",
    "prerequisite",
    "runtime_recovery",
    "workspace_recovery",
):
    if key in protocol:
        print(key, json.dumps(protocol[key], indent=2))
print((ROOT / "coordinator_status.json").read_text())
for path in sorted(ROOT.glob("*/runs/*/metrics.json")):
    row = json.loads(path.read_text())
    print(
        path.parts[-4],
        row["label"],
        row["full_validation"]["nll"],
        row["peak_job_allocated_bytes"] / 2**20,
        row["timing"]["update_ms"]["median"],
    )
