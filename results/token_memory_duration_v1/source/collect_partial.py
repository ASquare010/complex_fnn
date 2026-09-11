"""Expose completed original evidence without changing the failed study outcome."""

import json
from pathlib import Path

root = Path("results/token_memory_duration_v1")
assert json.loads((root / "coordinator_status.json").read_text())["status"] == "WORKER_FAILED"
assert not (root / "result.json").exists()
row = json.loads((root / "runs/b16_t128_s17_block/metrics.json").read_text())
assert row["status"] == "COMPLETE" and row["steps"] == 800
with (root / "partial_result.json").open("x") as stream:
    json.dump(
        {
            "status": "RUNTIME_INTERRUPTED",
            "posthoc_collection": True,
            "cases": [row],
            "pairs": [],
            "updates": 800,
            "planned_updates": 9600,
            "candidate_updates": 0,
            "broad_goal_achieved": False,
        },
        stream,
        indent=2,
    )
print("Collected one completed control; zero candidate updates or paired conclusions")
