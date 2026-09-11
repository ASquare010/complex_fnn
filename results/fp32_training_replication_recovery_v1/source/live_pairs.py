"""Read completed fixture comparisons; never treat partial results as qualification."""

import json
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
cases = [json.loads(p.read_text()) for p in (ROOT / "runs").glob("*/metrics.json")]
for fixture in dict.fromkeys(c["fixture"]["label"] for c in cases):
    peers = {c["policy"]: c for c in cases if c["fixture"]["label"] == fixture}
    if len(peers) != 3:
        continue
    candidate = peers["fp32_default_chunks"]
    row = dict(fixture=fixture, complete_policies=3, nll={p: c["final_validation"]["nll"] for p, c in peers.items()})
    row["candidate_ratios"] = {p: dict(
        nll=candidate["final_validation"]["nll"] / peers[p]["final_validation"]["nll"],
        memory=candidate["peak_job_allocated_bytes"] / peers[p]["peak_job_allocated_bytes"],
        time=candidate["timing"]["wall_update_ms"]["median"] / peers[p]["timing"]["wall_update_ms"]["median"])
        for p in ("bf16_default_native", "fp32_default_native")}
    print(json.dumps(row))
print(json.dumps(dict(completed_trials=len(cases), optimizer_updates=800 * len(cases), final_qualification=False)))
