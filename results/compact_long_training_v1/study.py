"""Reuse H117's long-training loop, with H125's already qualified adapter."""

# ruff: noqa: I001
from results.fp32_training_replication_recovery_v1.source import study as loop
from results.compact_training_v1.study import HostLedger
from results.compact_training_v1.adapter import Adapter
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import json

ROOT = Path("results/compact_long_training_v1")


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
        hashes(p[field])
    loop.write_json(ROOT / "environment.json", loop.environment())
    cases, boundaries = [], [boundary()]
    for i, f in enumerate(p["fixtures"]):
        for arm in ("native", "chunks", "combined") if i == 0 else ("combined", "chunks", "native"):
            policy = "fp32_default_native" if arm == "native" else "fp32_default_chunks"
            print(json.dumps(dict(starting_arm=arm, fixture=f["label"])), flush=True)
            with (
                patch.object(loop, "ROOT", ROOT / arm),
                patch.object(loop, "MemoryLedger", HostLedger),
                nullcontext() if arm == "native" else Adapter(loop, arm == "combined") as adapter,
            ):
                c = loop.run_case(f, policy, p)
                restored = [] if adapter is None else list(adapter.restorations)
            if arm == "combined":
                assert len(restored) == 801 and all(
                    v == dict(hooks=50, restored_bytes=36398592) for v in restored
                )
            cases.append(dict(arm=arm, measurement=c, restorations=restored))
            boundaries.append(boundary())
            loop.write_json(
                ROOT / "progress.json",
                dict(
                    completed_cases=len(cases),
                    updates=800 * len(cases),
                    last_arm=arm,
                    last_fixture=f["label"],
                ),
            )
    loop.write_json(
        ROOT / "result.json",
        dict(
            cases=cases,
            boundaries=boundaries,
            training_updates=4800,
            training_targets=19660800,
            backwards=4806,
            study_scores=24,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
