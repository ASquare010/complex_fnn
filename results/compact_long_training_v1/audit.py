"""Regenerate states and reuse the independent H117 gradient/data/score audit."""

# ruff: noqa: I001
from results.fp32_training_replication_recovery_v1.source import audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path

ROOT = Path("results/compact_long_training_v1")


def run():
    assert not (ROOT / "audit.json").exists()
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    initials, rows, boundaries = [], [], [boundary()]
    for f in p["fixtures"]:
        peers = [row for row in r["cases"] if row["measurement"]["fixture"] == f]
        initials.append(old.audit_initial(f, [row["measurement"] for row in peers], p))
        boundaries.append(boundary())
        for row in peers:
            check = old.audit_case(row["measurement"], p)
            rows.append(dict(arm=row["arm"], check=check))
            boundaries.append(boundary())
    passed = all(v["passed"] for v in initials) and all(
        v["check"]["all_scores_passed"] and v["check"]["replay"]["passed"] for v in rows
    )
    old.write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            initials=initials,
            cases=rows,
            boundaries=boundaries,
            backwards=6,
            optimizer_updates=0,
            native_scores=20,
            training_batches=4800,
        ),
    )
    print("Independent audit:", passed, flush=True)


if __name__ == "__main__":
    run()
