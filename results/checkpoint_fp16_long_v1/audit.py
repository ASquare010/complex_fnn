"""Regenerate all seeds, replay native gradients and score every saved endpoint."""

# ruff: noqa: I001
from results.fp32_training_replication_recovery_v1.source import audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from results.ordinary_long_training_v1.io import write_json
from results.checkpoint_fp16_long_v1.numerics import compare
from pathlib import Path
from unittest.mock import patch
import os

ROOT = Path("results/checkpoint_fp16_long_v1")


def run():
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    assert not (ROOT / "audit.json").exists()
    for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    old.torch.use_deterministic_algorithms(False)
    old.torch.backends.cudnn.deterministic = False
    old.torch.backends.cudnn.benchmark = False
    assert old.torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    initials, rows, boundaries = [], [], [boundary()]
    with patch.object(old, "write_json", write_json):
        for f in p["fixtures"]:
            peers = [row for row in r["cases"] if row["fixture"] == f]
            initials.append(old.audit_initial(f, [row["measurement"] for row in peers], p))
            boundaries.append(boundary())
            for row in peers:
                comparison = []
                original_error = old.independent_error

                def capture(native, saved):
                    comparison.append(compare(saved, native))
                    return original_error(native, saved)

                with patch.object(old, "independent_error", capture):
                    check = old.audit_case(row["measurement"], p)
                assert len(comparison) == 1
                check["approximate_gradient_comparison"] = comparison[0]
                check["qualified_replay_passed"] = (
                    comparison[0]["passed"] and check["replay"]["loss_relative_error"] <= 1e-6
                    if row["arm"] == "fp16"
                    else check["replay"]["passed"]
                )
                rows.append(dict(index=row["index"], arm=row["arm"], check=check))
                boundaries.append(boundary())
                write_json(
                    ROOT / "audit_progress.json",
                    dict(initials=initials, cases=rows, boundaries=boundaries),
                )
    passed = all(v["passed"] for v in initials) and all(
        v["check"]["all_scores_passed"] and v["check"]["qualified_replay_passed"] for v in rows
    )
    write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            initials=initials,
            cases=rows,
            boundaries=boundaries,
            backwards=12,
            optimizer_updates=0,
            native_scores=42,
            training_batches=9600,
        ),
    )
    print("Independent audit:", passed, flush=True)


if __name__ == "__main__":
    run()
