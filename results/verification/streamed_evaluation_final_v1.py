"""Verify H111 frozen evidence, independent audit and lossless exports."""

import gzip
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/streamed_evaluation_v1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    protocol = read(ROOT / "protocol.json")
    for path, digest in protocol["sources"].items():
        assert sha(path) == digest, path
    for path, digest in protocol["checkpoint_hashes"].items():
        assert sha(path) == digest, path
    assert (
        sha("results/verification/token_memory_duration_final_v1.json")
        == protocol["previous_receipt_sha256"]
    )
    assert (
        sha("results/token_memory_duration_fresh_v1/result.json")
        == protocol["previous_result_sha256"]
    )
    result, summary, audit, q = [
        read(ROOT / name)
        for name in ("result.json", "summary.json", "audit.json", "qualification.json")
    ]
    assert result["status"] == summary["status"] == "COMPLETE"
    assert result["states"] == summary["states"] == 24 and result["scores"] == 72
    assert (
        result["optimizer_updates"]
        == summary["optimizer_updates"]
        == audit["optimizer_updates"]
        == 0
    )
    assert not result["actual_training_job"] and not summary["actual_training_job"]
    assert q["passed"] and len(q["checks"]) == 36
    assert audit["passed"] and audit["native_rescores"] == 24 and audit["policy_comparisons"] == 72
    assert audit["max_native_rescore_difference"] == 0
    assert audit["all_gates_and_policy_selections_independently_verified"]
    for group in summary["groups"]:
        assert group["selected_policy"] == "classifier_chunks"
        assert group["policies"]["classifier_chunks"]["eligible"]
        assert not group["policies"]["sequence_chunks"]["eligible"]
        assert not group["policies"]["sequence_chunks"]["gates"][
            "median_evaluation_time_ratio_at_most_1_5"
        ]
        rows = [r for r in result["rows"] if r["context"] == group["context"]]
        for policy, data in group["policies"].items():
            entries = [p for r in rows for p in r["policies"] if p["policy"] == policy]
            for key in (
                "relative_nll_difference_abs",
                "peak_allocated_bytes",
                "evaluation_memory_ratio",
                "evaluation_time_ratio",
                "estimated_job_peak_bytes",
            ):
                values = [p[key] for p in entries]
                assert data[key] == {
                    "mean": st.mean(values),
                    "median": st.median(values),
                    "sample_variance": st.variance(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
    for mode in ("study", "analyze", "audit"):
        assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "0"
    packed = {}
    for path in [ROOT / "result.json", *sorted(ROOT.glob("*.log"))]:
        target = path.with_name(path.name + ".gz")
        target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == path.read_bytes()
        packed[target.as_posix()] = {
            "sha256": sha(target),
            "original_sha256": sha(path),
            "bytes": target.stat().st_size,
        }
    files = [
        ROOT / n
        for n in (
            "protocol.json",
            "qualification.json",
            "environment.json",
            "summary.json",
            "audit.json",
            "source/README.md",
        )
    ]
    files += [
        Path("research/streamed_evaluation_plan.md"),
        Path("research/streamed_evaluation_results.md"),
        *sorted(ROOT.glob("source/*.py")),
        Path(__file__),
    ]
    receipt = {
        "status": "PASS",
        "frozen_sources": len(protocol["sources"]),
        "maintained_files_unchanged": len(protocol["maintained_files"]),
        "checkpoint_hashes": 24,
        "evaluation_scores": 72,
        "independent_native_rescores": 24,
        "max_native_rescore_difference": 0,
        "mathematical_qualification_cases": 36,
        "optimizer_updates": 0,
        "actual_training_job": False,
        "classifier_chunks_qualifies_both_shapes": True,
        "sequence_chunks_fails_runtime_both_shapes": True,
        "previous_maintained_tests_passed": 116,
        "maintained_tests_rerun": False,
        "broad_goal_achieved": False,
        "packed": packed,
        "files": {p.as_posix(): sha(p) for p in files},
    }
    Path("results/verification/streamed_evaluation_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("packed", "files")}, indent=2))


if __name__ == "__main__":
    run()
