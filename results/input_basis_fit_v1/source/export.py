"""Compact lossless release of H092-H095 metadata, endpoints and local inventory."""

import csv
import gzip
import hashlib
import io
import json
import os
import statistics as stats
import zipfile
from pathlib import Path

ROOTS = [
    Path("results") / name
    for name in (
        "input_basis_lift_v1",
        "input_basis_recovery_v1",
        "input_basis_fit_v1",
        "input_basis_fold_v1",
    )
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def exclusive(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    assert Path(path).read_bytes() == payload


def compressed(path, destination=None):
    payload = Path(path).read_bytes()
    destination = Path(destination) if destination else Path(str(path) + ".gz")
    exclusive(destination, gzip.compress(payload, mtime=0))
    assert gzip.decompress(destination.read_bytes()) == payload
    return {
        "path": destination.as_posix(),
        "sha256": sha(destination),
        "uncompressed_sha256": sha(path),
    }


def main():
    folded = read(ROOTS[-1] / "result.json")
    assert folded["status"] == "PASS" and read(ROOTS[2] / "result.json")["status"] == "PASS"
    manifests = []
    for root in ROOTS:
        for filename in ("before.json", "protocol.json"):
            manifests.append(compressed(root / filename))
    manifests.append(compressed(ROOTS[2] / "metrics.csv"))
    manifests.append(compressed(ROOTS[3] / "result.json"))
    summary = {
        k: v for k, v in folded.items() if k not in ("rows", "checkpoint_hashes", "observations")
    }
    summary["raw_result_sha256"] = sha(ROOTS[3] / "result.json")
    summary["per_checkpoint"] = [
        {
            "task": r["task"],
            "seed": r["seed"],
            "folded_mse": r["folded_mse"],
            "original_mse": r["original_mse"],
        }
        for r in folded["rows"]
    ]
    summary["mean_median_wall_ms"] = {
        f: stats.mean(r["timing"][f]["median_wall_ms"] for r in folded["rows"])
        for f in folded["mean_median_gpu_event_ms"]
    }
    summary["max_peak_bytes"] = {
        f: max(r["timing"][f]["peak_bytes"] for r in folded["rows"])
        for f in folded["mean_median_gpu_event_ms"]
    }
    exclusive(
        ROOTS[3] / "summary.json", (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
    )
    # Snapshot is assembled after execution, from the unmodified pre-execution hashes.
    hashes = read(ROOTS[3] / "before.json")["source_hashes"]
    with (ROOTS[3] / "source.zip").open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, digest in hashes.items():
                assert sha(name) == digest
                archive.write(name, name)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(ROOTS[3] / "source.zip") as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(p)).hexdigest() == h for p, h in hashes.items())
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("path", "bytes", "sha256"))
    total, count = 0, 0
    for root in ROOTS:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            size = path.stat().st_size
            writer.writerow((path.as_posix(), size, sha(path)))
            total += size
            count += 1
    inventory = ROOTS[2] / "artifact_inventory.csv"
    exclusive(inventory, output.getvalue().encode())
    inventory_record = compressed(inventory, "research/evidence/input_basis_artifacts.csv.gz")
    receipt = {
        "status": "PASS",
        "research_goal_achieved": False,
        "qualification_checks": 25,
        "training_runs": 336,
        "training_updates": 100800,
        "independent_checkpoint_scores_exact": 672,
        "folded_checkpoints_verified": 12,
        "active_model_folders": 2,
        "active_variants": 5,
        "recipes": 8,
        "active_python_sources_unchanged": all(
            sha(p) == h for p, h in read("research/evidence/release_v1.json")["sources"].items()
        ),
        "numerical_sources_unchanged": all(sha(p) == h for p, h in hashes.items()),
        "new_qualification_and_audits_run": True,
        "unchanged_active_tests_rerun": False,
        "figure_visually_reviewed": "results/input_basis_fit_v1/plots/reporting.png",
        "local_artifact_files": count,
        "local_artifact_bytes": total,
        "inventory": inventory_record,
        "lossless_gzip_exports": manifests,
        "export_source_sha256": sha(__file__),
        "post_audit_report_source_sha256": sha(ROOTS[2] / "source/report.py"),
        "source_snapshot_assembly": "after execution; exact pre-execution hashes verified",
        "anchors": {
            p.as_posix(): sha(p)
            for p in (ROOTS[1] / "result.json", ROOTS[2] / "result.json", ROOTS[3] / "summary.json")
        },
    }
    assert receipt["active_python_sources_unchanged"] and receipt["numerical_sources_unchanged"]
    exclusive(
        "research/evidence/input_basis_release.json",
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "files": count,
                "bytes": total,
                "inventory_bytes": Path(inventory_record["path"]).stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
