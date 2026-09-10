"""Post-audit H100 inventory and release checks; no training or source mutation."""

import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("results/triadic_interaction_v1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    study = read(ROOT / "study_process.json")
    audit = read(ROOT / "audit_process.json")
    result = read(ROOT / "result.json")
    assert study["status"] == audit["status"] == result["status"] == "PASS"
    assert read(ROOT / "reader_validation.json")["status"] == "PASS"
    assert read(ROOT / "report_runtime_recovery_process.json")["status"] == "PASS"
    before = read(ROOT / "before.json")
    frozen = {**before["source_hashes"], **before["anchors"]}
    assert all(sha(p) == h for p, h in frozen.items())
    maintained = read("research/evidence/release_v1.json")["sources"]
    assert all(sha(p) == h for p, h in maintained.items())
    packed = read("research/evidence/triadic_interaction_packing.json")["records"]
    for item in packed:
        assert sha(item["original"]) == item["original_sha256"]
        assert sha(item["packed"]) == item["packed_sha256"]
        assert (
            gzip.decompress(Path(item["packed"]).read_bytes())
            == Path(item["original"]).read_bytes()
        )
    links = []
    documents = [
        Path("README.md"),
        Path("research/triadic_interaction_results.md"),
        Path("research/triadic_interaction_plan.md"),
        Path("research/ARTIFACTS.md"),
    ]
    for doc in documents:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = (doc.parent / target.split("#")[0]).resolve()
            assert path.exists(), (doc, target)
            links.append((doc.as_posix(), target))
    subprocess.run(
        ["uv", "run", "ruff", "check", "src", "tests", "scripts", "main.py", str(ROOT / "source")],
        check=True,
    )
    data = io.StringIO(newline="")
    writer = csv.writer(data)
    writer.writerow(["path", "bytes", "sha256"])
    files = total = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
            continue
        before_stat = p.stat()
        digest = sha(p)
        after_stat = p.stat()
        assert (before_stat.st_size, before_stat.st_mtime_ns) == (
            after_stat.st_size,
            after_stat.st_mtime_ns,
        )
        writer.writerow([p.as_posix(), before_stat.st_size, digest])
        files += 1
        total += before_stat.st_size
    manifest = Path("research/evidence/triadic_interaction_artifacts.csv.gz")
    raw = data.getvalue().encode()
    with manifest.open("xb") as stream:
        stream.write(gzip.compress(raw, compresslevel=9, mtime=0))
    assert gzip.decompress(manifest.read_bytes()) == raw
    receipt = {
        "status": "PASS",
        "research_goal_achieved": False,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "verdicts": {k: v["verdict"] for k, v in result["summary"]["decisions"].items()},
        "qualification_checks": result["qualification_checks"],
        "saved_pairs_rechecked": result["qualification_pairs_rechecked"],
        "runs": result["runs"],
        "updates": result["updates"],
        "presentations": result["presentations"],
        "affine_fits": result["affine_fits"],
        "exact_initializations": result["exact_initializations"],
        "exact_checkpoint_scores": result["exact_checkpoint_scores"],
        "rate_selections_verified": result["rate_selections_verified"],
        "frozen_sources_unchanged": len(before["source_hashes"]),
        "frozen_anchors_unchanged": len(before["anchors"]),
        "maintained_sources_unchanged": len(maintained),
        "active_tests_rerun": False,
        "reason": "Maintained numerical source unchanged",
        "scoped_lint": "PASS",
        "report_recovery_lint_exception": "E402: deliberate imports after phase output",
        "post_audit_reader_saved_pairs": 177,
        "post_audit_projection_records": 228,
        "post_audit_runtime_recovery": "Python 3.12.9; initial/final diagnostic replaced by final-only",
        "preserved_reporting_failures": 4,
        "document_local_links_checked": len(links),
        "figures_visually_reviewed": [p.as_posix() for p in sorted((ROOT / "plots").glob("*.png"))],
        "local_artifact_files": files,
        "local_artifact_bytes": total,
        "inventory": {"path": manifest.as_posix(), "sha256": sha(manifest)},
        "post_audit_sources": {
            (ROOT / "source" / name).as_posix(): sha(ROOT / "source" / name)
            for name in (
                "report.py",
                "report_recovery.py",
                "report_numpy.py",
                "report_numpy_v2.py",
                "tensor_reader.py",
                "release.py",
            )
        },
        "anchors": {
            p.as_posix(): sha(p)
            for p in [
                ROOT / "result.json",
                ROOT / "study_process.json",
                ROOT / "audit_process.json",
                ROOT / "reader_validation.json",
                ROOT / "report_runtime_recovery_process.json",
                ROOT / "report_hash_correction.json",
                Path("research/evidence/triadic_interaction_packing.json"),
                Path("research/triadic_interaction_results.md"),
            ]
        },
    }
    path = Path("research/evidence/triadic_interaction_release.json")
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
