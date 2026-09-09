"""Final H064 artifact, preservation and documentation audit; no GPU model execution."""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

from src.core.config import VARIANTS

root = Path("results/rational_memory_v1")
out = Path("results/verification/rational_memory_final_v1.json")
assert not out.exists()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

result, protocol, before = [read(root / n) for n in ("result.json", "protocol.json", "before.json")]
assert result["status"] == "complete" and result["attribution_valid"]
assert not result["research_goal_achieved"] and not result["repair_implementation_earned"]
assert not result["corpus_scoring"] and result["optimizer_updates"] == 0
assert result["synthetic_target_exposures"] == 16384
assert all(all(row.values()) for row in result["comparisons"].values())
assert all(all(row["accounting"].values()) for row in result["analyses"].values())
assert sha("research/rational_memory_plan.md") == before["plan_sha256"] == protocol["plan_sha256"]
assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
assert all(sha(n) == h for n, h in before["h063_source_hashes"].items())
assert sha("results/verification/native_recompute_final_v1.json") == before["h063_final_sha256"]
assert sha("results/native_recompute_v1/protocol.json") == before["h063_protocol_sha256"]
assert sha(root / "allocator_source_audit.json") == protocol["allocator_source_audit_sha256"]
assert set(protocol["provenance"]["source_files"]) - set(before["h063_source_hashes"]) == {
    "src/core/allocation_trace.py", "src/core/rational_memory_audit.py", "tests/test_allocation_trace.py"
}
with zipfile.ZipFile(root / "source.zip") as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in protocol["provenance"]["source_files"].items())
    assert hashlib.sha256(archive.read("research/rational_memory_plan.md")).hexdigest() == protocol["plan_sha256"]

recovery = read(root / "qualification_recovery.json")
processes = {}
for path in (root / "processes").glob("*.json"):
    row = read(path)
    assert row["source_unchanged"] and sha(path.with_suffix(".log")) == row["log_sha256"]
    log = path.with_suffix(".log").read_text(encoding="utf-8")
    if path.stem == "full_tests":
        assert row["status"] == "FAIL" and row["returncode"] != 0
        assert "107 passed" in log and "3221225477" in log and "PTXASError" in log
    else:
        assert row["status"] == "PASS" and row["returncode"] == 0
    for name, expected in row["sources"].items():
        if path.stem in ("full_tests", "failed_triton_test_retry") and name == recovery["changed_file"]:
            assert expected == recovery["before_sha256"] and sha(name) == recovery["after_sha256"]
        else:
            assert sha(name) == expected, (path, name)
    processes[path.stem] = {"sha256": sha(path), "status": row["status"]}
assert len(processes) == 9
assert "1 passed" in (root / "processes/failed_triton_test_retry.log").read_text()
assert "108 passed in 25.41s" in (root / "processes/full_tests_after_compiler_failure.log").read_text()
assert read(root / "preflight.json")["protocol_sha256"] == sha(root / "protocol.json")

artifacts, details = {}, {}
for cell in protocol["cells"]:
    folder = root / "workers" / cell
    row, config = read(folder / "result.json"), read(folder / "config.json")
    assert row["status"] == "PASS" and row["optimizer_updates"] == 0
    assert row["weights_moments_unchanged"] and row["rng_unchanged"] and row["all_gradients_finite"]
    assert row["synthetic_target_exposures"] == 4096 and not row["corpus_scoring"]
    assert row["token_record"] == protocol["token_record"]
    assert config["protocol_sha256"] == sha(root / "protocol.json") and config["scope"] == "block"
    assert config["model"]["width"] == 384 and config["model"]["layers"] == 8
    assert sha(folder / "signature.json") == row["signature_sha256"]
    assert all(sha(folder / n) == h for n, h in row["snapshot_hashes"].items())
    if row["traced"]:
        assert set(row["snapshot_hashes"]) == {"baseline.pickle", "forward_end.pickle", "backward_end.pickle"}
        assert "record_context_cpp is not supported" in (root / "processes" / f"worker_{cell}.log").read_text()
        replay, attribution = read(folder / "replay.json"), read(folder / "peak_attribution.json")
        key = "rational" if cell.startswith("rational") else "plain"
        assert replay["events"] < protocol["trace_cap"] and replay["peak_phase"] == "backward"
        assert replay["peak_bytes"] == row["marks"]["backward_end"]["peak_allocated"]
        assert sum(attribution["category_bytes"].values()) == replay["peak_bytes"]
        assert attribution["category_bytes"] == result["analyses"][key]["category_bytes"]
        pointwise = [b for b in attribution["allocations"] if b["category"] == "rational_pointwise"]
        if key == "rational":
            assert len(pointwise) == 11 and sum(b["size"] == 16 * 2**20 for b in pointwise) == 8
            assert all(any(f.get("name") == "recompute_fn" for f in b["frames"]) for b in pointwise)
        else:
            assert not pointwise
        details[key] = {
            "peak_allocated_bytes": replay["peak_bytes"],
            "rational_pointwise_bytes": sum(b["size"] for b in pointwise),
            "unmatched_runtime_empty_frames_bytes": sum(b["size"] for b in attribution["allocations"] if b["category"] == "other_runtime" and not b["frames"]),
            "unmatched_runtime_other_frames_bytes": sum(b["size"] for b in attribution["allocations"] if b["category"] == "other_runtime" and b["frames"]),
            "other_preexisting_bytes": attribution["category_bytes"]["other_preexisting"],
        }
    else:
        assert not row["snapshot_hashes"]
    artifacts[cell] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
for recipe in ("plain", "rational"):
    untraced, traced = [read(root / "workers" / f"{recipe}_{mode}" / "result.json") for mode in ("untraced", "traced")]
    assert untraced["signature_sha256"] == traced["signature_sha256"]
    assert untraced["marks"] == traced["marks"]
for row in protocol["inputs"].values():
    assert all(sha(Path("results/runs") / row["run"] / n) == h for n, h in row["files"].items())

previous = read("results/verification/native_recompute_final_v1.json")
for cell, files in previous["worker_artifact_hashes"].items():
    assert all(sha(Path("results/native_recompute_v1/workers") / cell / n) == h for n, h in files.items())
assert sha("results/native_recompute_v1/result.json") == previous["result_sha256"]
assert sha("results/native_recompute_v1/source.zip") == previous["source_archive_sha256"]
assert sha("results/native_recompute_v1/support.zip") == previous["support_archive_sha256"]
archive_manifest = read("research/archive/h061_manifest.json")
for name, expected in archive_manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, name
plans = {p.as_posix(): sha(p) for p in Path("research").glob("*plan.md")}
assert len(plans) == 51 and all(plans[n] == h for n, h in previous["frozen_plans"].items())
assert len(list(Path("results/runs").glob("*/metrics.json"))) == 171
assert len(VARIANTS) == 6 and len(list(Path("configs").glob("*.json"))) == 9
assert {p.name for p in Path("src").iterdir() if p.is_dir() and p.name.endswith("_ffn")} == {
    "dense_ffn", "blockshuffle_ffn", "rational_blockshuffle_ffn"
}
links, historical = 0, []
old_manifest = read("research/archive/h057_manifest.json")
for p in [Path("README.md"), *Path("src").rglob("*.md"), *Path("doc").glob("*.md"), *Path("configs").glob("*.md"), *Path("research").rglob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        resolved = (p.parent / clean).resolve()
        if resolved == out.resolve():
            continue
        if resolved.exists():
            links += 1
            continue
        assert p.parent == Path("research") and p.name.endswith("plan.md"), (p, clean)
        intended = resolved.relative_to(Path.cwd()).as_posix()
        assert intended in old_manifest["files"] and (Path("research/archive/retired") / intended).exists()
        historical.append({"plan": p.as_posix(), "target": clean})
assert len(historical) == 3
checks = {}
for name, args in {"lint": ["-m", "ruff", "check", "src", "tests"], "counts": ["-m", "src.core.cli", "counts"]}.items():
    process = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert process.returncode == 0, (name, process.stdout, process.stderr)
    checks[name] = process.stdout
assert {json.loads(line)["variant"] for line in checks["counts"].splitlines()} == set(VARIANTS)
support = [*Path("results/verification").glob("rational_memory*.py"), Path("README.md"), Path("research/rational_memory_plan.md"), Path("research/rational_memory_results.md"), Path("research/CURRENT_STATE.md"), Path("research/idea_bank.md"), Path("src/core/README.md"), Path("src/rational_blockshuffle_ffn/README.md"), Path("research/figures/rational_memory.png"), Path("research/figures/rational_memory.svg")]
with zipfile.ZipFile(root / "support.zip", "x", zipfile.ZIP_DEFLATED) as archive:
    for p in support:
        archive.write(p, p.as_posix())
record = {
    "status": "PASS", "goal_turn": "PROGRESS", "research_goal_achieved": False,
    "accounting_and_traced_fidelity_qualified_locally": True, "source_attribution": "partial",
    "repair_implementation_earned": False, "new_architecture_variants": 0,
    "tests_passed": 108, "workers_completed": 4, "worker_retries": 0,
    "optimizer_updates": 0, "synthetic_target_exposures": 16384,
    "corpus_training_or_scoring_added": False, "retained_lm_profile_runs": 171,
    "all_h063_source_configuration_tests_unchanged": True,
    "all_h063_worker_artifact_hashes_unchanged": True,
    "protected_evidence_files_metadata_unchanged": len(archive_manifest["protected_evidence_metadata"]),
    "frozen_plans": plans, "processes": processes, "compiler_failure_preserved": True,
    "compiler_failure_root_cause_established": False, "cpp_allocation_stacks_available": False,
    "allocation_details": details, "source_hashes": protocol["provenance"]["source_files"],
    "worker_artifact_hashes": artifacts, "checks": checks,
    "local_links_verified": links, "historical_links_resolve_in_archive": historical,
    "result_sha256": sha(root / "result.json"), "protocol_sha256": sha(root / "protocol.json"),
    "source_archive_sha256": sha(root / "source.zip"), "support_archive_sha256": sha(root / "support.zip"),
    "documents": {p.as_posix(): sha(p) for p in support if p.suffix == ".md"},
    "figure_visually_checked": True,
}
with out.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in record.items() if k not in ("source_hashes", "worker_artifact_hashes", "checks", "processes", "frozen_plans", "documents")}))
