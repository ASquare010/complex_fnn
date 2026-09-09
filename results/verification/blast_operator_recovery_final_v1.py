"""Final H080 preservation and evidence audit; standard library only."""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/blast_operator_recovery_v1")
OUT = Path("results/verification/blast_operator_recovery_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    payload = (json.dumps(value, indent=2) + "\n").encode()
    with Path(path).open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    assert Path(path).read_bytes() == payload


def archive(path, files):
    with Path(path).open("xb") as f:
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
            for n in files:
                z.write(n, Path(n).as_posix())
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)
        for n in files:
            assert hashlib.sha256(z.read(Path(n).as_posix())).hexdigest() == sha(n)


before, p, r = [read(ROOT / n) for n in ("before.json", "protocol.json", "result.json")]
a_path = Path("results/verification/blast_operator_recovery_analysis_v2.json")
a = read(a_path)
assert a["status"] == "PASS" and r["status"] == "LOCALLY_QUALIFIED"
assert a["qualified_checks"] == r["checks_passed"] == 34
assert a["saved_tensor_files_verified"] == 32 and a["checkpoint_tensor_pairs_exact"] == 114
assert a["projection_tensor_pairs_within_tolerance"] == 60
assert a["same_width_embeddings_exact"] == a["hadamard_certificates_exact"] == 4
assert a["squared_relative_rank6_error_bound"] == [7, 8]
assert len(a["prior_readable_tensor_payload_equality"]) == 24
assert all(a["prior_readable_tensor_payload_equality"].values())
assert sum(a["prior_json_tensor_hash_matches"].values()) == 17
assert len(a["prior_json_tensor_hash_matches"]) == 18
assert r["optimizer_updates"] == r["corpus_targets"] == r["full_transformer_runs"] == 0
assert r["explicit_qualification_repetitions"] == 1 and r["total_h079_h080_attempts"] == 2
assert not r["research_goal_achieved"]
assert len(p["sources"]) == 118 and all(sha(n) == h for n, h in p["sources"].items())
assert a["result_sha256"] == sha(ROOT / "result.json")
assert a["observation_hashes"] == r["observation_hashes"]
assert len(r["observation_hashes"]) == 66
assert all(sha(n) == h for n, h in r["observation_hashes"].items())
assert sha("research/blast_operator_recovery_plan.md") == p["plan_sha256"] == before["plan_sha256"]
assert sha("research/blast_operator_theory.md") == p["theory_sha256"] == before["h079_theory_sha256"]
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
for phase in ("collection", "qualification", "analysis_v2", "report"):
    e = read(ROOT / (phase + "_process.json"))
    assert e["status"] == "PASS" and e["returncode"] == 0 and e["source_unchanged"]
    assert sha(ROOT / (phase + ".log")) == e["log_sha256"]
for phase, version in (("analysis_v2", 2), ("report", 1)):
    e = read(ROOT / (phase + "_process.json"))
    name = "analysis" if phase == "analysis_v2" else phase
    assert e["scientific_source_unchanged"] and e["qualification_repetitions"] == 0
    assert sha(f"results/verification/blast_operator_recovery_{name}_v{version}.py") == e["source_sha256"]
correction = read(ROOT / "analysis_correction.json")
assert not correction["qualification_thresholds_changed"]
assert correction["qualification_repetitions"] == correction["optimizer_updates"] == 0
assert read(ROOT / "analysis_process.json")["status"] == "FAIL"
for name, key in (("analysis_process.json", "prior_analysis_process_sha256"),
                  ("analysis.log", "prior_analysis_log_sha256")):
    assert sha(ROOT / name) == correction[key]
assert sha("results/verification/blast_operator_recovery_analysis_v1.py") == correction["prior_analysis_source_sha256"]
assert sha("results/verification/blast_operator_recovery_analysis_v2.py") == correction["new_analysis_source_sha256"]
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None and len(z.namelist()) == 121
    for n, h in p["sources"].items():
        assert hashlib.sha256(z.read(n)).hexdigest() == h
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())

old_root = Path("results/blast_operator_v1")
integrity_path = old_root / "integrity_observation.json"
assert sha(integrity_path) == before["h079_integrity_observation_sha256"]
integrity = read(integrity_path)
for n, v in integrity["files"].items():
    assert sha(n) == v["sha256"] and Path(n).stat().st_size == v["size"]
    assert Path(n).stat().st_mtime_ns == v["mtime_ns"]
assert sum(v["all_zero"] for v in integrity["files"].values()) == 25
for name, key in (("protocol.json", "h079_protocol_sha256"), ("source.zip", "h079_source_sha256"),
                  ("process.json", "h079_process_sha256")):
    assert sha(old_root / name) == before[key]
assert sha("results/verification/blast_operator_integrity_final_v1.json") == before["h079_integrity_final_sha256"]
old_final = read("results/verification/blast_operator_integrity_final_v1.json")
assert sha("research/blast_operator_results.md") == old_final["report_sha256"]

# Preserve the latest four complete language/resource stages and all their checkpoints.
prior078_path = Path("results/verification/ungated_duration_final_v1.json")
assert sha(prior078_path) == before["h078_final_sha256"]
prior078 = read(prior078_path)
for label, folder, count, expected in (
    ("ungated_duration", "ungated_duration_v1", 6, before["h078_final_sha256"]),
    ("ungated_lm_replication", "ungated_lm_replication_v1", 18, prior078["prior_h077_final_sha256"]),
    ("ungated_lm_screen", "ungated_lm_screen_v1", 21, prior078["prior_h076_final_sha256"]),
):
    final_path = Path(f"results/verification/{label}_final_v1.json")
    assert sha(final_path) == expected
    final = read(final_path)
    root = Path("results") / folder
    for filename, key in (("protocol.json", "protocol_sha256"), ("result.json", "result_sha256"),
                          ("source.zip", "source_archive_sha256"), ("support.zip", "support_archive_sha256")):
        assert sha(root / filename) == final[key]
    analysis_path = Path(f"results/verification/{label}_analysis_v1.json")
    assert sha(analysis_path) == final["analysis_sha256"]
    old = read(analysis_path)
    assert len(old["cell_artifact_hashes"]) == count
    for cell, files in old["cell_artifact_hashes"].items():
        assert all(sha(root / "runs" / cell / n) == h for n, h in files.items())
    with zipfile.ZipFile(root / "support.zip") as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in final["support_artifacts"].items())
resource_final_path = Path("results/verification/ungated_resource_recovery_final_v1.json")
assert sha(resource_final_path) == prior078["prior_h075_final_sha256"]
resource_final = read(resource_final_path)
resource_root = Path("results/ungated_resource_recovery_v1")
assert sha("results/verification/ungated_resource_recovery_analysis_v1.json") == resource_final["analysis_sha256"]
resource_analysis = read("results/verification/ungated_resource_recovery_analysis_v1.json")
resource_protocol = read(resource_root / "protocol.json")
assert len(resource_analysis["worker_artifact_hashes"]) == 21
for cell, files in resource_analysis["worker_artifact_hashes"].items():
    assert all(sha(Path(resource_protocol["origins"][cell]) / "workers" / cell / n) == h for n, h in files.items())
for filename, key in (("protocol.json", "protocol_sha256"), ("result.json", "result_sha256"),
                      ("source.zip", "source_archive_sha256"), ("support.zip", "support_archive_sha256")):
    assert sha(resource_root / filename) == resource_final[key]
assert not Path("results/ungated_resource_v1/result.json").exists()
assert not list(Path("results/ungated_resource_v1/workers/gelu_same_block_inner").iterdir())
fitting_count = 0
for label, folder in (("ungated_fit", "ungated_fit_v1"), ("rotated_shuffle_fit", "rotated_shuffle_fit_v1"),
                      ("token_activation_fit", "token_activation_fit_v2")):
    final = read(f"results/verification/{label}_final_v1.json")
    path = Path(f"results/verification/{label}_analysis_v1.json")
    assert sha(path) == final["analysis_sha256"]
    old = read(path)
    for cell, files in old["cell_artifact_hashes"].items():
        assert all(sha(Path("results") / folder / "cells" / cell / n) == h for n, h in files.items())
    fitting_count += len(old["cell_artifact_hashes"])
assert fitting_count == 798
manifest = read("research/archive/h061_manifest.json")
for n, v in manifest["protected_evidence_metadata"].items():
    stat = Path(n).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == v
old_protocol = read("results/ungated_duration_v1/protocol.json")
assert all(sha(Path("data/wikitext2_v1") / n) == h for n, h in old_protocol["data_files"].items())
assert sha("data/wikitext2_v1/manifest.json") == old_protocol["data_manifest_sha256"]
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 69 and all(plans[n] == h for n, h in before["prior_plans"].items())
assert {n.name for n in Path("src").iterdir() if n.is_dir() and n.name.endswith("_ffn")} == {
    "dense_ffn", "blockshuffle_ffn", "rational_blockshuffle_ffn",
}
tree = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets))
assert len(variants) == 6 and len(list(Path("configs").glob("*.json"))) == 9
assert len(list(Path("results/runs").glob("*/metrics.json"))) == 171

links, historical = 0, []
archived = read("research/archive/h057_manifest.json")
for doc in [Path("README.md"), *Path("src").rglob("*.md"), *Path("doc").glob("*.md"),
            *Path("configs").glob("*.md"), *Path("research").rglob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        resolved = (doc.parent / clean).resolve()
        if resolved == OUT.resolve():
            continue
        if resolved.exists():
            links += 1
            continue
        assert doc.parent == Path("research") and doc.name.endswith("plan.md"), (doc, clean)
        intended = resolved.relative_to(Path.cwd()).as_posix()
        assert intended in archived["files"] and (Path("research/archive/retired") / intended).exists()
        historical.append({"plan": doc.as_posix(), "target": clean})
assert len(historical) == 3
lint = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "tests",
                       "results/blast_operator_v1/source", str(ROOT / "source"),
                       *map(str, Path("results/verification").glob("blast_operator_recovery*.py"))],
                      capture_output=True, text=True)
assert lint.returncode == 0, lint.stdout + lint.stderr

metadata = [ROOT / "protocol.json", ROOT / "result.json", ROOT / "coordinator_status.json",
            *ROOT.glob("*_process.json"), *ROOT.glob("*.log"),
            *ROOT.glob("*correction.json"), ROOT / "report_source_write_observation.json",
            *sorted((ROOT / "observations").glob("*.json"))]
metadata = sorted(set(metadata))
archive(ROOT / "evidence_metadata.zip", metadata)
support = [Path("README.md"), Path("research/CURRENT_STATE.md"), Path("research/idea_bank.md"),
           Path("research/PROGRESS_OVERVIEW.md"), Path("research/blast_operator_results.md"),
           Path("research/blast_operator_recovery_results.md"), Path("research/blast_operator_theory.md"),
           Path("research/blast_operator_plan.md"), Path("research/blast_operator_recovery_plan.md"),
           *Path("results/verification").glob("blast_operator_recovery*.py"),
           Path("results/verification/blast_operator_integrity_final_v1.json"), integrity_path,
           a_path, ROOT / "analysis_correction.json", ROOT / "evidence_metadata.zip"]
support = sorted(set(support))
archive(ROOT / "support.zip", support)
record = {
    "status": "PASS", "goal_turn": "PROGRESS", "scientific_status": "LOCALLY_QUALIFIED_COMPARATOR",
    "research_goal_achieved": False, "qualified_checks": 34, "saved_tensor_files_verified": 32,
    "exact_checkpoint_tensor_pairs": 114, "projection_pairs": 60, "optimizer_updates": 0,
    "corpus_targets": 0, "full_transformer_runs": 0, "explicit_qualification_repetitions": 1,
    "total_h079_h080_attempts": 2, "h079_status": "INCOMPLETE_ARTIFACT_INTEGRITY",
    "original_h079_files_preserved": len(integrity["files"]), "h079_zero_filled_files_preserved": 25,
    "prior_readable_tensor_payloads_identical": 24, "audit_v1_failure_preserved": True,
    "audit_v2_exact_original_device_norms": True, "qualification_thresholds_changed": False,
    "prior_h078_final_sha256": before["h078_final_sha256"], "prior_lm_profile_runs_preserved": 171,
    "prior_h076_h077_h078_lm_checkpoints_verified": 45, "prior_resource_pairs_verified": 21,
    "prior_fitting_cells_verified": fitting_count,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "active_model_folders": 3, "active_variants": 6, "recipe_files": 9, "prior_active_suite_tests": 108,
    "active_source_tests_configurations_unchanged": True, "source_hashes": p["sources"],
    "frozen_plans": plans, "protocol_sha256": sha(ROOT / "protocol.json"),
    "result_sha256": sha(ROOT / "result.json"), "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"), "analysis_sha256": sha(a_path),
    "evidence_metadata_archive_sha256": sha(ROOT / "evidence_metadata.zip"),
    "metadata_artifacts": {n.as_posix(): sha(n) for n in metadata},
    "support_artifacts": {n.as_posix(): sha(n) for n in support},
    "local_links_verified": links, "historical_archive_links": historical, "lint": lint.stdout,
}
write(OUT, record)
print(json.dumps({k: v for k, v in record.items()
                  if k not in ("source_hashes", "frozen_plans", "metadata_artifacts", "support_artifacts")}), flush=True)

