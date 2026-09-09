"""Audit H065 measured tensors, source preservation and documentation."""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote
import torch
from src.core.native_recompute_audit import digest, read, sha, tensor_record
from src.core.config import VARIANTS

root = Path("results/rational_staged_v1")
out = Path("results/verification/rational_staged_final_v1.json")
assert not out.exists()
before, protocol, result, process = [read(root / n) for n in ("before.json", "protocol.json", "result.json", "process.json")]
assert result["status"] == "complete" and result["scientific_verdict"] == "LOCALLY_QUALIFIED"
assert result["all_cases_exact"] and result["earns_full_model_qualification"]
assert result["optimizer_updates"] == result["corpus_targets"] == result["full_model_resource_workers"] == 0
assert not result["research_goal_achieved"]
assert sha("research/rational_staged_plan.md") == protocol["plan_sha256"] == before["plan_sha256"]
assert all(sha(n) == h for n, h in protocol["sources"].items())
assert all(sha(n) == h for n, h in before["h064_source_hashes"].items())
assert process["status"] == "PASS" and process["source_unchanged"] and process["returncode"] == 0
assert sha(root / "process.log") == process["log_sha256"]
assert sha(root / "protocol.json") == process["protocol_sha256"]
assert sha(root / "source.zip") == process["source_archive_sha256"]
with zipfile.ZipFile(root / "source.zip") as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n,h in protocol["sources"].items())
assert read(root / "mathematical_checks.json") == result["mathematical_checks"]
assert result["mathematical_checks"]["status"] == "PASS"
artifacts, comparisons = {}, 0
for cell, row in result["cases"].items():
    folder = root / "cases" / cell
    assert read(folder / "result.json") == row
    assert sha(folder / "tensors.pt") == row["tensors_sha256"]
    stored = torch.load(folder / "tensors.pt", map_location="cpu", weights_only=True)
    assert digest(stored["state"]) == row["state_hash"]
    assert [tensor_record(t) for t in stored["input_fp32"]] == row["input_fp32_hashes"]
    assert row["exact"] and row["finite"] and row["rng_unchanged"] and row["parameters_unchanged"]
    left, right = stored["outputs"]["native"], stored["outputs"]["staged"]
    pairs = {"output": (left["output"], right["output"]), **{n:(left["gradients"][n], right["gradients"][n]) for n in left["gradients"]}}
    for name, (a,b) in pairs.items():
        check = row["comparisons"][name]
        assert check["exact"] and check["finite"] and check["mismatched_elements"] == 0
        assert check["max_abs_error"] == check["relative_l2_error"] == 0
        assert tensor_record(a) == tensor_record(b) == check["reference"] == check["candidate"]
        comparisons += 1
    artifacts[cell] = {p.name:sha(p) for p in folder.iterdir() if p.is_file()}
assert len(artifacts) == 12 and comparisons == 60
original = Path("results/runs") / protocol["input"]["run"]
assert all(sha(original / n) == h for n,h in protocol["input"]["files"].items())
assert sha("results/verification/rational_memory_final_v1.json") == before["h064_final_sha256"]
old = read("results/verification/rational_memory_final_v1.json")
for cell, files in old["worker_artifact_hashes"].items():
    assert all(sha(Path("results/rational_memory_v1/workers") / cell / n) == h for n,h in files.items())
manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size":stat.st_size,"mtime_ns":stat.st_mtime_ns} == expected
plans = {p.as_posix():sha(p) for p in Path("research").glob("*plan.md")}
assert len(plans) == 52 and all(plans[n] == h for n,h in old["frozen_plans"].items())
assert len(VARIANTS) == 6 and len(list(Path("configs").glob("*.json"))) == 9
assert len(list(Path("results/runs").glob("*/metrics.json"))) == 171
old_manifest = read("research/archive/h057_manifest.json")
links, historical = 0, []
for p in [Path("README.md"),*Path("src").rglob("*.md"),*Path("doc").glob("*.md"),*Path("configs").glob("*.md"),*Path("research").rglob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://",clean): continue
        resolved = (p.parent / clean).resolve()
        if resolved == out.resolve(): continue
        if resolved.exists(): links += 1; continue
        assert p.parent == Path("research") and p.name.endswith("plan.md"), (p,clean)
        intended = resolved.relative_to(Path.cwd()).as_posix()
        assert intended in old_manifest["files"] and (Path("research/archive/retired") / intended).exists()
        historical.append({"plan":p.as_posix(),"target":clean})
assert len(historical) == 3
lint = subprocess.run([sys.executable,"-m","ruff","check","src","tests",str(root / "source")],capture_output=True,text=True)
assert lint.returncode == 0, lint.stdout
support = [Path(__file__),Path("research/rational_staged_plan.md"),Path("research/rational_staged_results.md"),Path("research/CURRENT_STATE.md"),Path("research/idea_bank.md")]
with zipfile.ZipFile(root / "support.zip","x",zipfile.ZIP_DEFLATED) as archive:
    for p in support: archive.write(p,p.relative_to(Path.cwd()).as_posix() if p.is_absolute() else p.as_posix())
record = {
    "status":"PASS","goal_turn":"PROGRESS","scientific_verdict":"LOCALLY_QUALIFIED","research_goal_achieved":False,
    "paired_cases":12,"exact_tensor_comparisons":60,"optimizer_updates":0,"corpus_targets":0,
    "full_model_resource_workers":0,"earns_full_model_qualification":True,"process_retries":0,
    "active_variants":6,"retained_lm_profile_runs":171,"active_suite_unchanged_from_h064_tests":108,
    "all_h064_sources_and_worker_artifacts_unchanged":True,
    "protected_evidence_files_metadata_unchanged":len(manifest["protected_evidence_metadata"]),
    "frozen_plans":plans,"source_hashes":protocol["sources"],"case_artifact_hashes":artifacts,
    "local_links_verified":links,"historical_links_resolve_in_archive":historical,
    "result_sha256":sha(root / "result.json"),"protocol_sha256":sha(root / "protocol.json"),
    "source_archive_sha256":sha(root / "source.zip"),"support_archive_sha256":sha(root / "support.zip"),
    "process_sha256":sha(root / "process.json"),"lint":lint.stdout,
}
with out.open("x",encoding="utf-8") as handle: handle.write(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("source_hashes","case_artifact_hashes","frozen_plans","historical_links_resolve_in_archive")}))
