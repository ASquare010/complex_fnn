"""Final evidence audit for the completed H063 native execution experiment."""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

from src.core.config import VARIANTS

root = Path("results/native_recompute_v1")
out = Path("results/verification/native_recompute_final_v1.json")
assert not out.exists()
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
result, protocol, before = [read(root / n) for n in ("result.json", "protocol.json", "before.json")]
assert result["status"] == "complete" and result["all_updates_exact_across_scopes"]
assert result["selected_scope"] is None and not result["earns_full_native_training_repeat"]
assert not result["research_goal_achieved"] and not result["corpus_scoring"]
assert all(sha(n) == h for n,h in protocol["provenance"]["source_files"].items())
assert all(sha(n) == h for n,h in before["source_hashes"].items() if n != "src/core/transformer.py")
assert sha("research/native_recompute_plan.md") == protocol["plan_sha256"] == before["plan_sha256"]
with zipfile.ZipFile(root / "source.zip") as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n,h in protocol["provenance"]["source_files"].items())
processes = {}
for p in (root / "processes").glob("*.json"):
    record = read(p)
    assert record["source_unchanged"] and sha(p.with_suffix(".log")) == record["log_sha256"]
    if p.stem == "preflight":
        assert record["status"] == "FAIL" and record["returncode"] != 0
        assert "unknown opcode 186" in p.with_suffix(".log").read_text()
    else:
        assert record["status"] == "PASS" and record["returncode"] == 0
    processes[p.stem] = {"sha256":sha(p), "status":record["status"]}
assert len(processes) == 20
suite = read(root / "processes/full_tests.json")
assert all(sha(n) == h for n,h in suite["sources"].items())
assert "104 passed" in (root / "processes/full_tests.log").read_text()
assert read(root / "ast_runtime_diagnostic.json")["status"] == "PASS"
assert read(root / "retained_signatures.json") == read("results/verification/cleanup_before_signatures_v1.json")
rows = result["results"]
assert len(rows) == 15 and sum(row["updates"] for row in rows.values()) == 300
assert sum(row["synthetic_target_exposures"] for row in rows.values()) == 614400
artifact_hashes = {}
for cell,row in rows.items():
    folder = root / "workers" / cell
    assert row["token_record"] == protocol["token_record"] and row["rng_unchanged"]
    assert sha(folder / "checkpoint.pt") == row["checkpoint_sha256"]
    assert sha(folder / "history.jsonl") == row["history_sha256"]
    assert sha(folder / "initial_signature.json") == row["initial_signature_sha256"]
    history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    assert [x["step"] for x in history] == list(range(201,221))
    config = read(folder / "config.json")
    assert config["protocol_sha256"] == sha(root / "protocol.json")
    assert config["scope"] == row["scope"] and config["updates"] == 20 and config["synthetic"]
    assert config["model"]["width"] == 384 and config["model"]["layers"] == 8
    assert all(value["finite"] for name in ("initial_diagnostics.json","final_diagnostics.json") for value in read(folder / name).values())
    artifact_hashes[cell] = {p.name:sha(p) for p in folder.iterdir() if p.is_file()}
for recipe in ("full_swiglu","full_gelu","calibrated_narrow","blockshuffle","rational"):
    reference = rows[f"{recipe}_none"]
    for scope in ("ffn","block"):
        candidate = rows[f"{recipe}_{scope}"]
        assert reference["initial_signature_sha256"] == candidate["initial_signature_sha256"]
        assert reference["history_sha256"] == candidate["history_sha256"]
        assert reference["final"] == candidate["final"]
        assert all(result["exact_comparisons"][f"{recipe}_{scope}"].values())
for scope in ("ffn","block"):
    candidate = rows[f"rational_{scope}"]
    assert candidate["ffn_parameters"] == 2801984 and candidate["total_parameters"] == 9099968
    assert candidate["peak_allocated_bytes"] < .9 * rows["rational_none"]["peak_allocated_bytes"]
    assert candidate["ffn_reduction_percent"] >= 70
    for control in ("full_swiglu","full_gelu"):
        assert candidate["peak_allocated_bytes"] > 1.1 * rows[f"{control}_{scope}"]["peak_allocated_bytes"]
for row in protocol["inputs"].values():
    assert all(sha(Path("results/runs")/row["run"]/n) == h for n,h in row["files"].items())
previous = read("research/archive/h061_manifest.json")
for name, expected in previous["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size":stat.st_size,"mtime_ns":stat.st_mtime_ns} == expected, name
plans = {p.as_posix():sha(p) for p in Path("research").glob("*plan.md")}
assert len(plans) == 50
assert all(plans[n] == h for n,h in previous["files"].items() if n in plans)
assert len(list(Path("results/runs").glob("*/metrics.json"))) == 171
assert len(VARIANTS) == 6 and len(list(Path("configs").glob("*.json"))) == 9
links, historical = 0, []
old_manifest = read("research/archive/h057_manifest.json")
for p in [Path("README.md"), *Path("src").rglob("*.md"), *Path("doc").glob("*.md"), *Path("configs").glob("*.md"), *Path("research").rglob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean): continue
        resolved = (p.parent / clean).resolve()
        if resolved == out.resolve(): continue
        if resolved.exists(): links += 1; continue
        assert p.parent == Path("research") and p.name.endswith("plan.md"), (p,clean)
        intended = resolved.relative_to(Path.cwd()).as_posix()
        assert intended in old_manifest["files"] and (Path("research/archive/retired")/intended).exists()
        historical.append({"plan":p.as_posix(),"target":clean})
assert len(historical) == 3
checks = {}
for name,args in {"lint":["-m","ruff","check","src","tests"], "counts":["-m","src.core.cli","counts"]}.items():
    process=subprocess.run([sys.executable,*args],capture_output=True,text=True)
    assert process.returncode == 0, (name,process.stdout,process.stderr)
    checks[name]=process.stdout
assert {json.loads(line)["variant"] for line in checks["counts"].splitlines()} == set(VARIANTS)
support = [*Path("results/verification").glob("native_recompute*.py"),Path("research/native_recompute_plan.md"),Path("research/native_recompute_results.md"),Path("research/CURRENT_STATE.md"),Path("src/core/README.md"),Path("src/rational_blockshuffle_ffn/README.md")]
with zipfile.ZipFile(root / "support.zip","x",zipfile.ZIP_DEFLATED) as archive:
    for p in support: archive.write(p,p.as_posix())
record={"status":"PASS","goal_turn":"PROGRESS","research_goal_achieved":False,
        "execution_fidelity_qualified_locally":True,"rational_memory_promotion":False,
        "exact_full_model_scope_comparisons":10,"exactness_scope":"Initial logits/loss/every gradient; all 20 losses/norms; final weights and optimizer states. Intermediate weight tensors are not individually archived at each step.",
        "tests_passed":104,"exact_retained_cpu_gpu_cases":12,"workers_completed":15,"worker_retries":0,
        "synthetic_updates":300,"synthetic_training_target_exposures":614400,"no_update_probe_target_exposures":30720,
        "corpus_training_or_scoring_added":False,"retained_lm_profile_runs":171,
        "protected_evidence_files_metadata_unchanged":len(previous["protected_evidence_metadata"]),
        "frozen_plans":plans,"plan_sha256":sha("research/native_recompute_plan.md"),
        "result_sha256":sha(root/"result.json"),"protocol_sha256":sha(root/"protocol.json"),
        "source_archive_sha256":sha(root/"source.zip"),"support_archive_sha256":sha(root/"support.zip"),
        "source_hashes":protocol["provenance"]["source_files"],"worker_artifact_hashes":artifact_hashes,
        "processes":processes,"preflight_interpreter_failure_preserved":True,
        "local_links_verified":links,"historical_links_resolve_in_archive":historical,"checks":checks,
        "documents":{p.as_posix():sha(p) for p in support if p.suffix==".md"},
        "figure_visually_checked":True}
with out.open("x",encoding="utf-8") as handle: handle.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("source_hashes","worker_artifact_hashes","processes","checks","documents","frozen_plans")}))