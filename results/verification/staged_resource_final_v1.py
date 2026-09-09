"""Final H066 evidence and preservation audit; no new model execution."""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote
import torch
from src.core.config import VARIANTS
from src.core.native_recompute_audit import digest, finite_tree, read, sha

root=Path("results/staged_resource_v1")
out=Path("results/verification/staged_resource_final_v1.json")
assert not out.exists()
before,protocol,result=[read(root/n) for n in ("before.json","protocol.json","result.json")]
assert result["status"] == "complete" and result["verdict"] == "REJECTED"
assert not result["earns_native_language_repeat"] and not result["research_goal_achieved"]
assert result["synthetic_updates"] == 120 and result["synthetic_training_targets"] == 245760
assert result["initial_probe_targets"] == 12288 and not result["corpus_scoring"]
assert all(all(c.values()) for c in result["comparisons"].values())
assert {n for n,v in result["gates"].items() if not v} == {
    "at_least_ten_percent_native_memory_reduction", "memory_within_ten_percent_best_full_swiglu", "memory_within_ten_percent_best_full_gelu"
}
assert sha("research/staged_resource_plan.md") == before["plan_sha256"] == protocol["plan_sha256"] == result["plan_sha256"]
assert all(sha(n)==h for n,h in protocol["provenance"]["source_files"].items())
assert all(sha(n)==h for n,h in before["h065_source_hashes"].items())
assert sha("results/verification/rational_staged_final_v1.json") == before["h065_final_sha256"]
with zipfile.ZipFile(root/"source.zip") as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in protocol["provenance"]["source_files"].items())
processes={}
for p in (root/"processes").glob("*.json"):
    row=read(p)
    assert row["status"] == "PASS" and row["returncode"] == 0 and row["source_unchanged"]
    assert row["protocol_sha256"] == sha(root/"protocol.json")
    assert sha(p.with_suffix(".log")) == row["log_sha256"]
    processes[p.stem]={"sha256":sha(p),"elapsed_seconds":row["elapsed_seconds"],"status":row["status"]}
assert len(processes)==7
assert read(root/"preflight.json")["protocol_sha256"] == sha(root/"protocol.json")
artifacts={}
for cell,row in result["results"].items():
    folder=root/"workers"/cell
    assert read(folder/"result.json")==row
    assert row["status"]=="PASS" and row["updates"]==20 and row["all_finite"] and row["rng_unchanged"]
    assert row["scope"]=="block" and row["token_record"]==protocol["token_record"]
    assert row["synthetic_target_exposures"]==40960 and not row["corpus_scoring"]
    assert sha(folder/"checkpoint.pt")==row["checkpoint_sha256"]
    assert sha(folder/"history.jsonl")==row["history_sha256"]
    assert sha(folder/"initial_signature.json")==row["initial_signature_sha256"]
    history=[json.loads(line) for line in (folder/"history.jsonl").read_text().splitlines()]
    assert [item["step"] for item in history]==list(range(201,221)) and finite_tree(history)
    config=read(folder/"config.json")
    assert config["protocol_sha256"]==sha(root/"protocol.json") and config["mode"]==row["mode"]
    assert config["model"]["width"]==384 and config["model"]["layers"]==8
    adapter=read(folder/"adapter.json")
    assert adapter["mode"]==row["mode"] and all(adapter[n] for n in ("parameter_objects_preserved","optimizer_references_preserved","weights_and_moments_preserved"))
    assert all(v["finite"] for n in ("initial_diagnostics.json","final_diagnostics.json") for v in read(folder/n).values())
    checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
    assert checkpoint["step"]==220 and checkpoint["synthetic"] and finite_tree(checkpoint)
    assert digest(checkpoint["model"])==row["final"]["weights"]
    assert digest(checkpoint["optimizer"])==row["final"]["optimizer"]
    del checkpoint
    artifacts[cell]={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
assert len(artifacts)==6
for recipe,treatment in (("full_swiglu","inner"),("full_gelu","inner"),("rational","staged")):
    a,b=[result["results"][f"{recipe}_{mode}"] for mode in ("native",treatment)]
    assert a["initial_signature_sha256"]==b["initial_signature_sha256"]
    assert a["history_sha256"]==b["history_sha256"] and a["final"]==b["final"]
    assert all(a[k]==b[k] for k in ("token_record","cpu_rng","cuda_rng"))
a,b=[result["results"][f"rational_{mode}"] for mode in ("native","staged")]
assert a["peak_allocated_bytes"]-b["peak_allocated_bytes"] == 32*2**20
assert b["ffn_parameters"]==2801984 and b["total_parameters"]==9099968
for recipe in ("full_swiglu","full_gelu"):
    limit=1.1*min(result["results"][f"{recipe}_{mode}"]["peak_allocated_bytes"] for mode in ("native","inner"))
    assert limit==result["dense_memory_limits_bytes"][recipe] and b["peak_allocated_bytes"] > limit
for row in protocol["inputs"].values():
    assert all(sha(Path("results/runs")/row["run"]/n)==h for n,h in row["files"].items())
prior=read("results/verification/rational_staged_final_v1.json")
for cell,files in prior["case_artifact_hashes"].items():
    assert all(sha(Path("results/rational_staged_v1/cases")/cell/n)==h for n,h in files.items())
for file,key in (("result.json","result_sha256"),("source.zip","source_archive_sha256"),("support.zip","support_archive_sha256"),("process.json","process_sha256")):
    assert sha(Path("results/rational_staged_v1")/file)==prior[key]
manifest=read("research/archive/h061_manifest.json")
for name,expected in manifest["protected_evidence_metadata"].items():
    stat=Path(name).stat()
    assert {"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}==expected
plans={p.as_posix():sha(p) for p in Path("research").glob("*plan.md")}
assert len(plans)==53 and all(plans[n]==h for n,h in prior["frozen_plans"].items())
assert len(VARIANTS)==6 and len(list(Path("configs").glob("*.json")))==9
assert len(list(Path("results/runs").glob("*/metrics.json")))==171
assert {p.name for p in Path("src").iterdir() if p.is_dir() and p.name.endswith("_ffn")}=={"dense_ffn","blockshuffle_ffn","rational_blockshuffle_ffn"}
old_manifest=read("research/archive/h057_manifest.json")
links,historical=0,[]
for p in [Path("README.md"),*Path("src").rglob("*.md"),*Path("doc").glob("*.md"),*Path("configs").glob("*.md"),*Path("research").rglob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf-8")):
        clean=unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://",clean):continue
        resolved=(p.parent/clean).resolve()
        if resolved==out.resolve():continue
        if resolved.exists():links+=1;continue
        assert p.parent==Path("research") and p.name.endswith("plan.md"),(p,clean)
        intended=resolved.relative_to(Path.cwd()).as_posix()
        assert intended in old_manifest["files"] and (Path("research/archive/retired")/intended).exists()
        historical.append({"plan":p.as_posix(),"target":clean})
assert len(historical)==3
checks={}
for name,args in {"lint":["-m","ruff","check","src","tests","results/rational_staged_v1/source",str(root/"source")],"counts":["-m","src.core.cli","counts"]}.items():
    process=subprocess.run([sys.executable,*args],capture_output=True,text=True)
    assert process.returncode==0,(name,process.stdout,process.stderr)
    checks[name]=process.stdout
assert {json.loads(line)["variant"] for line in checks["counts"].splitlines()}==set(VARIANTS)
support=[*Path("results/verification").glob("staged_resource*.py"),Path("README.md"),Path("research/staged_resource_plan.md"),Path("research/staged_resource_results.md"),Path("research/rational_staged_results.md"),Path("research/CURRENT_STATE.md"),Path("research/idea_bank.md"),Path("src/rational_blockshuffle_ffn/README.md"),Path("research/figures/staged_resource.png"),Path("research/figures/staged_resource.svg")]
with zipfile.ZipFile(root/"support.zip","x",zipfile.ZIP_DEFLATED) as archive:
    for p in support:archive.write(p,p.as_posix())
record={
    "status":"PASS","goal_turn":"PROGRESS","research_goal_achieved":False,"memory_repair":"REJECTED",
    "all_measured_numerical_comparisons_exact":True,"earns_native_language_repeat":False,
    "workers_completed":6,"worker_retries":0,"synthetic_updates":120,"synthetic_training_targets":245760,"initial_probe_targets":12288,
    "corpus_scoring":False,"retained_lm_profile_runs":171,"active_model_folders":3,"active_variants":6,
    "active_source_configuration_tests_unchanged_from_h064":True,"prior_active_suite_tests":108,
    "h065_case_artifacts_preserved":12,"protected_evidence_files_metadata_unchanged":len(manifest["protected_evidence_metadata"]),
    "saved_allocated_MiB":32,"saved_percent":100*(1-b["peak_allocated_bytes"]/a["peak_allocated_bytes"]),
    "extra_update_time_percent":100*(b["median_step_ms"]/a["median_step_ms"]-1),
    "frozen_plans":plans,"source_hashes":protocol["provenance"]["source_files"],"worker_artifact_hashes":artifacts,
    "processes":processes,"local_links_verified":links,"historical_links_resolve_in_archive":historical,"checks":checks,
    "result_sha256":sha(root/"result.json"),"protocol_sha256":sha(root/"protocol.json"),
    "source_archive_sha256":sha(root/"source.zip"),"support_archive_sha256":sha(root/"support.zip"),
    "documents":{p.as_posix():sha(p) for p in support if p.suffix==".md"},"figure_visually_checked":True,
}
with out.open("x",encoding="utf-8") as handle:handle.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("source_hashes","worker_artifact_hashes","processes","frozen_plans","checks","documents","historical_links_resolve_in_archive")}))
