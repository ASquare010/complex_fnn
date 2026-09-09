"""H067 final audit: qualified local evidence, scoped proof and preserved shortlist."""
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
from src.core.native_recompute_audit import read, sha

root=Path("results/token_activation_v1")
out=Path("results/verification/token_activation_final_v1.json")
assert not out.exists()
before,protocol,result,process=[read(root/n) for n in ("before.json","protocol.json","result.json","process.json")]
assert result["status"]=="complete" and result["scientific_verdict"]=="LOCALLY_QUALIFIED"
assert result["all_checks_pass"] and result["earns_fitting_resource_comparison"]
assert not result["research_goal_achieved"]
assert result["optimizer_updates"]==result["corpus_targets"]==result["full_model_resource_workers"]==0
assert sha("research/token_activation_plan.md")==before["plan_sha256"]==protocol["plan_sha256"]
assert sha("research/token_activation_theory.md")==protocol["theory_sha256"]==result["theory_sha256"]
assert all(sha(n)==h for n,h in protocol["sources"].items())
assert all(sha(n)==h for n,h in before["h066_source_hashes"].items())
assert process["status"]=="PASS" and process["returncode"]==0 and process["source_unchanged"]
assert sha(root/"process.log")==process["log_sha256"]
assert "21 passed in 4.86s" in (root/"process.log").read_text(encoding="utf-8")
assert sha(root/"protocol.json")==process["protocol_sha256"]
assert sha(root/"source.zip")==process["source_archive_sha256"]
with zipfile.ZipFile(root/"source.zip") as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in protocol["sources"].items())
with zipfile.ZipFile(root/"before_docs.zip") as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in before["stale_document_hashes"].items())
checks=result["checks"]
assert len(checks)==21
zero=[v for k,v in checks.items() if k.startswith("zero_")]
nonzero=[v for k,v in checks.items() if k.startswith("nonzero_")]
assert len(zero)==12 and len(nonzero)==4
comparisons=0
for row in zero:
    assert row["all_shared_exact"] and row["new_gradients_finite_nonzero"] and row["optimizer_updates"]==0
    assert all(v["finite"] and v["norm"]>0 for v in row["added_gradients"].values())
for row in [*zero,*nonzero]:
    entries=row["shared"] if "shared" in row else row["comparisons"]
    for c in entries.values():
        assert c["exact"] and c["finite"] and c["mismatched_elements"]==0
        assert c["max_abs_error"]==c["relative_l2_error"]==0 and c["reference"]==c["candidate"]
        comparisons+=1
assert comparisons==134
for name in ("counts","gradcheck","bounds","witness","residues"):
    assert checks[name]["status"]=="PASS"
assert checks["counts"]["counts"]["dynamic"]["ffn_total"]==2804744
assert checks["counts"]["counts"]["dynamic"]["total_model"]==9102728
assert checks["counts"]["counts"]["static"]["ffn_total"]==2801672
assert not checks["bounds"]["whole_network_lower_bound_claimed"]
assert checks["witness"]["construction_only_not_learning"] and checks["residues"]["numerical_check_is_not_proof"]
assert sha(root/"checks/witness_tensors.pt")==checks["witness"]["artifact_sha256"]
stored=torch.load(root/"checks/witness_tensors.pt",map_location="cpu",weights_only=True)
torch.testing.assert_close(stored["outputs"][:,0],stored["expected"],rtol=2e-14,atol=2e-14)
torch.testing.assert_close(stored["router_coordinate_gradient"],stored["expected_gradient"],rtol=2e-13,atol=2e-14)
assert torch.count_nonzero(stored["outputs"][:,1:])==0
for n,row in checks.items():assert read(root/"checks"/f"{n}.json")==row
assert sha("results/verification/staged_resource_final_v1.json")==before["h066_final_sha256"]
prior=read("results/verification/staged_resource_final_v1.json")
for cell,files in prior["worker_artifact_hashes"].items():
    assert all(sha(Path("results/staged_resource_v1/workers")/cell/n)==h for n,h in files.items())
for file,key in (("result.json","result_sha256"),("protocol.json","protocol_sha256"),("source.zip","source_archive_sha256"),("support.zip","support_archive_sha256")):
    assert sha(Path("results/staged_resource_v1")/file)==prior[key]
manifest=read("research/archive/h061_manifest.json")
for name,expected in manifest["protected_evidence_metadata"].items():
    stat=Path(name).stat()
    assert {"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}==expected
plans={p.as_posix():sha(p) for p in Path("research").glob("*plan.md")}
assert len(plans)==54 and all(plans[n]==h for n,h in prior["frozen_plans"].items())
assert len(VARIANTS)==6 and len(list(Path("configs").glob("*.json")))==9
assert len(list(Path("results/runs").glob("*/metrics.json")))==171
assert {p.name for p in Path("src").iterdir() if p.is_dir() and p.name.endswith("_ffn")}=={"dense_ffn","blockshuffle_ffn","rational_blockshuffle_ffn"}
links,historical=0,[]
old_manifest=read("research/archive/h057_manifest.json")
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
lint=subprocess.run([sys.executable,"-m","ruff","check","src","tests",str(root/"source")],capture_output=True,text=True)
assert lint.returncode==0,lint.stdout
assert "Its empirical comparison is still outstanding." not in Path("research/literature.md").read_text(encoding="utf-8")
assert "Two registered variants now" not in Path("research/current_comparator_audit.md").read_text(encoding="utf-8")
support=[*Path("results/verification").glob("token_activation*.py"),Path("README.md"),Path("research/token_activation_plan.md"),Path("research/token_activation_theory.md"),Path("research/token_activation_results.md"),Path("research/CURRENT_STATE.md"),Path("research/idea_bank.md"),Path("research/literature.md"),Path("research/current_comparator_audit.md"),Path("research/learnable_activation_domain.md"),Path("research/figures/token_activation.png"),Path("research/figures/token_activation.svg")]
with zipfile.ZipFile(root/"support.zip","x",zipfile.ZIP_DEFLATED) as archive:
    for p in support:archive.write(p,p.as_posix())
record={
    "status":"PASS","goal_turn":"PROGRESS","scientific_verdict":"LOCALLY_QUALIFIED","research_goal_achieved":False,
    "qualification_tests_passed":21,"exact_tensor_comparisons":134,"zero_router_cases_with_learning_signal":12,
    "optimizer_updates":0,"corpus_targets":0,"full_model_resource_workers":0,"earns_fitting_resource_comparison":True,
    "process_retries":0,"active_model_folders":3,"active_variants":6,"retained_lm_profile_runs":171,
    "all_h066_source_configuration_tests_unchanged":True,"prior_active_suite_tests":108,
    "protected_evidence_files_metadata_unchanged":len(manifest["protected_evidence_metadata"]),
    "theory_scope":"Exact real-arithmetic single-FFN nonrepresentation, manually reviewed; no approximation, optimization, full-Transformer or novelty claim. Numerical checks do not prove the theorem.",
    "counts":checks["counts"]["counts"],"frozen_plans":plans,"source_hashes":protocol["sources"],
    "check_artifact_hashes":{p.name:sha(p) for p in (root/"checks").iterdir() if p.is_file()},
    "local_links_verified":links,"historical_links_resolve_in_archive":historical,"lint":lint.stdout,
    "stale_comparator_documentation_corrected":True,"result_sha256":sha(root/"result.json"),
    "protocol_sha256":sha(root/"protocol.json"),"source_archive_sha256":sha(root/"source.zip"),
    "support_archive_sha256":sha(root/"support.zip"),"process_sha256":sha(root/"process.json"),
    "documents":{p.as_posix():sha(p) for p in support if p.suffix==".md"},"figure_visually_checked":True,
}
with out.open("x",encoding="utf-8") as handle:handle.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("source_hashes","frozen_plans","check_artifact_hashes","historical_links_resolve_in_archive","documents")}))
