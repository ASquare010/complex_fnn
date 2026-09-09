"""Close H039-H044 evidence and preserve all 28 models before further integration."""
import collections
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.core.benchmark import forward_flops
from src.core.config import ModelConfig, VARIANTS
from src.core.report import report
from src.core.reproducibility import provenance, sha256, write_json
from src.core.transformer import Transformer


def main():
    torch.set_num_threads(4)
    tested = json.loads(Path("results/verification/affine_corrected_tests_v2.json").read_text())
    assert tested["returncode"] == 0 and tested["source_unchanged_through_tests"]
    assert all(sha256(Path(n)) == h for n,h in tested["source_files"].items())
    old = torch.load("results/verification/affine_before_v1.pt", map_location="cpu", weights_only=True)
    for variant,row in old.items():
        mc = ModelConfig(**row["config"])
        model = Transformer(mc,17)
        x = torch.arange(16).reshape(2,8)%32
        loss = model.loss(x,(x+1)%32)
        loss.backward()
        assert forward_flops(mc) == row["flops"]
        assert torch.equal(model(x),row["logits"]) and torch.equal(loss,row["loss"])
        assert all(torch.equal(t,row["state"][n]) for n,t in model.state_dict().items())
        assert all(torch.equal(p.grad,row["gradients"][n]) for n,p in model.named_parameters())
    before = {}
    assert len(VARIANTS) == 28
    for variant in VARIANTS:
        mc = ModelConfig(variant=variant,width=24,layers=2,heads=3,context=8,vocab_size=32,hidden=48,groups=3)
        model = Transformer(mc,17)
        x = torch.arange(16).reshape(2,8)
        logits = model(x)
        loss = model.loss(x,x+1)
        loss.backward()
        before[variant] = {"config":asdict(mc),"state":{n:t.detach().clone() for n,t in model.state_dict().items()},"logits":logits.detach().clone(),"loss":loss.detach().clone(),"gradients":{n:p.grad.detach().clone() for n,p in model.named_parameters()},"flops":forward_flops(mc)}
    snapshot = Path("results/verification/multihead_before_v1.pt")
    assert not snapshot.exists()
    torch.save(before,snapshot)
    write_json(snapshot.with_suffix(".json"),{"variants":list(before),"sha256":sha256(snapshot),"provenance":provenance(),"scope":"Exact CPU initial parameters, logits, loss, all parameter gradients and matrix-work counts before comparator integration; tiny width24/layers2/seed17; not trained equivalence."})
    plans = {
        "activation_correction_repeat_v1":"activation_correction_repeat_plan.md",
        "affine_activation_screen_v1":"affine_activation_plan.md",
        "affine_activation_shapes_v1":"affine_activation_plan.md",
        "activation_correction_execution_v1":"activation_correction_execution_plan.md",
        "activation_screen_v1":"learnable_activation_plan.md",
        "activation_shapes_v1":"learnable_activation_plan.md",
        "activation_execution_v1":"activation_execution_plan.md",
        "activation_training_repeat_v1":"activation_training_repeat_plan.md",
        "affine_longer_v2":"affine_activation_longer_plan.md",
        "affine_replication_v1":"affine_activation_replication_plan.md",
        "affine_removal_v1":"affine_activation_removal_plan.md",
        "affine_rate_control_v1":"affine_activation_rate_control_plan.md",
        "affine_rate_replication_v1":"affine_rate_replication_plan.md",
    }
    for root,plan in plans.items():
        protocol = json.loads(Path(f"results/{root}/protocol.json").read_text())
        assert protocol["plan_sha256"] == sha256(Path("research")/plan)
    removal_root = Path("results/affine_removal_v1")
    protocol = json.loads((removal_root/"protocol.json").read_text())
    with zipfile.ZipFile(removal_root/"source.zip") as archive:
        assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in protocol["provenance"]["source_files"].items())
    removal=[]
    for seed in (17,29,43):
        path=removal_root/f"s{seed}"
        r=json.loads((path/"result.json").read_text())
        source=Path("results/runs")/r["source_run"]/"checkpoint.pt"
        assert sha256(source)==r["source_checkpoint_sha256"]
        assert sha256(path/"stripped_checkpoint.pt")==r["stripped_checkpoint_sha256"]
        old_state=torch.load(source,map_location="cpu",weights_only=True)["model"]
        deployment=torch.load(path/"stripped_checkpoint.pt",map_location="cpu",weights_only=True)
        common={n:t for n,t in old_state.items() if ".curve." not in n}
        assert set(common)==set(deployment["model"])
        assert all(torch.equal(t,deployment["model"][n]) for n,t in common.items())
        assert deployment["optimizer_steps_after_source"]==0 and deployment["source_step"]==800
        assert r["passes"]==all(r["gates"].values()) and r["reset_nll"]==r["plain_nll"]
        removal.append({"seed":seed,"common_weights_exact":True,"gates_pass":r["passes"],"cost_percent":r["relative_reset_cost_percent"]})
        del old_state,deployment,common
    files=[Path("README.md"),Path("research/CURRENT_STATE.md"),Path("research/idea_bank.md"),*Path("src/learnable_activation_ffn").glob("*.md"),*Path("src/blockshuffle_ffn").glob("*.md"),*Path("research").glob("learnable_activation*.md"),*Path("research").glob("activation_*.md"),*Path("research").glob("affine_*.md"),Path("research/wikitext_screen_results.md")]
    links=[]
    for path in files:
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if target.startswith(("https:","http:","#")):
                continue
            target=target.split("#")[0]
            if target:
                assert (path.parent/target).exists(),(str(path),target)
                links.append({"source":path.as_posix(),"target":target})
    rows=report(Path("results"))
    assert len(rows)==132
    counts=collections.Counter(json.loads(p.read_text())["training"]["steps"] for p in Path("results/runs").glob("*/metrics.json"))
    assert counts=={200:74,800:55,20:3}
    corrected=json.loads(Path("results/affine_rate_replication_v1/result.json").read_text())
    assert corrected["status"]=="complete" and corrected["plain_passes_all_seeds"] and not corrected["material_activation_benefit"]
    result={"status":"complete","old_variants_exact":len(old),"new_before_snapshot_variants":len(before),"plans_unchanged":plans,"removal_common_weights_verified":removal,"local_markdown_links_checked":len(links),"retained_lm_profile_runs":len(rows),"step_counts":dict(counts),"full_test_record":"results/verification/affine_corrected_tests_v2.json","full_tests_passed":207,"full_test_sources_still_exact":True,"corrected_same_rate_result_sha256":sha256(Path("results/affine_rate_replication_v1/result.json")),"stronger_plain_passes":True,"material_activation_benefit":False,"verification_script_sha256":sha256(Path(__file__)),"provenance":provenance()}
    write_json(Path("results/verification/affine_corrected_final_v1.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="provenance"},indent=2))

if __name__=="__main__":
    main()
