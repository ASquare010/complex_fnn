"""Final source, artifact, baseline and documentation consistency audit."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.core.activation_training_repeat import source_check, verify
from src.core.benchmark import forward_flops
from src.core.config import ModelConfig
from src.core.report import report
from src.core.reproducibility import provenance, sha256, write_json
from src.core.transformer import Transformer


torch.set_num_threads(4)
old=torch.load("results/verification/activation_before_v1.pt",map_location="cpu",weights_only=True)
for variant,row in old.items():
    mc=ModelConfig(**row["config"])
    model=Transformer(mc,17)
    x=torch.arange(16).reshape(2,8)%32
    loss=model.loss(x,(x+1)%32)
    loss.backward()
    assert forward_flops(mc)==row["flops"]
    assert torch.equal(model(x),row["logits"]) and torch.equal(loss,row["loss"])
    assert all(torch.equal(t,row["state"][n]) for n,t in model.state_dict().items())
    assert all(torch.equal(p.grad,row["gradients"][n]) for n,p in model.named_parameters())
plans={
    "results/activation_screen_v1/protocol.json":"research/learnable_activation_plan.md",
    "results/activation_shapes_v1/protocol.json":"research/learnable_activation_plan.md",
    "results/activation_execution_v1/protocol.json":"research/activation_execution_plan.md",
    "results/activation_training_repeat_v1/protocol.json":"research/activation_training_repeat_plan.md",
}
for protocol,plan in plans.items():
    assert json.loads(Path(protocol).read_text())["plan_sha256"]==sha256(Path(plan))
files=[Path("README.md"),Path("research/CURRENT_STATE.md"),Path("research/idea_bank.md"),
       *Path("src/learnable_activation_ffn").glob("*.md"),
       *Path("research").glob("learnable_activation*.md"),
       *Path("research").glob("activation_*.md"),Path("research/wikitext_screen_results.md")]
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
assert len(rows)==106
result={"status":"complete","original_variants_exact":len(old),"plans_unchanged":plans,
        "local_markdown_links_checked":len(links),"retained_lm_profile_runs":len(rows),
        "training_loop_source_check":source_check(),"compiled_training_repeat":verify(),
        "full_test_record":"results/verification/activation_tests_v2.json",
        "full_tests_returncode":json.loads(Path("results/verification/activation_tests_v2.json").read_text())["returncode"],
        "verification_script_sha256":sha256(Path(__file__)),"provenance":provenance()}
assert result["full_tests_returncode"]==0
write_json(Path("results/verification/activation_final_v1.json"),result)
print(json.dumps({k:v for k,v in result.items() if k not in ("provenance","compiled_training_repeat")},indent=2))
