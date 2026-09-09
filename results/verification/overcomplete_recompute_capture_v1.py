"""Capture every default variant before optional overcomplete gate recomputation."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from dataclasses import asdict
from src.core.config import VARIANTS,ModelConfig
from src.core.transformer import Transformer
from src.core.benchmark import forward_flops
from src.core.reproducibility import provenance

torch.set_num_threads(4)
p=Path('results/verification/overcomplete_recompute_before_v1.pt')
assert not p.exists() and len(VARIANTS)==32
old=torch.load('results/verification/overcomplete_before_v1.pt',map_location='cpu',weights_only=True)
rows={};x=torch.arange(16).reshape(2,8)
for variant in VARIANTS:
 c=ModelConfig(**old[variant]['config']) if variant in old else ModelConfig(variant=variant,width=24,layers=2,heads=3,context=8,vocab_size=32,hidden=16,groups=2)
 m=Transformer(c,17);loss=m.loss(x,x+1);loss.backward()
 row={'config':asdict(c),'state':{n:t.detach().clone() for n,t in m.state_dict().items()},'gradients':{n:v.grad.detach().clone() for n,v in m.named_parameters()},'loss':loss.detach().clone(),'logits':m(x).detach().clone(),'flops':forward_flops(c)}
 if variant in old:
  assert all(torch.equal(t,old[variant][key][n]) for key in ('state','gradients') for n,t in row[key].items())
  assert torch.equal(row['loss'],old[variant]['loss']) and torch.equal(row['logits'],old[variant]['logits']) and row['flops']==old[variant]['flops']
 rows[variant]=row
torch.save(rows,p)
r={'status':'PASS','variants':32,'previous_31_exact':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'provenance':provenance(),'plan_sha256':hashlib.sha256(Path('research/overcomplete_recompute_plan.md').read_bytes()).hexdigest()}
p.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in r.items() if k!='provenance'}),flush=True)
