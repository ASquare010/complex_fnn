"""Capture exact 31-model regression signatures before any integration edit."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from dataclasses import asdict
from src.core.benchmark import forward_flops
from src.core.config import VARIANTS,ModelConfig
from src.core.reproducibility import provenance
from src.core.transformer import Transformer

torch.set_num_threads(4)
output=Path('results/verification/overcomplete_before_v1.pt')
assert not output.exists() and len(VARIANTS)==31
old=torch.load('results/verification/headwise_before_v1.pt',map_location='cpu',weights_only=True)
rows={}
x=torch.arange(16).reshape(2,8)
for variant in VARIANTS:
 config=ModelConfig(**old[variant]['config']) if variant in old else ModelConfig(variant=variant,width=24,layers=2,heads=3,context=8,vocab_size=32,hidden=48,groups=3)
 model=Transformer(config,17)
 loss=model.loss(x,x+1)
 loss.backward()
 row={'config':asdict(config),'state':{n:t.detach().clone() for n,t in model.state_dict().items()},
      'gradients':{n:p.grad.detach().clone() for n,p in model.named_parameters()},
      'loss':loss.detach().clone(),'logits':model(x).detach().clone(),'flops':forward_flops(config)}
 if variant in old:
  for key in ('state','gradients'):
   assert all(torch.equal(t,old[variant][key][n]) for n,t in row[key].items())
  assert torch.equal(row['loss'],old[variant]['loss']) and torch.equal(row['logits'],old[variant]['logits'])
  assert row['flops']==old[variant]['flops']
 rows[variant]=row
assert len(rows)==31
torch.save(rows,output)
record={'status':'PASS','variants':31,'previous_30_exact':True,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'provenance':provenance(),'plan_sha256':hashlib.sha256(Path('research/overcomplete_integration_plan.md').read_bytes()).hexdigest()}
output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='provenance'}),flush=True)
