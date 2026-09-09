"""CPU-only constructive SwiGLU witness and independent Hessian checks."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.dense_ffn import DenseFFN

torch.set_num_threads(4)
root=Path("results/headwise_hessian_witness_v1")
root.mkdir(exist_ok=False)

def construct(d):
    model=DenseFFN(d,8*d//3,"swiglu").double()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        directions=torch.cat([torch.eye(d,dtype=torch.float64),torch.ones(1,d,dtype=torch.float64)])
        for j,v in enumerate(directions):
            model.up.weight[2*j].copy_(v)
            model.up.weight[2*j+1].copy_(-v)
            if j<d:
                model.down.weight[0,2*j:2*j+2]=.5
                model.down.weight[1,2*j:2*j+2]=(j+1)/2
            else: model.down.weight[2,2*j:2*j+2]=.5
        model.gate.weight.copy_(model.up.weight)
    return model

def target(x):
    y=torch.zeros_like(x)
    y[...,0]=x.square().sum(-1)/2
    y[...,1]=(x.square()*torch.arange(1,x.shape[-1]+1,dtype=x.dtype)).sum(-1)/2
    y[...,2]=x.sum(-1).square()/2
    return y

records=[]
for d in (3,6,24,384):
    model=construct(d)
    generator=torch.Generator().manual_seed(7051+d)
    x=torch.randn(31,d,dtype=torch.float64,generator=generator)
    actual=model(x);expected=target(x)
    max_error=(actual-expected).abs().max().item()
    relative=(actual-expected).norm().item()/expected.norm().item()
    assert relative<1e-12
    result={"width":d,"hidden":8*d//3,"used_hidden":2*(d+1),"forward_max_absolute_error":max_error,"forward_relative_l2":relative}
    if d in (3,6):
        hs=[torch.eye(d,dtype=torch.float64),torch.diag(torch.arange(1,d+1,dtype=torch.float64)),torch.ones(d,d,dtype=torch.float64)]
        errors=[]
        for point in (torch.zeros(d,dtype=torch.float64),x[0]):
            for i,h in enumerate(hs):
                observed=torch.autograd.functional.hessian(lambda v:model(v)[i],point)
                errors.append((observed-h).abs().max().item())
                assert torch.allclose(observed,h,atol=1e-11,rtol=1e-11)
        result["hessian_max_absolute_error"]=max(errors)
        basis=torch.eye(d*d,dtype=torch.float64).reshape(d*d,d,d)
        operator=torch.cat([(basis@h-h@basis).reshape(d*d,d*d).T for h in hs[1:]])
        _,s,vh=torch.linalg.svd(operator,full_matrices=False)
        rank=(s>1e-10).sum().item()
        alignment=abs(torch.dot(vh[-1],torch.eye(d,dtype=torch.float64).flatten()/d**.5).item())
        assert rank==d*d-1 and abs(alignment-1)<1e-12
        result.update(commutant_dimension=d*d-rank,identity_nullspace_alignment=alignment,smallest_positive_commutator_singular_value=s[-2].item(),numerical_rank_tolerance=1e-10)
    records.append(result)
result={"status":"PASS","records":records,"compute_device":"cpu","optimizer_updates":0,"validation_targets_scored":0,"scope":"Numerical sanity checks of explicit construction; general impossibility follows from written proof, not numerical rank. No novelty, value-approximation, whole-Transformer or training-performance claim.","script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"dense_source_sha256":hashlib.sha256(Path("src/dense_ffn/__init__.py").read_bytes()).hexdigest()}
(root/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
