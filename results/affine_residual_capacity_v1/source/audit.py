"""Independent QR projections and direct residual SVD for H106."""

import json
import math
from pathlib import Path

import torch

from src.core.reproducibility import sha256, write_json

ROOT=Path("results/affine_residual_capacity_v1")
PRIOR=Path("results/real_subspace_v1")


def read(path):
    return json.loads(Path(path).read_text())


def run():
    result,protocol=read(ROOT/"result.json"),read(ROOT/"protocol.json")
    for path,digest in protocol["sources"].items():
        assert sha256(Path(path))==digest
    assert len(result["rows"])==126
    checks=[]
    for metadata in read(PRIOR/"result.json")["diagnostics"]:
        a,l,s=(metadata[k] for k in ("teacher","layer","seed"))
        label=f"{a}_l{l}_s{s}"
        data=torch.load(PRIOR/"pairs"/label/"data.pt",weights_only=True)
        saved=torch.load(ROOT/"cases"/label/"statistics.pt",weights_only=True)
        x=torch.column_stack((data["x"].double(),torch.ones(len(data["x"]),dtype=torch.float64)))
        y=data["y"].double()
        q,r=torch.linalg.qr(x[:32768],mode="reduced")
        assert r.diag().abs().min()>1e-8
        w=torch.linalg.solve_triangular(r,q.T@y[:32768],upper=True)
        torch.testing.assert_close(x[:32768]@w,x[:32768]@saved["affine"],atol=1e-8,rtol=1e-8)
        residual_train=y[:32768]-q@(q.T@y[:32768])
        covariance=residual_train.T@residual_train/32768
        b=saved["output_basis"]
        offdiag=b.T@covariance@b
        torch.testing.assert_close(offdiag,torch.diag(offdiag.diag()),atol=1e-9,rtol=1e-9)
        q,r=torch.linalg.qr(x[32768:],mode="reduced")
        assert r.diag().abs().min()>1e-8
        residual=y[32768:]-q@(q.T@y[32768:])
        values=torch.linalg.svdvals(residual)
        report_residual=y[32768:]-x[32768:]@w
        for row in [v for v in result["rows"] if (v["teacher"],v["layer"],v["seed"])==(a,l,s)]:
            rank=row["rank"]
            tail=values[rank:].square().sum().item()/residual.numel()/metadata["variance"]
            bound=max(0,math.sqrt(tail)-math.sqrt(metadata["bf16_target_normalized_drift"]))**2
            assert abs(tail-row["oracle_bf16_normalized_tail"])<1e-10
            assert abs(bound-row["fp32_lower_bound"])<1e-10
            c=b[:,-rank:]
            error=(report_residual-report_residual@c@c.T).square().mean().item()/metadata["variance"]
            assert abs(error-row["train_basis_oracle_bf16_error"])<1e-10
            checks.append({"teacher":a,"layer":l,"seed":s,"rank":rank,"bound_error":abs(bound-row["fp32_lower_bound"]),"train_basis_error":abs(error-row["train_basis_oracle_bf16_error"])})
        print(f"Audited {label}",flush=True)
    write_json(ROOT/"audit.json",{"passed":True,"checks":checks,"count":126,"independent_method":"QR projection and rectangular residual SVD"})


if __name__=="__main__":
    torch.set_num_threads(4)
    run()
