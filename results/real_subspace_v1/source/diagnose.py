"""Post-screen output-rank error floor; reporting data is diagnostic, never fitted into a model."""

import json
import math
from pathlib import Path

import torch

ROOT=Path("results/real_subspace_v1")


def run():
    result=json.loads((ROOT/"result.json").read_text())
    audit=json.loads((ROOT/"audit.json").read_text())
    assert audit["passed"] and audit["score_count"]==378
    rows=[]
    for metadata in result["diagnostics"]:
        activation,layer,seed=(metadata[k] for k in ("teacher","layer","seed"))
        folder=ROOT/"pairs"/f"{activation}_l{layer}_s{seed}"
        data=torch.load(folder/"data.pt",weights_only=True)
        state=torch.load(folder/"statistics.pt",weights_only=True)
        y=data["y"][32768:].double()
        centered=y-y.mean(0)
        spectrum=torch.linalg.eigvalsh(centered.T@centered/len(y)).clamp_min(0)
        for rank in (32,64,128):
            # Best affine rank-r output fit on this finite BF16 reporting matrix:
            # centered singular-value tail, via Eckart--Young. Triangle inequality
            # transfers a conservative bound to the freshly evaluated FP32 teacher.
            tail=spectrum[:-rank].sum().item()/y.shape[1]
            normalized_tail=tail/metadata["variance"]
            bound=max(0,math.sqrt(normalized_tail)-math.sqrt(metadata["bf16_target_normalized_drift"]))**2
            c=state["output_basis"][:,-rank:]
            output_only=(y-state["my"])-(y-state["my"])@c@c.T
            rows.append({"teacher":activation,"layer":layer,"seed":seed,"rank":rank,
                "oracle_bf16_tail_over_fp32_variance":normalized_tail,
                "fp32_normalized_error_lower_bound":bound,
                "fixed_output_pca_bf16_normalized_error":output_only.square().mean().item()/metadata["variance"],
                "bf16_fp32_normalized_drift":metadata["bf16_target_normalized_drift"]})
    output={"status":"COMPLETE","post_screen_diagnostic":True,"neural_updates":0,
        "scope":"Finite captured reporting matrices and any prediction matrix in an affine output subspace of rank r; no LM or optimization theorem",
        "equation":"max(0, sqrt(sum(tail_singular_values_squared)/(N*d*V)) - sqrt(BF16_FP32_MSE/V))^2",
        "rows":rows}
    (ROOT/"rank_diagnosis.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps([r for r in rows if r["rank"]==64],indent=2))


if __name__=="__main__":
    torch.set_num_threads(4)
    run()
