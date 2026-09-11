"""Cross-check covariance-based rank floors using direct rectangular SVD tails."""

import json
import math
from pathlib import Path

import torch

ROOT = Path("results/real_subspace_v1")


def run():
    diagnosis = json.loads((ROOT / "rank_diagnosis.json").read_text())
    result = json.loads((ROOT / "result.json").read_text())
    checks = []
    for metadata in result["diagnostics"]:
        activation,layer,seed = (metadata[k] for k in ("teacher","layer","seed"))
        data = torch.load(ROOT / "pairs" / f"{activation}_l{layer}_s{seed}" / "data.pt",weights_only=True)
        y = data["y"][32768:].double()
        y -= y.mean(0)
        singular = torch.linalg.svdvals(y)
        for rank in (32,64,128):
            row = next(r for r in diagnosis["rows"] if (r["teacher"],r["layer"],r["seed"],r["rank"]) == (activation,layer,seed,rank))
            tail = singular[rank:].square().sum().item() / (y.numel()*metadata["variance"])
            assert abs(tail-row["oracle_bf16_tail_over_fp32_variance"]) < 1e-10
            bound = max(0, math.sqrt(tail)-math.sqrt(metadata["bf16_target_normalized_drift"]))**2
            assert abs(bound-row["fp32_normalized_error_lower_bound"]) < 1e-10
            for candidate in result["rows"]:
                if (candidate["teacher"],candidate["layer"],candidate["seed"],candidate["rank"]) == (activation,layer,seed,rank):
                    assert candidate["normalized_mse"] >= bound - 1e-9
            checks.append({"teacher":activation,"layer":layer,"seed":seed,"rank":rank,
                           "direct_svd_lower_bound":bound,"eigen_vs_svd_error":abs(tail-row["oracle_bf16_tail_over_fp32_variance"])})
    value={"passed":True,"checks":checks,"cases":54,"rank64_all_above_five_percent":all(c["direct_svd_lower_bound"]>.05 for c in checks if c["rank"]==64)}
    (ROOT / "rank_audit.json").write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps({"passed":True,"cases":54,"rank64_all_above_five_percent":value["rank64_all_above_five_percent"]}))


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()
