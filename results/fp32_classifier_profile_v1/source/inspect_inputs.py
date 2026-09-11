"""Read existing metadata before freezing H114; no model execution or updates."""

import json
from pathlib import Path

root = Path("results/whole_job_memory_workspace_v1")
result = json.loads((root / "result.json").read_text())
print("H112 keys:", list(result))
print(
    "H112 case metadata:",
    {
        k: v
        for k, v in result["cases"][0].items()
        if k
        in ("label", "dataset", "batch", "context", "seed", "policy", "checkpoints", "model_config")
    },
)
for variant in ("gelu", "swiglu"):
    folder = Path(f"results/ungated_duration_v1/runs/full_{variant}_seed17")
    config = json.loads((folder / "config.json").read_text())
    print(variant, {k: config[k] for k in ("model", "training")})
    print("qualification:", (folder / "qualification.json").read_text())
prior = json.loads(Path("results/verification/classifier_precision_final_v1.json").read_text())
print(
    "Prior receipt:",
    {k: v for k, v in prior.items() if k not in ("files", "packed", "tensor_hashes")},
)
