"""Post-hoc descriptive subsets; frozen H073 promotion gates are unchanged."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path("results/ungated_fit_v1")
OUT = Path("results/verification/ungated_fit_subsets_v1.json")
assert not OUT.exists()
r = json.loads((ROOT / "result.json").read_text(encoding="utf-8"))
rows = {(v["task"], v["form"], v["seed"]): v["heldout_mse"] for v in r["selected_rows"]}
tasks = list(r["task_mean_mse"])
subsets = {
    "all_tasks": tasks,
    "without_linear": [t for t in tasks if t != "linear"],
    "generic_only": ["smooth", "oscillatory", "multiplicative", "piecewise"],
}
references = ("plain", "gelu_same", "narrow_gelu", "narrow_swiglu", "full_gelu", "full_swiglu")
ratios = {}
for name, ts in subsets.items():
    ratios[name] = {}
    for form in ("gelu_same", "gelu_matched"):
        ratios[name][form] = {
            ref: math.exp(
                statistics.mean(
                    math.log(rows[t, form, s] / rows[t, ref, s]) for t in ts for s in (17, 29, 43)
                )
            )
            for ref in references
        }
record = {
    "status": "PASS",
    "post_hoc_descriptive_subsets": True,
    "decision_gates_changed": False,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "subsets": subsets,
    "ratios": ratios,
    "result_sha256": hashlib.sha256((ROOT / "result.json").read_bytes()).hexdigest(),
}
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record))
