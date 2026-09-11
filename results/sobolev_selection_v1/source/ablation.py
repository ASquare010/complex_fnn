"""Attribute the observed gain using the predeclared 2x2 selector/readout grid."""

import json
import statistics as st
from pathlib import Path

root = Path("results/sobolev_selection_v1")
summary = json.loads((root / "summary.json").read_text())
result = json.loads((root / "result.json").read_text())
groups = {(r["teacher"], r["budget"], r["method"]): r for r in summary["groups"]}
rows = []
for teacher in ("gelu", "swiglu"):
    for budget in ("primary", "half", "three_quarter"):
        baseline = groups[teacher, budget, "value_greedy"]
        readout = groups[teacher, budget, "value_select_sobolev_fit"]
        selection = groups[teacher, budget, "sobolev_select_value_fit"]
        both = groups[teacher, budget, "sobolev_greedy"]
        metrics = {}
        for key in ("value_nmse", "derivative_relative_mse"):
            base = baseline[key]["mean"]
            metrics[key] = {
                "baseline": base,
                "readout_only": readout[key]["mean"],
                "selection_only": selection[key]["mean"],
                "both": both[key]["mean"],
                "readout_fraction_of_combined_absolute_gain": (base - readout[key]["mean"])
                / (base - both[key]["mean"]),
                "selection_fraction_of_combined_absolute_gain": (base - selection[key]["mean"])
                / (base - both[key]["mean"]),
                "interaction": both[key]["mean"]
                - readout[key]["mean"]
                - selection[key]["mean"]
                + base,
            }
        rows.append({"teacher": teacher, "budget": budget, "metrics": metrics})
output = {
    "posthoc_attribution_of_predeclared_ablations": True,
    "note": "Fractions use arithmetic mean errors, are descriptive and need not add to one because of interactions. No threshold or allocation is changed.",
    "rows": rows,
    "full_reference": {
        t: {
            k: {
                "mean": st.mean(
                    c["teacher_resources"][k] for c in result["cases"] if c["teacher"] == t
                ),
                "min": min(c["teacher_resources"][k] for c in result["cases"] if c["teacher"] == t),
                "max": max(c["teacher_resources"][k] for c in result["cases"] if c["teacher"] == t),
            }
            for k in ("peak_allocated_bytes", "inference_ms")
        }
        for t in ("gelu", "swiglu")
    },
}
(root / "ablation.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
