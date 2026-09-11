"""Print a compact verified numeric view for the research report."""

import json
from pathlib import Path

root = Path("results/decoder_gradient_transport_v1")
summary = json.loads((root / "summary.json").read_text())
print(
    "TOTAL",
    {
        k: v
        for k, v in summary.items()
        if k not in ("groups", "decisions", "cross_mode", "independent_audit")
    },
)
for group in summary["groups"]:
    print(
        "MODE",
        group["mode"],
        "exact repeats",
        group["bitwise_repeat_matches"],
        "/",
        group["repeated_comparisons"],
    )
    for key, values in group["metrics"].items():
        print(key, {k: values[k] for k in ("median", "min", "max")})
print("DECISIONS", summary["decisions"])
for mode in summary["cross_mode"][0]["forward_comparisons_to_bf16_default_block"]:
    vals = [
        r["forward_comparisons_to_bf16_default_block"][mode]["relative_l2"]
        for r in summary["cross_mode"]
    ]
    print("FORWARD", mode, min(vals), max(vals))
print("AUDIT", summary["independent_audit"])
