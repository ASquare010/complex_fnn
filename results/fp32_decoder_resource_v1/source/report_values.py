"""Compact report values from completed exports only."""

import json
from pathlib import Path

summary = json.loads(Path("results/fp32_decoder_resource_v1/summary.json").read_text())
print(
    json.dumps(
        {k: v for k, v in summary.items() if k not in ("groups", "decisions", "pareto_by_fixture")},
        indent=2,
    )
)
for g in summary["groups"]:
    if g["policy"].endswith("chunks"):
        print(
            json.dumps(
                dict(
                    scope=g["scope"],
                    policy=g["policy"],
                    metrics={
                        k: g["metrics"][k]
                        for k in (
                            "peak_job_mib",
                            "wall_update_ms",
                            "memory_ratio_bf16",
                            "memory_ratio_native",
                            "wall_ratio_bf16",
                            "wall_ratio_native",
                            "nll_ratio_bf16",
                            "nll_ratio_native",
                        )
                    },
                )
            )
        )
for decision in summary["decisions"]:
    print(
        json.dumps(
            dict(
                scope=decision["scope"],
                policy=decision["policy"],
                failed=[k for k, v in decision["gates"].items() if not v],
            )
        )
    )
