"""Read final raw resource and arithmetic limits for reporting."""

import json
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
result = json.loads((ROOT / "result.json").read_text())
audit = json.loads((ROOT / "audit.json").read_text())
print(
    "MAXIMA",
    dict(
        paired_loss=max(r["loss_relative_error"] for r in audit["paired_initial_errors"]),
        paired_gradient=max(
            r["error"]["global_relative_l2"] for r in audit["paired_initial_errors"]
        ),
        paired_tensor=max(
            r["error"]["max_tensor_relative_l2"] for r in audit["paired_initial_errors"]
        ),
        replay_tensor=max(
            r["replay"]["error"]["max_tensor_relative_l2"] for r in audit["rows"] if r["replay"]
        ),
        timing_stability=max(c["timing_stability_ratio"] for c in result["cases"]),
    ),
)
for case in result["cases"]:
    phases = [
        p["phase"]
        for p in case["memory_phases"]
        if p["peak_allocated_bytes"] == case["peak_job_allocated_bytes"]
    ]
    print(
        json.dumps(
            dict(
                label=case["label"],
                peak_mib=case["peak_job_allocated_bytes"] / 2**20,
                reserved_mib=case["peak_job_reserved_bytes"] / 2**20,
                phases=phases[:4],
                phase_count=len(phases),
                parameter_bytes=case["parameter_bytes"],
                optimizer_bytes=case["optimizer_bytes"],
                wall_ms=case["timing"]["wall_update_ms"]["median"],
                event_ms=case["timing"]["event_update_ms"]["median"],
                loop_seconds=case["training_loop_wall_seconds"],
            )
        )
    )
