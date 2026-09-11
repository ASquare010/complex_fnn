"""Print compact already-completed H113 statistics without importing CUDA."""

import json
from pathlib import Path

summary = json.loads(Path("results/classifier_precision_v1/summary.json").read_text())
for group in summary["groups"]:
    m = group["metrics"]
    print(
        group["dataset"],
        group["policy"],
        "dW error median",
        m["weight_error"]["median"],
        "dH error median",
        m["hidden_error"]["median"],
        "weight/cached",
        m["weight_error_ratio_to_cached"]["median"],
        "memory MiB",
        m["allocated_bytes"]["median"] / 2**20,
        "memory/native",
        m["memory_ratio_to_native_bf16"]["max"],
        "time ms",
        m["median_total_ms"]["median"],
        "time/native",
        m["time_ratio_to_native_bf16"]["median"],
        "time ratio range",
        m["time_ratio_to_native_bf16"]["min"],
        m["time_ratio_to_native_bf16"]["max"],
    )
