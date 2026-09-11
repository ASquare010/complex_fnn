"""Compact human review of complete H109 results; never runs models."""

import json
from pathlib import Path

s = json.loads(Path("results/sobolev_learning_v1/summary.json").read_text())
for g in s["groups"]:
    print(
        g["teacher"],
        g["method"],
        "value",
        g["reporting_mse"]["mean"],
        "jvp",
        g["derivative_relative_mse"]["mean"],
        "initial",
        {k: v["mean"] for k, v in g["initial"].items()},
        "selected/initial",
        g["selected_over_initial_ratios"],
        "train MiB",
        g["peak_allocated_bytes"]["mean"] / 2**20,
        "inference MiB",
        g["inference"]["peak_allocated_bytes"]["mean"] / 2**20,
        "ms",
        g["median_update_ms"]["mean"],
        g["inference"]["inference_ms"]["mean"],
        "grad",
        g["training"],
    )
for method, decision in s["components"].items():
    for teacher, d in decision["teachers"].items():
        print(method, teacher, d["paired_ratios"], "gates", d["gates"])
print("full", s["full_references"])
print("pipeline MiB", s["pipeline_peak_cuda_bytes"] / 2**20, "seconds", s["elapsed_seconds"])
print("allocation", s["allocation"])
