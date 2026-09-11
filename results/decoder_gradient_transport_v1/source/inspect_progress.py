"""Bounded read-only progress and trace summary."""

import json
from pathlib import Path

root = Path("results/decoder_gradient_transport_v1")
files = sorted((root / "conditions").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
print("Completed:", len(files))
for path in files[:5] + files[-2:]:
    r = json.loads(path.read_text())
    print(
        r["label"],
        "backend_nodes",
        r["attention_backward_nodes"],
        "MiB",
        r["peak_diagnostic_allocated_bytes"] / 2**20,
        "incoming_error",
        r["classifier_difference"]["hidden_gradient"]["relative_l2"],
        "trace",
        {k: round(v["relative_l2"], 9) for k, v in r["paired_repetitions"][0]["trace"].items()},
    )
