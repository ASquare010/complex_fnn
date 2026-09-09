"""Export shape trajectories from already audited records without new computation on GPU."""

from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json

root = Path("results/neuron_geometry_recovery_v1")
audit = Path("results/neuron_geometry_audit_v1/result.json")
assert read(audit)["status"] == "PASS"
result = read(root / "result.json")
rows = []
for selected in result["selected"]:
    path = root / "cells" / selected["label"] / "training.json"
    assert sha(path) == result["raw_cell_record_hashes"][path.as_posix()]
    training = read(path)
    if "shape" not in training["diagnostics"][0]:
        continue
    rows.append(
        {
            "label": selected["label"],
            "reporting_mse": selected["reporting_mse"],
            "diagnostics": [
                {key: row[key] for key in ("step", "shape", "control_saturation_fraction")}
                for row in training["diagnostics"]
            ],
            "maximum_preclip_gradient_norm": max(
                row["preclip_norm"] for row in training["history"]
            ),
            "raw_training_sha256": sha(path),
        }
    )
assert len(rows) == 72
write_json(
    Path("results/neuron_geometry_audit_v1/summary.json"),
    {
        "description": "All selected learned/fixed shape trajectories; derived from audited histories.",
        "audit_sha256": sha(audit),
        "training_result_sha256": sha(root / "result.json"),
        "export_source_sha256": sha(__file__),
        "new_optimizer_updates": 0,
        "rows": rows,
    },
)
