"""Read-only H160 evidence verification and timing diagnosis; no training."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/checkpoint_fp16_long_v1")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


receipt = json.loads((ROOT / "receipt.json").read_text())
for path, expected in receipt["files"].items():
    assert digest(path) == expected, path
summary = json.loads((ROOT / "summary.json").read_text())
case = json.loads((ROOT / "case01/case.json").read_text())
records = case["measurement"]["records"]
longest = max(records, key=lambda row: row["wall_update_ms"])
output = {
    "receipt_sha256": digest(ROOT / "receipt.json"),
    "verified_files": len(receipt["files"]),
    "audit_passed": summary["audit_passed"],
    "qualification_passed": summary["passed"],
    "timing_outlier": {
        key: longest[key]
        for key in ("step", "wall_update_ms", "event_update_ms", "at_ns", "until_ns")
    },
    "outlier_hours": longest["wall_update_ms"] / 3600000,
    "cause": "Unestablished; recorded elapsed time alone does not identify sleep, driver, or hardware cause.",
    "comparisons": summary["comparisons"],
    "optimizer_updates": 0,
    "gpu_work": False,
}
Path("research/checkpoint_fp16_long_postmortem.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            key: output[key]
            for key in ("verified_files", "audit_passed", "qualification_passed", "outlier_hours")
        }
    )
)
