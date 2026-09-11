"""Read compact publication schemas without importing the scientific runtime."""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("results/decoder_gradient_transport_v1")


def compact(value, depth=0):
    if depth > 1:
        if isinstance(value, dict):
            return {"keys": list(value)}
        if isinstance(value, list):
            return {"length": len(value), "first": compact(value[0], depth) if value else None}
        return value
    if isinstance(value, dict):
        return {k: compact(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return {"length": len(value), "first": compact(value[0], depth + 1) if value else None}
    return value


for name in ("result", "audit", "audit_protocol", "qualification", "environment"):
    value = json.loads((ROOT / f"{name}.json").read_text())
    if name == "audit_protocol":
        value = {k: len(v) if isinstance(v, dict) else v for k, v in value.items()}
    print(name, json.dumps(compact(value)))
protocol = json.loads((ROOT / "protocol.json").read_text())
print("protocol", {k: type(v).__name__ for k, v in protocol.items()})
print("maintained", len(protocol["maintained_files"]))
print("directories", [p.name for p in ROOT.iterdir()])
for name in protocol["before_documents"]:
    content = Path(name).read_text(encoding="utf-8")
    print("NAV", name, "\n".join(content.splitlines()[:45]))
    if name == ".gitignore":
        print("TAIL", "\n".join(content.splitlines()[-65:]))
result = json.loads((ROOT / "result.json").read_text())
case = result["conditions"][0]
for name in ("fixture", "records", "paired_repetitions", "memory_phases", "artifacts"):
    value = case[name]
    print("CASE", name, json.dumps(compact(value[0] if isinstance(value, list) else value)))
print("QUAL", (ROOT / "qualification.json").read_text())
