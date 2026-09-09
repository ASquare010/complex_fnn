"""One read-only tokenization probe; does not import Torch or retry a worker."""

import ast
import hashlib
import io
import json
import sys
import tokenize
from datetime import datetime, timezone
from pathlib import Path

out = Path("results/verification/ungated_resource_tokenizer_probe_v1.json")
assert not out.exists()
paths = [
    Path(".venv/Lib/site-packages/torch/_functorch/config.py"),
    Path(".venv/Lib/site-packages/torch/utils/_config_module.py"),
    Path(tokenize.__file__),
]
records = {}
for p in paths:
    source = p.read_text(encoding="utf-8")
    record = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    try:
        tree = ast.parse(source, filename=str(p))
        token_list = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
        record.update(status="PASS", ast_top_level_nodes=len(tree.body), tokens=len(token_list))
    except Exception as exc:
        record.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    records[str(p)] = record
value = {
    "status": "PASS" if all(r["status"] == "PASS" for r in records.values()) else "FAIL",
    "utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "files": records,
    "torch_imported": False,
    "optimizer_updates": 0,
    "scientific_worker_retries": 0,
    "interpretation": "A standalone file parse/tokenization probe only; does not reproduce the original import state or establish its failure cause.",
}
out.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
print(json.dumps(value))
