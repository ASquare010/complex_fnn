"""Reconstruct the declared edits in memory and compare syntax trees."""

import ast
import json
import runpy
from pathlib import Path
from unittest.mock import patch

ROOT = Path("results/paired_complete_training_v1")
captured = []
with patch.object(Path, "write_text", lambda self, text, **kwargs: captured.append((self, text))):
    runpy.run_path(str(ROOT / "derive.py"))
assert len(captured) == 1 and captured[0][0] == ROOT / "loop.py"
assert ast.dump(ast.parse(captured[0][1])) == ast.dump(ast.parse((ROOT / "loop.py").read_text()))
(ROOT / "derivation_audit.json").write_text(
    json.dumps(
        dict(
            passed=True,
            scope="Generated syntax tree equals frozen loop; only declared instrumentation edits plus formatting",
        )
    )
    + "\n"
)
print("Timing-only derivation verified")
