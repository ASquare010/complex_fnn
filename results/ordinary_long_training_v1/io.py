"""Compact, atomic JSON artifacts; no numerical serialization changes."""

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
