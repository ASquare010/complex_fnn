"""Improve spacing only in unfrozen H110 reports; never edit frozen plans/source."""

import re
from pathlib import Path

paths = [
    Path("research/token_memory_duration_results.md"),
    Path("research/runtime_import_diagnosis.md"),
    Path("results/token_memory_duration_v1/source/README.md"),
    Path("results/token_memory_duration_fresh_v1/source/README.md"),
]
words = (
    "all|All|at|At|through|among|from|an|a|width|hidden|vocabulary|epsilon|clip|seeds|Seeds|"
    "seed|updates|steps|previous|PyTorch|Python|UV|RTX|of|with|betas|LR|is|took|has|"
    "and|case|before|after|over|than|the|range|chunk|chunks|passed|was|decay|most"
)
for path in paths:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\b(" + words + r")(?=[+-]?\d)", r"\1 ", text)
    text = re.sub(r"\b(at|At)(?=[BT]\d)", r"\1 ", text)
    text = text.replace("/9,600", "/ 9,600").replace("/6,400", "/ 6,400")
    path.write_text(text, encoding="utf-8")
print("Polished four unfrozen H110 documents; scientific files unchanged")
