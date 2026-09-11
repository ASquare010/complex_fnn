"""Run unchanged plotting after its required numeric exports are complete."""

import runpy
from pathlib import Path

root = Path("results/fp32_classifier_profile_v1")
assert (root / "summary.json").exists() and (root / "metrics.csv.gz").exists()
runpy.run_module("results.fp32_classifier_profile_v1.source.plot", run_name="__main__")
