"""One CPU-only recovery of a Matplotlib-import access violation; no GPU work."""

import runpy
from pathlib import Path

ROOT = Path("results/paired_workspace_timing_v1")
assert (ROOT / "plot_exit.txt").read_text().strip() == "3221225477"
assert not Path("research/figures/paired_workspace_timing.png").exists()
runpy.run_module("results.paired_workspace_timing_v1.plot", run_name="__main__")
