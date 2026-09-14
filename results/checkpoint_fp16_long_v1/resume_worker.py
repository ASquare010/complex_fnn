"""Run only the interrupted case in a fresh output root, using unchanged worker."""

# ruff: noqa: I001
from results.checkpoint_fp16_long_v1 import worker as current
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path
import sys

ROOT = Path("results/checkpoint_fp16_long_v1")
assert sys.argv[1:] == ["3"]
r = read(ROOT / "resume_protocol.json")
hashes(r["sources"])
assert read(ROOT / "preflight.json")["passed"]
assert (ROOT / "preflight_exit.txt").read_text().strip() == "0"
current.base.ROOT = ROOT / "restart"
current.base.one = current.one
current.base.run()
