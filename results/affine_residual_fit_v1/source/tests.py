"""Run the maintained suite after the single-GPU experiment and audit finish."""

import os
from pathlib import Path

os.environ["CC"] = str(Path(".venv/Lib/site-packages/triton/runtime/tcc/tcc.exe").resolve())

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:anyio"]))
