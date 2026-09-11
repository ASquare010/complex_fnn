"""Reuse native replay under the exact intermediate-workspace policy."""

from pathlib import Path
from unittest.mock import patch

from results.blas_workspace_mid_v1.common import configure
from results.blas_workspace_v1 import audit as previous

if __name__ == "__main__":
    with (
        patch.object(previous, "ROOT", Path("results/blas_workspace_mid_v1")),
        patch.object(previous, "configure", configure),
    ):
        previous.run()
