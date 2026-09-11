"""Reuse H131's measured probe loop with the explicit workspace setter."""

from pathlib import Path
from unittest.mock import patch

from results.blas_workspace_mid_v1.common import configure
from results.blas_workspace_v1 import worker as previous

if __name__ == "__main__":
    with (
        patch.object(previous, "ROOT", Path("results/blas_workspace_mid_v1")),
        patch.object(previous, "configure", configure),
    ):
        previous.run()
