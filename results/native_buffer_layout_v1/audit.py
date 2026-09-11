"""Reuse the independent H127 native-gradient replay, with the revised root."""

from pathlib import Path
from unittest.mock import patch

from results.native_buffer_loss_v1 import audit as previous

if __name__ == "__main__":
    with patch.object(previous, "ROOT", Path("results/native_buffer_layout_v1")):
        previous.run()
