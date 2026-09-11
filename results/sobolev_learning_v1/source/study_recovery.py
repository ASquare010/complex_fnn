"""First scientific execution after the preserved pre-protocol path-type failure."""

import traceback
from pathlib import Path

import torch

from results.sobolev_learning_v1.source.study import run
from src.core.reproducibility import write_json

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    run()
except Exception:
    write_json(
        Path("results/sobolev_learning_v1/study_recovery_failure.json"),
        {"traceback": traceback.format_exc()},
    )
    raise
