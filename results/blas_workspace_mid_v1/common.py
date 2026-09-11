"""Keep deterministic environment fixed and vary only external workspace bytes."""

import os

from results.blas_workspace_v1.common import environment, torch


def configure(mode):
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    size = (32 if mode == "high" else 8) * 2**20
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.backends.cuda.cublas_workspace_size(size) == size
    assert torch.backends.cuda.cublas_workspace_size() == size
    return dict(
        environment=environment(),
        workspace_env=":4096:8",
        workspace_bytes=size,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        threads=torch.get_num_threads(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
