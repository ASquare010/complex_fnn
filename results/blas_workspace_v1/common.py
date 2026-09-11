"""Shared policy setup and one-batch evaluation; no algorithm fallback."""
# ruff: noqa: I001

from results.checkpoint_input_offload_v1.source.common import torch, environment
from results.streamed_evaluation_v1.source.evaluation import evaluate
import os


def configure(mode):
    expected = ":4096:8" if mode == "high" else ":16:8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == expected
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    settings = dict(
        environment=environment(),
        workspace_env=expected,
        workspace_bytes=torch.backends.cuda.cublas_workspace_size(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        threads=torch.get_num_threads(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    assert settings["workspace_bytes"] == (32 * 2**20 if mode == "high" else 128 * 1024)
    return settings


def warm(model, data):
    return evaluate(model, data, 8, 512, 1, "classifier_chunks")
