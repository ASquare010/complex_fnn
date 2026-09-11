"""Reuse complete training without changing its clipping, Adam or evaluator."""

# ruff: noqa: I001
from results.optimizer_memory_v1.source import profile as loop
from results.compact_training_v1.study import HostLedger
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.offload import install
from results.native_buffer_layout_v1.operator import model_loss
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from src.core.reproducibility import environment
from pathlib import Path
from unittest.mock import patch
import os
import sys

ROOT = Path("results/native_buffer_training_v1")
torch = loop.torch


def one(f, p, arm):
    restore = []
    original = loop.training_loss

    def loss(model, *args, **kwargs):
        if arm in ("offload", "reuse") and not restore:
            restore.append(install(model, True))
        return (model_loss if arm == "reuse" else original)(model, *args, **kwargs)

    try:
        with (
            patch.object(loop, "ROOT", ROOT / arm),
            patch.object(loop, "MemoryLedger", HostLedger),
            patch.object(loop, "training_loss", loss),
        ):
            result = loop.run_case(f, "default", p)
    finally:
        for fn in restore:
            fn()
        restore.clear()
    return dict(arm=arm, measurement=result)


def run():
    mode = sys.argv[1]
    assert mode in ("ordinary", "deterministic")
    p = read(ROOT / "protocol.json")
    for key in ("sources", "input_hashes", "maintained_files"):
        hashes(p[key])
    deterministic = mode == "deterministic"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == (":4096:8" if deterministic else None)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    settings = dict(
        environment=environment(),
        deterministic=deterministic,
        workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        benchmark=torch.backends.cudnn.benchmark,
        threads=torch.get_num_threads(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    loop.write_json(ROOT / (mode + "_environment.json"), settings)
    boundaries = [boundary()]
    cases = []
    for i, f in enumerate(p["fixtures"]):
        arms = ("native", "offload", "reuse") if deterministic else ("ordinary",)
        for arm in arms if i == 0 else tuple(reversed(arms)):
            cases.append(one(f, p, arm))
            boundaries.append(boundary())
            loop.write_json(
                ROOT / (mode + "_progress.json"),
                dict(completed=len(cases), training_updates=len(cases) * 30),
            )
    loop.write_json(
        ROOT / (mode + "_result.json"),
        dict(
            cases=cases,
            boundaries=boundaries,
            settings=settings,
            training_updates=len(cases) * 30,
            backwards=len(cases) * 30,
            training_targets=len(cases) * 122880,
        ),
    )


if __name__ == "__main__":
    run()
