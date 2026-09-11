"""One fresh long-training run with qualified offload and buffer loss."""

# ruff: noqa: I001
from results.ordinary_long_training_v1 import loop
from results.ordinary_long_training_v1.io import write_json
from results.compact_training_v1.study import HostLedger
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.offload import install
from results.native_buffer_layout_v1.operator import model_loss
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path
from unittest.mock import patch
import datetime
import os
import subprocess
import sys
import time

ROOT = Path("results/ordinary_long_training_v1")
torch = loop.torch


def one(row, p, folder):
    restore = []
    original = loop.training_loss

    def loss(model, *args, **kwargs):
        if row["arm"] != "ordinary" and not restore:
            restore.append(install(model, True))
        return (model_loss if row["arm"] == "reuse" else original)(model, *args, **kwargs)

    try:
        with (
            patch.object(loop, "ROOT", folder),
            patch.object(loop, "MemoryLedger", HostLedger),
            patch.object(loop, "training_loss", loss),
            patch.object(loop, "write_json", write_json),
        ):
            return loop.run_case(row["fixture"], "fp32_default_native", p)
    finally:
        for fn in restore:
            fn()
        restore.clear()


def run():
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
        hashes(p[field])
    row = p["schedule"][int(sys.argv[1])]
    folder = ROOT / f"case{row['index']:02d}"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    assert torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    settings = dict(
        environment=loop.environment(),
        deterministic=False,
        workspace_bytes=torch.backends.cuda.cublas_workspace_size(),
        threads=torch.get_num_threads(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    boundaries = [boundary()]
    fields = "timestamp,index,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu"
    metadata = dict(
        start_ns=time.time_ns(),
        timezone=str(datetime.datetime.now().astimezone().tzinfo),
        fields=fields,
    )
    with (
        (folder / "telemetry.csv").open("x") as out,
        (folder / "telemetry.stderr").open("x") as err,
    ):
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=out,
            stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            result = one(row, p, folder)
            boundaries.append(boundary())
            assert monitor.poll() is None
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            metadata.update(end_ns=time.time_ns(), monitor_stopped=monitor.poll() is not None)
            write_json(folder / "telemetry_metadata.json", metadata)
    for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
        hashes(p[field])
    write_json(
        folder / "case.json",
        dict(**row, measurement=result, settings=settings, boundaries=boundaries),
    )
    print("Complete", row["index"], row["arm"], flush=True)


if __name__ == "__main__":
    run()
