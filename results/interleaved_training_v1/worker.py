"""One isolated policy per run, with passive GPU telemetry."""

# ruff: noqa: I001
from results.interleaved_training_v1 import loop
from results.compact_training_v1.study import HostLedger
from results.checkpoint_input_offload_v1.source.common import boundary
from results.partial_offload_training_v1.offload import install_subset
from results.native_buffer_layout_v1.operator import model_loss
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from src.core.reproducibility import environment
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
from unittest.mock import patch
import datetime
import os
import subprocess
import time

ROOT = Path("results/interleaved_training_v1")
torch = loop.torch


def one(row, p, folder):
    restore = []
    original = loop.training_loss

    counts = {"ordinary": 0, "buffer0": 0, "buffer4": 4, "buffer8": 8}
    selected = []

    def loss(model, *args, **kwargs):
        if not restore:
            fn, blocks = install_subset(model, counts[row["arm"]])
            restore.append(fn)
            selected.extend(blocks)
        return (original if row["arm"] == "ordinary" else model_loss)(model, *args, **kwargs)

    try:
        with (
            patch.object(loop, "ROOT", folder),
            patch.object(loop, "MemoryLedger", HostLedger),
            patch.object(loop, "training_loss", loss),
            patch.object(loop, "write_json", write_json),
        ):
            result = loop.run_case(row["fixture"], "default", p)
            assert selected == list(range(8 - counts[row["arm"]], 8))
            return dict(**result, offloaded_blocks=selected)
    finally:
        for fn in restore:
            fn()
        restore.clear()


def run():
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    assert torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    settings = dict(
        environment=environment(),
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
    with (ROOT / "telemetry.csv").open("x") as out, (ROOT / "telemetry.stderr").open("x") as err:
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=out,
            stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            for row in p["schedule"]:
                folder = ROOT / f"case{row['index']:02d}"
                assert not (folder / "case.json").exists()
                start = time.perf_counter()
                result = one(row, p, folder)
                segment_wall_ms = 1000 * (time.perf_counter() - start)
                boundaries.append(boundary())
                write_json(
                    folder / "case.json",
                    dict(
                        **row,
                        measurement=result,
                        settings=settings,
                        segment_wall_ms=segment_wall_ms,
                    ),
                )
                write_json(ROOT / "boundaries.json", boundaries)
                print("Complete", row["index"], row["arm"], flush=True)
                assert monitor.poll() is None
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            metadata.update(end_ns=time.time_ns(), monitor_stopped=monitor.poll() is not None)
            write_json(ROOT / "telemetry_metadata.json", metadata)
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])


if __name__ == "__main__":
    run()
