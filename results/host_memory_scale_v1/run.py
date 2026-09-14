"""H162: H161's unchanged training loop with process host-memory measurements."""

import sys
import time
from pathlib import Path

from results.exact_offload_scale_v1 import study

ROOT = Path("results/host_memory_scale_v1")


def prepare():
    prior = study.read("results/exact_offload_scale_v1/receipt.json")
    for path, expected in prior["hashes"].items():
        assert study.sha(path) == expected, path
    p = study.verify()
    check = study.read(ROOT / "api_check.json")
    assert check["passed"] and not check["gpu_work"]
    paths = [
        *ROOT.glob("*.py"),
        ROOT / "api_check.json",
        Path("research/host_memory_scale_plan.md"),
        Path("results/exact_offload_scale_v1/receipt.json"),
        Path("results/exact_offload_scale_v1/recovery.py"),
    ]
    p["hashes"].update({f.as_posix(): study.sha(f) for f in paths})
    p.update(
        study="H162",
        frozen_at_ns=time.time_ns(),
        host_increment_limit_bytes=256 * 2**20,
        host_peak_limit_bytes=8 * 2**30,
    )
    study.write(ROOT / "protocol.json", p)


def worker(index):
    from results.host_memory_scale_v1.host import snapshot

    folder = ROOT / f"case{index:02d}"
    snapshots = {"before_torch": snapshot()}
    from results.exact_offload_scale_v1.recovery import cleanup_adapter

    cleanup_adapter()
    snapshots["before_worker"] = snapshot()
    study.worker(index)
    snapshots["after_worker"] = snapshot()
    study.write(folder / "host_memory.json", snapshots)


def audit():
    from results.exact_offload_scale_v1.recovery import cleanup_adapter

    cleanup_adapter()
    from results.exact_offload_scale_v1 import audit as base

    original_write = base.write

    def write(path, value):
        if path.name == "summary.json":
            value = dict(value, study="H162", frozen_at_ns=time.time_ns(), audit_template="H161")
        original_write(path, value)

    base.write = write
    base.main()
    from results.host_memory_scale_v1.qualify import main

    main()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        prepare()
    else:
        study.ROOT = ROOT
        study.__file__ = __file__
        if mode == "run":
            study.run()
        elif mode == "worker":
            worker(int(sys.argv[2]))
        elif mode == "audit":
            audit()
        else:
            raise ValueError(mode)
