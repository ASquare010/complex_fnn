"""H163 launcher: long loop plus validated process counters and cleanup."""

import sys
from pathlib import Path

from results.exact_offload_long_scale_v1 import study

ROOT = Path("results/exact_offload_long_scale_v1")


def worker(index):
    from results.host_memory_scale_v1.host import snapshot

    values = dict(before_torch=snapshot())
    from results.exact_offload_scale_v1.recovery import cleanup_adapter

    cleanup_adapter()
    values["before_worker"] = snapshot()
    study.worker(index)
    values["after_worker"] = snapshot()
    study.write(ROOT / f"case{index:02d}" / "host_memory.json", values)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        study.prepare()
    else:
        study.__file__ = __file__
        if mode == "run":
            study.run()
        elif mode == "worker":
            worker(int(sys.argv[2]))
        elif mode == "audit":
            from results.exact_offload_scale_v1.recovery import cleanup_adapter

            cleanup_adapter()
            from results.exact_offload_long_scale_v1.audit import main

            main()
            from results.exact_offload_long_scale_v1.qualify import main as qualify

            qualify()
        else:
            raise ValueError(mode)
