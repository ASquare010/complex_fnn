"""Preserve plotting failure; try the successful dependency order once, without CUDA.

The native plot module remains unchanged. This is postprocessing only, using
the same UV-managed interpreter and packages. Preload is an operational attempt,
not a demonstrated fix for the intermittent Windows native import failure.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")


def main():
    assert (ROOT / "plot_exit.txt").read_text().strip() == "3221225477"
    paths = [
        ROOT / name
        for name in (
            "plot.log",
            "plot_exit.txt",
            "summary.json",
            "audit.json",
            "source/plot.py",
            "source/launch.py",
        )
    ]
    paths.append(Path(__file__).resolve().relative_to(Path.cwd()))
    record = dict(
        optimizer_updates=0,
        cuda_requested=False,
        root_cause_proved=False,
        files={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    with (ROOT / "plot_recovery_before.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    import sympy
    import torch
    import torch._dynamo

    print(f"CPU preload passed: sympy={sympy.__version__}, torch={torch.__version__}", flush=True)
    assert not torch.cuda.is_initialized()
    from results.whole_job_memory_workspace_v1.source.plot import run

    run()
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
