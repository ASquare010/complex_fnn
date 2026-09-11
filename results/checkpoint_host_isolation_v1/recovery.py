"""One documented pre-model startup retry, then finish untouched cases."""

import os
import subprocess
from pathlib import Path

from results.checkpoint_host_isolation_v1 import study as s


def main():
    p, _ = s.check_protocol()
    assert (s.ROOT / "run_exit.txt").read_text() == "1"
    assert (s.ROOT / "worker1_exit.txt").read_text() == "3221225477"
    assert (s.ROOT / "worker0_exit.txt").read_text() == "0"
    for name in ("case01.json", "gradient01.pt", "boundary01.json"):
        assert not (s.ROOT / name).exists()
    paths = [
        s.ROOT / "recovery.py",
        s.ROOT / "launch_recovery.py",
        s.ROOT / "audit_recovery.py",
        Path("research/checkpoint_host_isolation_recovery.md"),
        s.ROOT / "run.log",
        s.ROOT / "worker1.log",
        s.ROOT / "worker1_exit.txt",
    ]
    s.write(
        s.ROOT / "recovery_protocol.json",
        dict(
            original_protocol=s.sha(s.ROOT / "protocol.json"),
            inputs={f.as_posix(): s.sha(f) for f in paths},
            retry_index=1,
            remaining_indices=list(range(2, 18)),
            gates_unchanged=True,
        ),
    )
    for i in range(1, len(p["schedule"])):
        name = "worker1_retry" if i == 1 else f"worker{i}"
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
            PYTHONMALLOC="pymalloc",
            PYTHONHASHSEED="107",
        )
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        with (s.ROOT / (name + ".log")).open("x") as log:
            code = subprocess.run(
                [
                    s.PYTHON,
                    "-B",
                    "-X",
                    "pycache_prefix=" + str(s.ROOT.resolve() / "unused_cache"),
                    "-X",
                    "faulthandler",
                    "-u",
                    "-m",
                    "results.checkpoint_host_isolation_v1.study",
                    "worker",
                    str(i),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        (s.ROOT / (name + "_exit.txt")).write_text(str(code))
        assert code == 0, (name, code)
        print("completed", i, flush=True)


if __name__ == "__main__":
    main()
