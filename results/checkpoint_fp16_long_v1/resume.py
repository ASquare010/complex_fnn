"""Freeze the user-authorized restart, then execute only unfinished cases."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path("results/checkpoint_fp16_long_v1")
PYTHON = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def verify(m):
    for path, digest in m.items():
        assert sha(path) == digest, path


def main():
    assert not (ROOT / "resume_protocol.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
        verify(p[field])
    assert read(ROOT / "preflight.json")["passed"]
    for i in range(3):
        assert (ROOT / f"case{i:02d}_exit.txt").read_text().strip() == "0"
        assert read(ROOT / f"case{i:02d}" / "case.json")["measurement"]["optimizer_updates"] == 800
    assert not (ROOT / "case03" / "case.json").exists()
    assert shutil.disk_usage(".").free > 20 * 2**30
    preserved = [f for i in range(4) for f in (ROOT / f"case{i:02d}").rglob("*") if f.is_file()]
    preserved += [ROOT / "protocol.json", ROOT / "pause.json"]
    sources = [
        ROOT / n
        for n in (
            "resume.py",
            "resume_worker.py",
            "prepare_audit_resume.py",
            "analyze_resume.py",
            "finish_resume.py",
        )
    ]
    sources.append(Path("research/checkpoint_fp16_long_resume_plan.md"))
    rp = dict(
        status="RESUMED_BY_USER",
        sources={f.as_posix(): sha(f) for f in sources},
        preserved={f.as_posix(): sha(f) for f in preserved},
        restart_index=3,
        remaining_indices=list(range(4, 12)),
        discarded_recorded_updates=207,
        total_recorded_updates_if_complete=9807,
        total_backwards_min_if_audited=9844,
        cross_session_timing_pair=dict(dataset="wikitext2", seed=113),
    )
    (ROOT / "resume_protocol.json").write_text(json.dumps(rp, indent=2) + "\n")
    restart = ROOT / "restart"
    (restart / "case03" / "runs").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT / "protocol.json", restart / "protocol.json")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
        PYTHONMALLOC="pymalloc",
        PYTHONHASHSEED="107",
    )
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    for i in range(3, 12):
        folder = restart if i == 3 else ROOT
        module = "resume_worker" if i == 3 else "worker"
        with (folder / f"case{i:02d}.log").open("x") as log:
            code = subprocess.run(
                [
                    PYTHON,
                    "-B",
                    "-X",
                    "pycache_prefix=" + str(ROOT.resolve() / "unused_cache"),
                    "-X",
                    "faulthandler",
                    "-u",
                    "-m",
                    "results.checkpoint_fp16_long_v1." + module,
                    str(i),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        (folder / f"case{i:02d}_exit.txt").write_text(str(code))
        assert code == 0, (i, code)
        print("Completed resumed case", i, flush=True)
    verify(rp["sources"])
    verify(rp["preserved"])


if __name__ == "__main__":
    main()
