"""H161 cleanup-only recovery; original worker and failed run stay immutable."""

import sys
from pathlib import Path

from results.exact_offload_scale_v1 import study

ORIGINAL_ROOT = study.ROOT
ROOT = ORIGINAL_ROOT / "recovery"
BASE_VERIFY = study.verify


def verify():
    protocol = BASE_VERIFY()
    recovery = study.read(ORIGINAL_ROOT / "recovery_protocol.json")
    for path, expected in recovery["hashes"].items():
        assert study.sha(path) == expected, path
    return protocol


def prepare():
    study.verify()
    diagnosis = study.read(ORIGINAL_ROOT / "cleanup_diagnosis.json")
    assert diagnosis["after_workspace_clear"] == [0, 0]
    assert diagnosis["after_original_cleanup"][0] > 0
    assert (ORIGINAL_ROOT / "case00/exit.txt").read_text() == "1"
    assert len((ORIGINAL_ROOT / "case00/history.jsonl").read_text().splitlines()) == 30
    assert not (ORIGINAL_ROOT / "case00/result.json").exists()
    paths = [
        Path(__file__),
        Path("research/exact_offload_scale_recovery_plan.md"),
        ORIGINAL_ROOT / "diagnose.py",
        ORIGINAL_ROOT / "cleanup_diagnosis.json",
    ]
    paths += [f for f in (ORIGINAL_ROOT / "case00").rglob("*") if f.is_file()]
    study.write(
        ORIGINAL_ROOT / "recovery_protocol.json",
        dict(
            hashes={f.as_posix(): study.sha(f) for f in paths},
            discarded_updates=30,
            discarded_backwards=31,
            accepted_updates=180,
            total_updates=210,
            total_backwards=217,
        ),
    )
    ROOT.mkdir(exist_ok=False)
    (ROOT / "protocol.json").write_bytes((ORIGINAL_ROOT / "protocol.json").read_bytes())


def cleanup_adapter():
    import torch

    original = torch.cuda.empty_cache

    def clear():
        torch.clear_autocast_cache()
        torch._C._cuda_clearCublasWorkspaces()
        original()
        torch.cuda.synchronize()

    torch.cuda.empty_cache = clear


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        prepare()
    else:
        study.ROOT = ROOT
        study.__file__ = __file__
        study.verify = verify
        if mode == "run":
            study.run()
        elif mode == "worker":
            cleanup_adapter()
            study.worker(int(sys.argv[2]))
        elif mode == "audit":
            cleanup_adapter()
            from results.exact_offload_scale_v1.audit import main

            main()
        else:
            raise ValueError(mode)
