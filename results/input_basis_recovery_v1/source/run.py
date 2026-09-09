"""H093: explicit single-process replay after H092 DLL initialization failure."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.input_basis_lift_v1.source import launch as original
from results.input_basis_lift_v1.source import test_qualification as tests

ROOT = Path("results/input_basis_recovery_v1")


def main():
    prior = Path("results/input_basis_lift_v1")
    assert read(prior / "coordinator_status.json")["status"] == "FAIL"
    assert not list((prior / "qualification").iterdir())
    source = dict(read(prior / "before.json")["source_hashes"])
    source[Path(__file__).relative_to(Path.cwd()).as_posix()] = sha(__file__)
    assert all(sha(p) == h for p, h in source.items())
    preserved = {
        p.as_posix(): sha(p)
        for p in prior.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    write_json(
        ROOT / "before.json",
        {
            "source_hashes": source,
            "anchors": read(prior / "before.json")["anchors"],
            "preserved_h092": preserved,
            "reason": "WinError 1114 shm.dll initialization before numerical tests",
            "root_cause": "undetermined; memory pressure not established",
            "change": "one interpreter; invoke the unchanged 25 test functions directly",
            "optimizer_updates": 0,
            "repetitions": 1,
        },
    )
    write_json(
        ROOT / "protocol.json",
        {
            **read(ROOT / "before.json"),
            "before_sha256": sha(ROOT / "before.json"),
            "environment": original.environment(),
            "provenance": original.provenance(),
        },
    )
    (ROOT / "qualification").mkdir()
    tests.ROOT = ROOT / "qualification"
    original.ROOT = ROOT
    cases = [(f"count_{f}", tests.test_count, (f,)) for f in tests.FORMS]
    cases += [(f"formula_{f}", tests.test_formula, (f,)) for f in tests.PAIRS]
    cases += [
        (name, fn, ())
        for name, fn in (
            ("initial", tests.test_initial),
            ("collapse", tests.test_collapse),
            ("kernel", tests.test_kernel),
            ("bounds", tests.test_bounds),
        )
    ]
    cases += [(f"cuda_{f}", tests.test_cuda, (f,)) for f in tests.FORMS[:-1]]
    assert len(cases) == 25
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "completed_checks": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "qualification_process.json", status)
    started = time.perf_counter()
    try:
        for name, fn, args in cases:
            status["current_check"] = name
            write_json(ROOT / "qualification_process.json", status, exclusive=False)
            fn(*args)
            status["completed_checks"].append(name)
            print(name, "PASS", flush=True)
        original.audit()
        assert all(sha(p) == h for p, h in preserved.items())
        status["status"] = "PASS"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        status.update(
            elapsed_seconds=time.perf_counter() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(ROOT / "qualification_process.json", status, exclusive=False)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
