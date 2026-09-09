"""H085 frozen preflight followed by one equal-budget controlled fitting grid."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.latent_offset_fit_v1.source import study as s
from src.core.reproducibility import environment, provenance

ROOT = s.ROOT
before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert sha(s.PLAN) == before["plan_sha256"]
assert all(sha(n) == h for n, h in before["anchors"].items())
assert all(sha(n) == h for n, h in before["prior_sources"].items())
assert all(sha(n) == h for n, h in before["prior_plans"].items())
sources = {**before["prior_sources"], **{n.as_posix(): sha(n) for n in (ROOT / "source").glob("*.py")}}
assert len(sources) == 135
protocol = {"plan_sha256": sha(s.PLAN), "sources": sources, "environment": environment(), "provenance": provenance(),
            "forms": list(s.FORMS), "tasks": list(s.TASKS), "seeds": list(s.SEEDS), "rates": list(s.RATES),
            "counts": s.COUNTS, "hidden": s.HIDDEN, "steps": 600, "batch": 256,
            "train_rows": 65536, "selection_rows": 4096, "heldout_rows": 4096, "input_seed": 9813,
            "feature_permutation_seed": 9283, "first_factor_lr_multiplier": 5/8,
            "preflight_checks": 8, "qualification_optimizer_updates": 0,
            "fitting_cells": 240, "fitting_optimizer_updates": 144000,
            "training_example_presentations": 36864000, "corpus_targets": 0,
            "full_model_resource_workers": 0, "scientific_retries": 0,
            "post_training_reset_ablations": 48, "selected_rows": 120,
            "candidates": ["latent_offset", "latent_offset_first_lr"]}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as f:
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sources:
            z.write(n, n)
        z.write(s.PLAN, s.PLAN.as_posix())
    f.flush()
    os.fsync(f.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None


def execute(label, args):
    path, log_path = ROOT / "processes" / (label+".json"), ROOT / "processes" / (label+".log")
    path.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-X", "faulthandler", *args]
    record = {"status": "RUNNING", "coordinator_pid": os.getpid(), "command": command,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(ROOT / "protocol.json"), "source_archive_sha256": sha(ROOT / "source.zip")}
    write_json(path, record)
    start = time.perf_counter()
    with log_path.open("xb") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
        record["pid"] = worker.pid
        write_json(path, record, exclusive=False)
        print(json.dumps({"label": label, "pid": worker.pid, "status": "STARTED"}), flush=True)
        code = worker.wait()
        log.flush()
        os.fsync(log.fileno())
    record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                  finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start,
                  log_sha256=sha(log_path), source_unchanged=all(sha(n) == h for n, h in sources.items()))
    write_json(path, record, exclusive=False)
    print(json.dumps(record), flush=True)
    print(log_path.read_text(encoding="utf-8")[-3500:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


terminal = {"status": "RUNNING", "coordinator_pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
write_json(ROOT / "coordinator_status.json", terminal)
try:
    execute("preflight", ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short", str(ROOT / "source/test_harness.py")])
    observations = list((ROOT / "preflight").glob("*.json"))
    assert len(observations) == 8 and all(read(n)["status"] == "PASS" and read(n)["optimizer_updates"] == 0 for n in observations)
    execute("fitting", ["-m", "results.latent_offset_fit_v1.source.study"])
except BaseException as exc:
    terminal.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    terminal["status"] = "PASS"
finally:
    terminal.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    source_unchanged=all(sha(n) == h for n, h in sources.items()))
    write_json(ROOT / "coordinator_status.json", terminal, exclusive=False)
