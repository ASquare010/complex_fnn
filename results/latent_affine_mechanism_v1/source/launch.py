"""Durable one-shot H084 coordinator; freeze sources before checkpoint interventions."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.latent_affine_mechanism_v1.source.study import PLAN, ROOT, check_prior
from src.core.reproducibility import environment, provenance

before = read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
check_prior(before)
assert sha(PLAN) == before["plan_sha256"]
sources = {**before["prior_sources"], **{n.as_posix(): sha(n) for n in (ROOT / "source").glob("*.py")}}
assert len(sources) == 131
protocol = {"plan_sha256": sha(PLAN), "sources": sources, "environment": environment(),
            "provenance": provenance(), "checkpoint_cases": 12, "reporting_passes": 72,
            "reporting_examples": 294912, "optimizer_updates": 0, "corpus_targets": 0,
            "fp64_rtol": 1e-10, "fp64_atol": 1e-10, "fp32_fold_relative_mse_tolerance": 1e-5,
            "training_repetitions": 0, "independent_new_reset_passes": 24,
            "selected_cells": [r["cell"] for r in before["selected_rows"]]}
write_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as f:
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sources:
            z.write(n, n)
        z.write(PLAN, PLAN.as_posix())
    f.flush()
    os.fsync(f.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
command = [sys.executable, "-X", "faulthandler", "-m", "results.latent_affine_mechanism_v1.source.study"]
record = {"status": "RUNNING", "coordinator_pid": os.getpid(), "command": command,
          "started_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0,
          "source_archive_sha256": sha(ROOT / "source.zip"), "protocol_sha256": sha(ROOT / "protocol.json")}
write_json(ROOT / "process.json", record)
start = time.perf_counter()
with (ROOT / "study.log").open("xb") as log:
    worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    record["pid"] = worker.pid
    write_json(ROOT / "process.json", record, exclusive=False)
    code = worker.wait()
    log.flush()
    os.fsync(log.fileno())
record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
              finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start,
              log_sha256=sha(ROOT / "study.log"), source_unchanged=all(sha(n) == h for n, h in sources.items()))
write_json(ROOT / "process.json", record, exclusive=False)
print(json.dumps(record), flush=True)
print((ROOT / "study.log").read_text(encoding="utf-8")[-3500:], flush=True)
raise SystemExit(code or int(not record["source_unchanged"]))
