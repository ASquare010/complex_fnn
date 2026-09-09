"""Freeze H082, qualify once, then run the bounded sequential fitting grid."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

from results.latent_activation_v1.source import study as s
from src.core.reproducibility import environment, provenance

ROOT = s.ROOT
before = s.read(ROOT / "before.json")
assert not (ROOT / "protocol.json").exists()
assert s.sha(s.PLAN) == before["plan_sha256"]
assert s.sha("research/latent_activation_theory.md") == before["theory_sha256"]
assert s.sha("results/verification/blast_learning_screen_final_v1.json") == before["h081_final_sha256"]
assert s.sha("results/blast_learning_screen_v1/result.json") == before["h081_result_sha256"]
assert s.sha("results/blast_learning_screen_v1/source.zip") == before["h081_source_sha256"]
assert all(s.sha(n) == h for n, h in before["prior_sources"].items())
sources = {**before["prior_sources"], **{n.as_posix(): s.sha(n) for n in (ROOT / "source").glob("*.py")}}
assert len(sources) == 125
protocol = {
    "plan_sha256": s.sha(s.PLAN), "theory_sha256": before["theory_sha256"], "sources": sources,
    "environment": environment(), "provenance": provenance(), "forms": list(s.FORMS),
    "tasks": list(s.TASKS), "seeds": list(s.SEEDS), "rates": list(s.RATES),
    "counts": s.COUNTS, "hidden": s.HIDDEN, "steps": 600, "batch": 256,
    "train_rows": 65536, "selection_rows": 4096, "heldout_rows": 4096,
    "feature_permutation_seed": 9283, "preflight_checks": 8, "qualification_optimizer_updates": 0,
    "fitting_cells": 192, "fitting_optimizer_updates": 115200,
    "training_example_presentations": 29491200, "corpus_targets": 0,
    "full_model_resource_workers": 0, "scientific_retries": 0,
    "h081_final_sha256": before["h081_final_sha256"],
}
s.durable_json(ROOT / "protocol.json", protocol)
with (ROOT / "source.zip").open("xb") as f:
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sources:
            z.write(n, n)
        z.write(s.PLAN, s.PLAN.as_posix())
        z.write("research/latent_activation_theory.md", "research/latent_activation_theory.md")
    f.flush()
    os.fsync(f.fileno())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None


def execute(label, args):
    path, log_path = ROOT / "processes" / (label+".json"), ROOT / "processes" / (label+".log")
    command = [sys.executable, "-X", "faulthandler", *args]
    record = {"status": "RUNNING", "coordinator_pid": os.getpid(), "command": command,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": s.sha(ROOT / "protocol.json"), "source_archive_sha256": s.sha(ROOT / "source.zip")}
    s.write_json(path, record)
    started = time.perf_counter()
    with log_path.open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
        record["pid"] = child.pid
        s.write_json(path, record, exclusive=False)
        print(json.dumps({"label": label, "pid": child.pid, "status": "STARTED"}), flush=True)
        code = child.wait()
        log.flush()
        os.fsync(log.fileno())
    record.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                  finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
                  log_sha256=s.sha(log_path), source_unchanged=all(s.sha(n) == h for n, h in sources.items()))
    s.write_json(path, record, exclusive=False)
    print(json.dumps(record), flush=True)
    print(log_path.read_text(encoding="utf-8")[-3500:], flush=True)
    if code or not record["source_unchanged"]:
        raise SystemExit(code or 1)


terminal = {"status": "RUNNING", "coordinator_pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat()}
s.write_json(ROOT / "coordinator_status.json", terminal)
try:
    execute("preflight", ["-m", "pytest", "-p", "no:anyio", "-q", "--tb=short", str(ROOT / "source/test_harness.py")])
    observations = list((ROOT / "preflight").glob("*.json"))
    assert len(observations) == 8 and sum(s.read(n)["optimizer_updates"] for n in observations) == 0
    execute("fitting", ["-m", "results.latent_activation_v1.source.study"])
except BaseException as exc:
    terminal.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
    raise
else:
    terminal["status"] = "PASS"
finally:
    terminal.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    source_unchanged=all(s.sha(n) == h for n, h in sources.items()))
    s.write_json(ROOT / "coordinator_status.json", terminal, exclusive=False)
