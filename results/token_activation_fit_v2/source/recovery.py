"""One explicit unchanged recovery of H068 after verified process loss."""

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from results.token_activation_fit_v1.source import study
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/token_activation_fit_v2")
OLD = Path("results/token_activation_fit_v1")
RECOVERY = Path("research/token_activation_fit_recovery_plan.md")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


if "--worker" in sys.argv:
    study.ROOT = ROOT
    original_make_data = study.make_data

    def matching_data():
        data = original_make_data()
        manifest = read(OLD / "data_manifest.json")
        assert study.tensor_sha(data["x"]) == manifest["x_sha256"]
        assert {t: study.tensor_sha(y) for t, y in data["targets"].items()} == manifest["targets"]
        assert {str(s): study.tensor_sha(v) for s, v in data["streams"].items()} == manifest[
            "streams"
        ]
        return data

    study.make_data = matching_data
    study.run()
    a, b = read(OLD / "data_manifest.json"), read(ROOT / "data_manifest.json")
    assert all(a[k] == b[k] for k in ("x_sha256", "targets", "streams"))
    study.write_new(
        ROOT / "data_reproduction.json",
        {
            "tensor_data_and_streams_exact": True,
            "original_data_sha256": a["data_sha256"],
            "recovered_data_sha256": b["data_sha256"],
        },
    )
    raise SystemExit(0)

assert not (ROOT / "protocol.json").exists()
protocol = read(OLD / "protocol.json")
assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
assert read(OLD / "harness_process.json")["status"] == "PASS"
assert read(OLD / "interruption.json")["completed_endpoints"] == 0
sources = dict(protocol["sources"])
for name in ("data_manifest.json", "interruption.json", "harness_process.json"):
    sources[(OLD / name).as_posix()] = sha256(OLD / name)
sources[RECOVERY.as_posix()] = sha256(RECOVERY)
sources[Path(__file__).relative_to(Path.cwd()).as_posix()] = sha256(Path(__file__))
protocol.update(
    sources=sources,
    recovery_plan_sha256=sha256(RECOVERY),
    original_protocol_sha256=sha256(OLD / "protocol.json"),
    original_interruption_sha256=sha256(OLD / "interruption.json"),
    unchanged_harness_process_sha256=sha256(OLD / "harness_process.json"),
    recovery_attempt=1,
    output_root=ROOT.as_posix(),
)
study.write_new(ROOT / "protocol.json", protocol)
with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for name in sources:
        z.write(name, name)
    z.write(study.PLAN, study.PLAN.as_posix())
command = [
    sys.executable,
    "-X",
    "faulthandler",
    "-m",
    "results.token_activation_fit_v2.source.recovery",
    "--worker",
]
record = {
    "status": "RUNNING",
    "command": command,
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "coordinator_pid": os.getpid(),
    "protocol_sha256": sha256(ROOT / "protocol.json"),
    "source_zip_sha256": sha256(ROOT / "source.zip"),
}
study.write_new(ROOT / "process.json", record)
start = time.perf_counter()
with (ROOT / "process.log").open("x", encoding="utf-8") as log:
    child = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
    )
    record["pid"] = child.pid
    write_json(ROOT / "process.json", record)
    code = child.wait()
record.update(
    status="PASS" if code == 0 else "FAIL",
    returncode=code,
    finished_utc=datetime.now(timezone.utc).isoformat(),
    elapsed_seconds=time.perf_counter() - start,
    log_sha256=sha256(ROOT / "process.log"),
    source_unchanged=all(sha256(Path(n)) == h for n, h in sources.items()),
)
write_json(ROOT / "process.json", record)
print(json.dumps(record), flush=True)
raise SystemExit(code or (0 if record["source_unchanged"] else 1))
