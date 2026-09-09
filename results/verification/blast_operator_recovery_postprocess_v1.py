"""Durable analysis/report runner for the completed H080 qualification."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json

parser=argparse.ArgumentParser()
parser.add_argument("phase",choices=("analysis","report","final"))
phase=parser.parse_args().phase
root=Path("results/blast_operator_recovery_v1")
assert read(root/"coordinator_status.json")["status"]=="PASS"
assert read(root/"result.json")["status"]=="LOCALLY_QUALIFIED"
protocol=read(root/"protocol.json")
assert all(sha(n)==h for n,h in protocol["sources"].items())
source=Path(f"results/verification/blast_operator_recovery_{phase}_v1.py")
path=root/(phase+"_process.json")
logpath=root/(phase+".log")
assert not path.exists() and not logpath.exists()
record={"status":"RUNNING","source_sha256":sha(source),"coordinator_pid":os.getpid(),
        "started_utc":datetime.now(timezone.utc).isoformat(),"optimizer_updates":0,
        "qualification_repetitions":0,"result_sha256":sha(root/"result.json")}
write_json(path,record)
env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
if phase=="report":
    env["MPLCONFIGDIR"]=str((root/"report_mplconfig").resolve())
start=time.perf_counter()
with logpath.open("xb") as log:
    child=subprocess.Popen([sys.executable,"-X","faulthandler","-m",f"results.verification.blast_operator_recovery_{phase}_v1"],stdout=log,stderr=subprocess.STDOUT,env=env)
    record["pid"]=child.pid
    write_json(path,record,exclusive=False)
    code=child.wait()
    log.flush()
    os.fsync(log.fileno())
record.update(status="PASS" if code==0 else "FAIL",returncode=code,
              finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
              log_sha256=sha(logpath),source_unchanged=sha(source)==record["source_sha256"],
              scientific_source_unchanged=all(sha(n)==h for n,h in protocol["sources"].items()))
write_json(path,record,exclusive=False)
print(json.dumps(record),flush=True)
raise SystemExit(code)
