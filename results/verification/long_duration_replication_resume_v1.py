"""Qualified transport-only continuation of H051 after report-import worker crash."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
root = Path("results/long_duration_replication_v1")
qualification_path = root / "worker_transport_qualification_v1.json"
qualification = json.loads(qualification_path.read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == qualification["dispatcher_sha256"]
assert hashlib.sha256(Path("src/core/frozen_train_worker.py").read_bytes()).hexdigest() == qualification["minimal_worker_sha256"]

import src.core.long_duration_replication as study
from src.core.frozen_train_worker import read_cell

original = study.run_worker

def dispatch(output, name, command, timeout, attempt):
    if command[:3] != ["-m", "src.core.long_duration_replication", "worker"]:
        return original(output, name, command, timeout, attempt)
    assert len(command) == 7 and command[3] == "--recipe" and command[5] == "--seed"
    recipe, seed = command[4], int(command[6])
    cell = f"s{seed}_{recipe}"
    raw, path = read_cell(root / "protocol.json", qualification_path, cell)
    mc, tc = study.configuration(recipe, seed)
    assert raw == {"model": asdict(mc), "training": asdict(tc)}
    assert path == study.output_path(recipe, seed) and name == path.name
    worker = ["-m", "src.core.frozen_train_worker", "--protocol", str(root / "protocol.json"), "--qualification", str(qualification_path), "--cell", cell]
    return original(output, name, worker, timeout, attempt)

study.run_worker = dispatch
study.audit(root, root / "resume_qualification_v2.json")
