"""One explicit continuation after the recorded terminal torch-import failure."""

import json
import subprocess
import sys
import traceback
from pathlib import Path

from src.core.joint_conditioning import PLAN, SEEDS, TAIL, compare_seed
from src.core.reproducibility import sha256, write_json

output = Path("results/joint_conditioning_scale384_v1")
protocol = json.loads((output / "protocol.json").read_text())
assert sha256(PLAN) == protocol["plan_sha256"]
assert sha256(TAIL) == protocol["reference_audit_sha256"]
critical = [
    "src/core/joint_conditioning.py",
    "src/core/conditioning.py",
    "src/core/compile_serving.py",
    "src/core/fused_serving.py",
    "src/core/model_audit.py",
    "src/core/benchmark.py",
    "src/core/validation_tail.py",
    "src/core/data.py",
    "src/core/transformer.py",
    "src/core/config.py",
    "src/blockshuffle_ffn/triton_inference.py",
    "src/blockshuffle_ffn/__init__.py",
    "src/grouped_ffn/__init__.py",
    "pyproject.toml",
    "uv.lock",
]
assert all(sha256(Path(p)) == protocol["provenance"]["source_files"][p] for p in critical)
probes = json.loads(Path("results/verification/joint_import_probes_v1.json").read_text())
assert len(probes) == 3 and all(r["returncode"] == 0 for r in probes)
assert (output / "failure.json").exists() and not (output / "s17_floor").exists()
continuation = {
    "reason": "Terminal 3221225477 access violation during torch import, before s17_floor worker setup. Three unchanged fresh import probes passed. Resources were available; root cause unresolved.",
    "original_failure_preserved": True,
    "critical_sources_unchanged": critical,
    "script_sha256": sha256(Path(__file__)),
    "additional_retry_policy": "none",
}
write_json(output / "continuation.json", continuation)
records = {"s17_original": json.loads((output / "s17_original/result.json").read_text())}
try:
    for seed, floored in ((17, True), (29, False), (29, True), (43, False), (43, True)):
        name = f"s{seed}_" + ("floor" if floored else "original")
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "src.core.joint_conditioning",
            "worker",
            "--seed",
            str(seed),
            "--output",
            str(output / name),
        ]
        if floored:
            command.append("--floored")
        logpath = output / (f"{name}_retry1.log" if seed == 17 else f"{name}.log")
        print(f"Starting {name}", flush=True)
        with logpath.open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=420)
        records[name] = json.loads((output / name / "result.json").read_text())
        write_json(
            output / "progress.json",
            {"status": "running", "completed": list(records), "continuation": True},
        )
        print(f"Completed {name}", flush=True)
    references = json.loads(TAIL.read_text())["records"]
    gates = {}
    for seed in SEEDS:
        controls = {
            k: references[f"scale384_{k}_s{seed}_800"]
            for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
        }
        gates[seed] = compare_seed(
            records[f"s{seed}_original"], records[f"s{seed}_floor"], controls
        )
    result = {
        "status": "complete",
        "protocol": protocol,
        "continuation": continuation,
        "records": records,
        "gates": gates,
        "all_gates_pass": all(all(g.values()) for g in gates.values()),
    }
    write_json(output / "result.json", result)
    write_json(
        output / "progress.json",
        {"status": "complete", "completed": list(records), "continuation": True},
    )
    print(json.dumps({"all_gates_pass": result["all_gates_pass"], "gates": gates}), flush=True)
except Exception as exc:
    write_json(
        output / "continuation_failure.json",
        {"error": repr(exc), "traceback": traceback.format_exc(), "completed": list(records)},
    )
    raise
