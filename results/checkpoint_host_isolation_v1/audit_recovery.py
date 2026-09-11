"""H158 isolated host qualification; immutable H157 probe, independent audit."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/checkpoint_host_isolation_v1")
PRIOR = Path("results/checkpoint_fp16_v1")
PYTHON = "C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    assert not Path(path).exists()
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify(mapping):
    for path, digest in mapping.items():
        assert sha(path) == digest, path


def launch(stage, *args):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages"),
        PYTHONMALLOC="pymalloc",
        PYTHONHASHSEED="107",
    )
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    name = stage + (args[0] if args else "")
    with (ROOT / (name + ".log")).open("x") as log:
        code = subprocess.run(
            [
                PYTHON,
                "-B",
                "-X",
                "faulthandler",
                "-u",
                "-m",
                "results.checkpoint_host_isolation_v1.study",
                stage,
                *args,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    (ROOT / (name + "_exit.txt")).write_text(str(code))
    assert code == 0, (name, code)


def prepare():
    verify(read(PRIOR / "receipt.json")["files"])
    p = read(PRIOR / "protocol.json")
    assert shutil.disk_usage(".").free > 3 * 2**30
    schedule = []
    for i, f in enumerate(p["fixtures"]):
        for arm in (
            ["native0", "offload4", "fp16"] if i % 2 == 0 else ["fp16", "offload4", "native0"]
        ):
            schedule.append(dict(index=len(schedule), fixture=f, arm=arm))
    for name, path in [("README", "README.md"), ("CURRENT_STATE", "research/CURRENT_STATE.md")]:
        (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
    write(
        ROOT / "protocol.json",
        dict(
            study="H158",
            prior_receipt=sha(PRIOR / "receipt.json"),
            sources={
                str(f).replace("\\", "/"): sha(f)
                for f in [ROOT / "study.py", Path("research/checkpoint_host_isolation_plan.md")]
            },
            schedule=schedule,
            gpu_backwards=18,
            training_updates=0,
        ),
    )


def check_protocol():
    p = read(ROOT / "protocol.json")
    verify(p["sources"])
    assert sha(PRIOR / "receipt.json") == p["prior_receipt"]
    old = read(PRIOR / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        verify(old[field])
    return p, old


def worker(index):
    from results.checkpoint_fp16_v1 import worker as probe
    from src.core.reproducibility import environment

    p, old = check_protocol()
    row = p["schedule"][index]
    probe.ROOT = ROOT
    t = probe.torch
    t.use_deterministic_algorithms(False)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = False
    assert t.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    env = environment()
    before = probe.boundary()
    result = probe.one(row["fixture"], old, row["arm"], index)
    after = probe.boundary()
    write(
        ROOT / f"boundary{index:02d}.json",
        dict(before=before, after=after, environment=env, pid=os.getpid()),
    )
    print(index, row["arm"], result["peak_bytes"], result["host"], flush=True)


def audit():
    import math

    import numpy as np

    from results.checkpoint_input_offload_v1.source.common import torch

    p, old = check_protocol()
    verify(read(PRIOR / "receipt.json")["files"])
    results = []
    cases = []
    for row in p["schedule"]:
        i = row["index"]
        assert (
            ROOT / ("worker1_retry_exit.txt" if i == 1 else f"worker{i}_exit.txt")
        ).read_text() == "0"
        c = read(ROOT / f"case{i:02d}.json")
        b = read(ROOT / f"boundary{i:02d}.json")
        assert b["before"]["gpu"] == b["after"]["gpu"] == dict(allocated=0, reserved=0)
        assert c["arm"] == row["arm"] and (c["dataset"], c["seed"]) == (
            row["fixture"]["dataset"],
            row["fixture"]["seed"],
        )
        assert sha(c["gradient_path"]) == c["gradient_sha256"]
        c["initial_host"] = b["before"]["host"]["allocated_bytes.current"]
        cases.append(c)
    original = read(PRIOR / "result.json")["cases"]
    for f in old["fixtures"]:
        cs = {c["arm"]: c for c in cases if (c["dataset"], c["seed"]) == (f["dataset"], f["seed"])}
        baseline = cs["native0"]
        prior = next(
            c
            for c in original
            if (c["dataset"], c["seed"], c["arm"]) == (f["dataset"], f["seed"], "native0")
        )
        ref = torch.load(baseline["gradient_path"], map_location="cpu", weights_only=True)
        numeric = []
        for arm, c in cs.items():
            for key in (
                "model_hash",
                "optimizer_hash",
                "tokens_hash",
                "targets_hash",
                "parameter_count",
            ):
                assert c[key] == baseline[key] == prior[key]
            grad = torch.load(c["gradient_path"], map_location="cpu", weights_only=True)
            assert grad.keys() == ref.keys() and sum(v.numel() for v in grad.values()) == 9099648
            ee = aa = bb = ab = 0.0
            per = {}
            for name in ref:
                u, v = [z[name].numpy().astype(np.float64).ravel() for z in (grad, ref)]
                assert u.shape == v.shape and np.isfinite(u).all() and np.isfinite(v).all()
                delta = u - v
                e, a, b, dot = [float(x @ y) for x, y in ((delta, delta), (u, u), (v, v), (u, v))]
                ee += e
                aa += a
                bb += b
                ab += dot
                per[name] = math.sqrt(e) / max(math.sqrt(b), 1e-8)
            error, cosine = math.sqrt(ee) / max(math.sqrt(bb), 1e-12), ab / math.sqrt(aa * bb)
            loss = abs(c["loss"] / baseline["loss"] - 1)
            half = arm == "fp16"
            passed = (
                error <= (0.002 if half else 1e-5)
                and max(per.values()) <= (0.02 if half else 1e-4)
                and loss <= 1e-6
                and (not half or cosine >= 0.99999)
            )
            if half:
                assert len(c["hook_records"]) == c["unpacks"] == 8
                assert all(
                    v["shape"] == [16, 512, 384]
                    and v["original_bytes"] == 12582912
                    and v["stored_bytes"] == 6291456
                    for v in c["hook_records"]
                )
            numeric.append(
                dict(
                    arm=arm,
                    global_relative_l2=error,
                    cosine=cosine,
                    max_tensor_relative_l2=max(per.values()),
                    loss_error=loss,
                    per_tensor=per,
                    passed=passed,
                )
            )
        half, off = cs["fp16"], cs["offload4"]
        h, n, o = [c["host"] for c in (half, baseline, off)]
        gpu_native, gpu_off = (
            half["peak_bytes"] / baseline["peak_bytes"],
            half["peak_bytes"] / off["peak_bytes"],
        )
        host_pass = (
            h["allocated_bytes.peak"] <= n["allocated_bytes.peak"]
            and h["active_bytes.peak"] <= n["active_bytes.peak"]
            and h["allocated_bytes.peak"] <= 0.01 * o["allocated_bytes.peak"]
        )
        passed = (
            all(c["initial_host"] == 0 for c in cs.values())
            and all(c["passed"] for c in numeric)
            and gpu_native <= 0.9
            and gpu_off <= 1.02
            and host_pass
        )
        results.append(
            dict(
                dataset=f["dataset"],
                seed=f["seed"],
                numeric=numeric,
                gpu_native_ratio=gpu_native,
                gpu_offload_ratio=gpu_off,
                host={arm: c["host"] for arm, c in cs.items()},
                initial_host={arm: c["initial_host"] for arm, c in cs.items()},
                passed=passed,
            )
        )
    assert not torch.cuda.is_initialized()
    write(
        ROOT / "audit.json",
        dict(
            status="EVIDENCE_VERIFIED",
            passed=all(r["passed"] for r in results),
            fixtures=results,
            cases=cases,
            gpu_backwards=18,
            training_updates=0,
        ),
    )
    print("All-fixture qualification:", all(r["passed"] for r in results))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "launch":
        launch(*sys.argv[2:])
    elif stage == "prepare":
        prepare()
    elif stage == "worker":
        worker(int(sys.argv[2]))
    elif stage == "run":
        check_protocol()
        for i in range(18):
            launch("worker", str(i))
            print("completed", i, flush=True)
    elif stage == "audit":
        audit()
    else:
        raise ValueError(stage)
