"""Independent NumPy FP64 gradient and FP16-roundtrip audit."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import numpy as np
import math

ROOT = Path("results/checkpoint_fp16_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
assert not (ROOT / "audit.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
assert len(r["cases"]) == 36 and len(r["diagnostics"]) == 6 and len(r["boundaries"]) == 43
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])
checks = []
comparisons = []
for f in p["fixtures"]:
    cs = {c["arm"]: c for c in r["cases"] if (c["dataset"], c["seed"]) == (f["dataset"], f["seed"])}
    assert set(cs) == set(p["arms"])
    ref = torch.load(cs["native0"]["gradient_path"], weights_only=True)
    for arm, c in cs.items():
        assert (
            c == read(ROOT / f"case{c['index']:02d}.json")
            and sha(c["gradient_path"]) == c["gradient_sha256"]
        )
        assert c["model_hash"] == f["model_hash"] and c["optimizer_hash"] == f["optimizer_hash"]
        assert (
            c["tokens_hash"] == cs["native0"]["tokens_hash"]
            and c["targets_hash"] == cs["native0"]["targets_hash"]
        )
        grad = torch.load(c["gradient_path"], weights_only=True)
        assert (
            set(grad) == set(ref)
            and sum(v.numel() for v in grad.values()) == c["parameter_count"] == 9099648
        )
        ee = aa = bb = ab = 0.0
        per = {}
        for name in ref:
            u = grad[name].numpy().astype(np.float64).ravel()
            v = ref[name].numpy().astype(np.float64).ravel()
            assert np.isfinite(u).all() and np.isfinite(v).all()
            diff = u - v
            e = float(np.dot(diff, diff))
            an = float(np.dot(u, u))
            bn = float(np.dot(v, v))
            ee += e
            aa += an
            bb += bn
            ab += float(np.dot(u, v))
            per[name] = math.sqrt(e) / max(math.sqrt(bn), 1e-8)
        error = math.sqrt(ee) / max(math.sqrt(bb), 1e-12)
        cosine = ab / math.sqrt(aa * bb)
        loss_error = abs(c["loss"] / cs["native0"]["loss"] - 1)
        approximate = arm == "fp16"
        passed = (
            error <= (0.002 if approximate else 1e-5)
            and max(per.values()) <= (0.02 if approximate else 1e-4)
            and loss_error <= 1e-6
            and (not approximate or cosine >= 0.99999)
        )
        if arm in ("identity", "fp16"):
            assert len(c["hook_records"]) == c["unpacks"] == 8
            assert all(
                v["shape"] == [16, 512, 384]
                and v["original_bytes"] == 16 * 512 * 384 * 4
                and v["stored_bytes"] == 16 * 512 * 384 * (2 if approximate else 4)
                for v in c["hook_records"]
            )
        else:
            assert not c["hook_records"] and c["unpacks"] == 0
        checks.append(
            dict(
                dataset=f["dataset"],
                seed=f["seed"],
                arm=arm,
                global_relative_l2=error,
                max_tensor_relative_l2=max(per.values()),
                global_cosine=cosine,
                loss_relative_error=loss_error,
                per_tensor=per,
                passed=passed,
            )
        )
    memory_native = cs["fp16"]["peak_bytes"] / cs["native0"]["peak_bytes"]
    memory_offload = cs["fp16"]["peak_bytes"] / cs["offload4"]["peak_bytes"]
    passed = (
        all(c["passed"] for c in checks if (c["dataset"], c["seed"]) == (f["dataset"], f["seed"]))
        and memory_native <= 0.9
        and memory_offload <= 1.02
        and cs["fp16"]["host"]["allocated_bytes.peak"] == 0
    )
    comparisons.append(
        dict(
            dataset=f["dataset"],
            seed=f["seed"],
            memory_native_ratio=memory_native,
            memory_offload_ratio=memory_offload,
            passed=passed,
        )
    )
diag_checks = []
for d in r["diagnostics"]:
    assert sha(d["path"]) == d["sha256"]
    inputs = torch.load(d["path"], weights_only=True)
    assert len(inputs) == 8
    for item, row in zip(inputs, d["blocks"], strict=True):
        x = item["input"].numpy()
        q = x.astype(np.float16).astype(np.float32)
        assert np.isfinite(q).all()
        relative = float(
            np.linalg.norm((q - x).astype(np.float64).ravel())
            / np.linalg.norm(x.astype(np.float64).ravel())
        )
        assert math.isclose(relative, row["relative_l2"], rel_tol=1e-10, abs_tol=1e-12)
        assert (
            float(np.abs(x).max()) == row["max_abs"]
            and int(((x != 0) & (q == 0)).sum()) == row["underflow_count"]
            and int((q == 0).sum()) == row["zero_count"]
        )
        diag_checks.append(
            dict(
                dataset=d["dataset"],
                seed=d["seed"],
                block=row["block"],
                relative_l2=relative,
                max_abs=row["max_abs"],
                underflow_count=row["underflow_count"],
            )
        )
assert not torch.cuda.is_initialized()
write_json(
    ROOT / "audit.json",
    dict(
        status="EVIDENCE_VERIFIED",
        checks=checks,
        comparisons=comparisons,
        diagnostics=diag_checks,
        passed=all(c["passed"] for c in comparisons),
        inputs={
            f.as_posix(): sha(f)
            for f in ROOT.iterdir()
            if f.is_file() and f.suffix in (".py", ".pt", ".json")
        },
    ),
)
print("All-fixture preflight:", all(c["passed"] for c in comparisons))
for c in comparisons:
    print(c)
