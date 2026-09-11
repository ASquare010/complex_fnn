"""Independently check saved tensors, repeat hashes, numerical metrics and budgets."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read

ROOT = Path("results/attention_reproducibility_v1")
torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
p = read(ROOT / "protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
summary = []
for mode in p["policies"]:
    r = read(ROOT / mode / "result.json")
    rows = r["cases"]
    assert len(rows) == r["counts"]["backwards"] <= 12
    assert r["training_updates"] == 0 and r["diagnostic_targets"] == len(rows) * 4096
    assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])
    groups = []
    artifact_checks = 0
    for fixture in p["fixtures"]:
        peers = [x for x in rows if x["dataset"] == fixture["dataset"]]
        if len(peers) != 6:
            groups.append(dict(dataset=fixture["dataset"], complete=False, passed=False))
            continue
        anchor = next(x for x in peers if x["arm"] == "native_resident" and x["repetition"] == 0)
        ref = torch.load(anchor["artifact"]["path"], map_location="cpu", weights_only=True)
        assert len(ref) == 50
        for row in peers:
            first = next(x for x in peers if x["arm"] == row["arm"] and x["repetition"] == 0)
            assert row["provenance"] == anchor["provenance"]
            assert row["same_arm_exact"] == (
                row["hashes"] == first["hashes"] and row["loss"] == first["loss"]
            )
            assert row["exact_loss"] == (row["loss"] == anchor["loss"])
            assert row["comparison"]["exact"] == (row["hashes"] == anchor["hashes"])
            assert row["comparison"]["exact_tensors"] == sum(
                row["hashes"][k] == anchor["hashes"][k] for k in anchor["hashes"]
            )
            assert (
                row["finite"]
                and row["state_unchanged"]
                and row["parameters"] == 9099648
                and row["operators"]
            )
            if "artifact" not in row:
                continue
            artifact_checks += 1
            value = torch.load(row["artifact"]["path"], map_location="cpu", weights_only=True)
            ds = rs = 0.0
            for key, tensor in value.items():
                raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                assert hashlib.sha256(raw).hexdigest() == row["hashes"][key]
                x = tensor.numpy().astype(np.float64)
                base = ref[key].numpy().astype(np.float64)
                assert np.isfinite(x).all()
                delta = x - base
                d2 = float(np.sum(delta * delta))
                r2 = float(np.sum(base * base))
                ds += d2
                rs += r2
                expected = row["comparison"]["layers"][key]
                assert (
                    raw == ref[key].contiguous().view(torch.uint8).numpy().tobytes()
                ) == expected["exact"]
                assert np.isclose(
                    float(np.max(np.abs(delta))), expected["max_abs"], atol=1e-15, rtol=1e-10
                )
                assert np.isclose(
                    (d2 / max(r2, 1e-60)) ** 0.5, expected["relative_l2"], atol=1e-15, rtol=1e-10
                )
            assert np.isclose(
                (ds / max(rs, 1e-60)) ** 0.5,
                row["comparison"]["relative_l2"],
                atol=1e-15,
                rtol=1e-10,
            )
        groups.append(
            dict(
                dataset=fixture["dataset"],
                complete=True,
                passed=all(
                    x["comparison"]["exact"] and x["exact_loss"] and x["same_arm_exact"]
                    for x in peers
                ),
                exact_to_native=sum(x["comparison"]["exact"] and x["exact_loss"] for x in peers),
                same_arm_repeat_exact=sum(
                    x["same_arm_exact"] for x in peers if x["repetition"] == 1
                ),
                max_relative_l2=max(x["comparison"]["relative_l2"] for x in peers),
                operators=sorted(set(k for x in peers for k in x["operators"])),
            )
        )
    passed = r["failure"] is None and all(x["passed"] for x in groups)
    assert passed == r["passed"]
    summary.append(
        dict(
            policy=mode,
            passed=passed,
            groups=groups,
            artifact_checks=artifact_checks,
            backwards=r["counts"]["backwards"],
            failure=r["failure"],
        )
    )
assert not torch.cuda.is_initialized()
result = dict(
    verification_passed=True,
    policies=summary,
    backwards=0,
    training_updates=0,
    total_study_backwards=sum(x["backwards"] for x in summary),
    broad_goal_achieved=False,
)
(ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
