"""Independently verify H069's norm-energy records and every saved paired tensor."""

import json
import math
from pathlib import Path

import torch

from results.factor_balance_v2.source.logging_fix import tensor_hash
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/factor_balance_v2")
OLD = Path("results/factor_balance_v1")
OUT = Path("results/verification/factor_balance_analysis_v1.json")
assert not OUT.exists()
torch.set_num_threads(4)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
protocol = read(ROOT / "protocol.json")
process = read(ROOT / "diagnosis_process.json")
assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
assert sha256(ROOT / "diagnosis.log") == process["log_sha256"]
assert not r["earns_optimizer_state_update_qualification"]
assert r["optimizer_updates"] == r["corpus_targets"] == 0
projections = []
state_artifacts = {}
for state, summary in r["states"].items():
    path = ROOT / "states" / (state + ".json")
    saved = read(path)
    assert saved["summary"] == summary and len(saved["projections"]) == 24
    rows = saved["projections"]
    for row in rows:
        a = torch.tensor(row["incoming_norms"], dtype=torch.float64)
        b = torch.tensor(row["outgoing_norms"], dtype=torch.float64)
        s1, s2 = row["s1"], row["s2"]
        assert a.numel() == b.numel() == 384 and (a > 0).all() and (b > 0).all()
        optimum_scale = torch.sqrt(b / a) * (s1 / s2) ** 0.25
        exponent = torch.round(torch.log2(optimum_scale))
        assert torch.equal(exponent, torch.tensor(row["exponents"], dtype=torch.float64))
        c = 2.0**exponent
        before = a.square() / s1 + b.square() / s2
        after = (a * c).square() / s1 + (b / c).square() / s2
        optimum = 2 * a * b / math.sqrt(s1 * s2)
        ratio = (a / math.sqrt(s1)) / (b / math.sqrt(s2))
        ratio_after = ratio * c.square()
        for name, value in (
            ("energy_before", before),
            ("energy_after", after),
            ("energy_optimal", optimum),
            ("weighted_norm_ratio", ratio),
            ("weighted_norm_ratio_after", ratio_after),
        ):
            torch.testing.assert_close(
                value, torch.tensor(row[name], dtype=torch.float64), rtol=2e-14, atol=1e-14
            )
        assert (after <= before * (1 + 1e-12)).all() and (
            after <= 1.25 * optimum * (1 + 1e-12)
        ).all()
        assert (ratio_after >= 0.5 * (1 - 1e-12)).all() and (ratio_after <= 2 * (1 + 1e-12)).all()
        assert int(((ratio < 0.25) | (ratio > 4)).sum()) == row["channels_outside_fourfold"]
        row["minimum_weighted_ratio"] = ratio.min().item()
        row["maximum_weighted_ratio"] = ratio.max().item()
    assert math.isclose(
        sum(v["sum_energy_before"] for v in rows), summary["energy_before"], rel_tol=1e-14
    )
    assert math.isclose(
        sum(v["sum_energy_after"] for v in rows), summary["energy_after"], rel_tol=1e-14
    )
    assert sum(v["channels_outside_fourfold"] for v in rows) == summary["channels_outside_fourfold"]
    assert (
        summary["energy_reduction_fraction"]
        == 1 - summary["energy_after"] / summary["energy_before"]
    )
    projections.extend(rows)
    state_artifacts[path.name] = sha256(path)
assert len(projections) == 216
comparisons, check_artifacts = 0, {}
for check in r["checks"]:
    name = f"s{check['seed']}_{check['step']}_l{check['layer']}_{check['device']}"
    assert read(ROOT / "checks" / (name + ".json")) == check
    path = ROOT / "checks" / (name + ".pt")
    assert sha256(path) == check["artifact_sha256"]
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert tensor_hash(raw["input"]) == check["input_sha256"]
    for n, baseline in raw["baseline"].items():
        transformed = raw["transformed"][n]
        mapped = transformed * raw["factor_scales"][n] if n in raw["factor_scales"] else transformed
        assert torch.equal(mapped, raw["mapped"][n]) and torch.equal(baseline, mapped)
        assert torch.isfinite(baseline).all() and torch.isfinite(transformed).all()
        comparison = check["comparisons"][n]
        assert comparison["exact"] and comparison["finite"] and comparison["max_abs_error"] == 0
        assert (
            tensor_hash(baseline) == comparison["reference_sha256"] == comparison["mapped_sha256"]
        )
        comparisons += 1
    assert len(check["comparisons"]) == 8
    check_artifacts[name] = {"json": sha256(ROOT / "checks" / (name + ".json")), "pt": sha256(path)}
assert comparisons == 192 and len(r["checks"]) == 24
oldraw = torch.load(OLD / "checks/s17_0_l0_cpu.pt", map_location="cpu", weights_only=True)
newraw = torch.load(ROOT / "checks/s17_0_l0_cpu.pt", map_location="cpu", weights_only=True)


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b) and a.dtype == b.dtype
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    return a == b


assert exact_tree(oldraw, newraw)
for spec in protocol["checkpoints"].values():
    folder = Path(spec["path"])
    assert sha256(folder / "checkpoint.pt") == spec["checkpoint_sha256"]
    assert sha256(folder / "metrics.json") == spec["metrics_sha256"]
energy_gate = all(
    r["states"][f"s{s}_3200"]["energy_reduction_fraction"] >= 0.2 for s in (17, 29, 43)
)
imbalance_gate = all(
    r["states"][f"s{s}_3200"]["outside_fourfold_fraction"] >= 0.1 for s in (17, 29, 43)
)
assert energy_gate == r["every_seed_energy_gate"] and not energy_gate
assert imbalance_gate == r["every_seed_imbalance_gate"] and not imbalance_gate
record = {
    "status": "PASS",
    "scientific_verdict": r["scientific_verdict"],
    "projection_records": 216,
    "channel_records": 216 * 384,
    "complete_paired_cases": 24,
    "exact_mapped_tensor_comparisons": 192,
    "original_cpu_raw_tensors_reproduced": True,
    "original_checkpoint_and_metric_files_unchanged": 12,
    "minimum_weighted_norm_ratio": min(v["minimum_weighted_ratio"] for v in projections),
    "maximum_weighted_norm_ratio": max(v["maximum_weighted_ratio"] for v in projections),
    "channels_outside_fourfold": sum(v["channels_outside_fourfold"] for v in projections),
    "state_artifact_hashes": state_artifacts,
    "check_artifact_hashes": check_artifacts,
    "source_unchanged": True,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "research_goal_achieved": False,
}
write_json(OUT, record)
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k not in ("state_artifact_hashes", "check_artifact_hashes")
        }
    )
)
