"""Independent H070 raw-tensor, routing and matrix-witness audit."""

import json
import math
from pathlib import Path

import torch

from results.factor_balance_v2.source.logging_fix import tensor_hash
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/rotated_shuffle_v1")
OUT = Path("results/verification/rotated_shuffle_analysis_v1.json")
assert not OUT.exists()
torch.set_num_threads(4)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r, p, process = [read(ROOT / n) for n in ("result.json", "protocol.json", "process.json")]
assert r["status"] == "complete" and r["all_checks_pass"] and len(r["checks"]) == 21
assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
assert all(sha256(Path(n)) == h for n, h in p["sources"].items())
assert sha256(ROOT / "process.log") == process["log_sha256"]
comparisons, angle_vectors = 0, 0
artifacts = {}
for name, check in r["checks"].items():
    path = ROOT / "checks" / (name + ".json")
    assert read(path) == check and check["passed"]
    artifacts[path.name] = sha256(path)
    if "artifact_sha256" not in check:
        continue
    path = ROOT / "checks" / (name + ".pt")
    assert sha256(path) == check["artifact_sha256"]
    artifacts[path.name] = sha256(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if name.startswith("zero_") or name.startswith("nonzero_"):
        first = raw["plain"] if name.startswith("zero_") else raw["eager"]
        second = raw["rotated"] if name.startswith("zero_") else raw["checkpointed"]
        for n, comparison in check["comparisons"].items():
            a, b = first[n], second[n]
            assert torch.equal(a, b) and torch.isfinite(a).all() and torch.isfinite(b).all()
            assert tensor_hash(a) == comparison["first_sha256"] == comparison["second_sha256"]
            assert comparison["exact"] and comparison["finite"] and comparison["max_abs_error"] == 0
            comparisons += 1
        if name.startswith("zero_"):
            assert tensor_hash(raw["input"]) == check["input_sha256"]
            for n, g in check["angle_gradients"].items():
                assert torch.isfinite(second[n]).all() and g["finite"] and g["norm"] > 0
                assert math.isclose(second[n].norm().item(), g["norm"], rel_tol=1e-7)
                angle_vectors += 1
assert comparisons == 140 and angle_vectors == 36
routes = {}
for k in (48, 64, 384):
    counts = torch.zeros(8, 8, dtype=torch.int64)
    for destination in range(k):
        origin = (destination % 8) * (k // 8) + destination // 8
        counts[destination // (k // 8), origin // (k // 8)] += 1
    assert counts.tolist() == r["checks"]["counts_topology"]["routing_counts"][str(k)]
    routes[str(k)] = counts.tolist()

raw = torch.load(ROOT / "checks/rank_witness.pt", map_location="cpu", weights_only=True)
w = raw["parameters"]
a = torch.block_diag(*w["first.weight"])
b = torch.block_diag(*w["second.weight"])
q = torch.eye(384, dtype=torch.float64)
for i, angle in enumerate(w["rotation.theta"]):
    j = 192 + (i + 1) % 192
    c, s = math.cos(angle.item()), math.sin(angle.item())
    q[i, i] = q[j, j] = c
    q[i, j] = -s
    q[j, i] = s
permutation = torch.zeros(384, 384, dtype=torch.float64)
for destination in range(384):
    permutation[destination, (destination % 8) * 48 + destination // 8] = 1
matrix = b @ q @ permutation @ a
error = (matrix - raw["canonical_matrix"]).abs().max().item()
assert error <= 1e-12
torch.testing.assert_close(matrix, raw["expected"], rtol=1e-12, atol=1e-12)
singular = torch.linalg.svdvals(matrix[:256, 48:96])
assert torch.linalg.matrix_rank(matrix[:256, 48:96], atol=1e-12) == 7
lower = singular[6:].square().sum().sqrt().item()
assert math.isclose(lower, 1 / math.sqrt(2), rel_tol=1e-12)
assert math.isclose(lower / matrix.norm().item(), 1 / math.sqrt(13), rel_tol=1e-12)
control = torch.load(ROOT / "checks/absorbable.pt", map_location="cpu", weights_only=True)
torch.testing.assert_close(
    control["rotated_matrix"], control["absorbed_matrix"], rtol=1e-12, atol=1e-14
)
for row in r["checks"]["isometry"]["rows"]:
    assert row["matrix_orthogonality_max_error"] <= 1e-13
    tolerance = 0.01 if row["device"] == "cuda" else 1e-12
    assert (
        row["max_forward_norm_relative_error"] <= tolerance
        and row["max_backward_norm_relative_error"] <= tolerance
    )
for count in r["checks"]["counts_topology"]["counts"].values():
    assert (
        count["per_ffn"] == 350784
        and count["ffn_total"] == 2806272
        and count["total_model"] == 9104256
    )
assert not r["checks"]["rank_witness"]["full_ffn_separation_claimed"]
assert not r["checks"]["isometry"]["whole_ffn_gradient_bound_claimed"]
record = {
    "status": "PASS",
    "qualification_checks": 21,
    "exact_raw_tensor_comparisons": 140,
    "finite_nonzero_angle_gradient_vectors": 36,
    "independently_reconstructed_witness_max_error": error,
    "witness_rank": 7,
    "original_block_rank_bound": 6,
    "relative_matrix_error_lower_bound": lower / matrix.norm().item(),
    "routing_counts": routes,
    "artifact_hashes": artifacts,
    "source_unchanged": True,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "research_goal_achieved": False,
}
write_json(OUT, record)
print(
    json.dumps({k: v for k, v in record.items() if k not in ("routing_counts", "artifact_hashes")})
)
