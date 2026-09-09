"""Independently inspect H072 paired tensors, represented matrices and exact moments."""

import json
import math
from fractions import Fraction
from pathlib import Path

import torch

from results.factor_balance_v2.source.logging_fix import tensor_hash
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/ungated_blockshuffle_v1")
OUT = Path("results/verification/ungated_blockshuffle_analysis_v1.json")
assert not OUT.exists()
torch.set_num_threads(4)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


p, r, process = [read(ROOT / n) for n in ("protocol.json", "result.json", "process.json")]
assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
assert r["all_checks_pass"] and len(r["checks"]) == 18
assert sha256(ROOT / "process.log") == process["log_sha256"]
assert all(sha256(Path(n)) == h for n, h in p["sources"].items())
artifacts = {n.name: sha256(n) for n in (ROOT / "checks").iterdir() if n.is_file()}
assert len(artifacts) == 35
exact = 0
for name, record in r["checks"].items():
    assert read(ROOT / "checks" / (name + ".json")) == record and record["passed"]
    if "artifact_sha256" in record:
        assert artifacts[name + ".pt"] == record["artifact_sha256"]
    if name.startswith("checkpoint_"):
        raw = torch.load(ROOT / "checks" / (name + ".pt"), map_location="cpu", weights_only=True)
        for key, left in raw["eager"].items():
            right = raw["checkpointed"][key]
            assert (
                torch.equal(left, right)
                and torch.isfinite(left).all()
                and torch.isfinite(right).all()
            )
            comparison = record["comparisons"][key]
            assert (
                tensor_hash(left)
                == tensor_hash(right)
                == comparison["first_sha256"]
                == comparison["second_sha256"]
            )
            assert left.float().norm() > 0
            exact += 1
assert exact == 72
calibration = torch.load(ROOT / "checks/calibration.pt", map_location="cpu", weights_only=True)
reconstructed = {}
for name, raw in calibration.items():
    a, b = raw["parameters"]["first.weight"], raw["parameters"]["second.weight"]
    n, m = 8 * a.shape[-1], 8 * b.shape[1]
    x = torch.eye(n, dtype=torch.float64)
    first = torch.einsum("bgi,goi->bgo", x.reshape(n, 8, n // 8), a).reshape(n, 384)
    shuffled = first.reshape(n, 8, 48).transpose(1, 2).reshape(n, 8, 48)
    second = torch.einsum("bgi,goi->bgo", shuffled, b).reshape(n, m)
    matrix = second.reshape(n, m // 8, 8).transpose(1, 2).reshape(n, m).T
    torch.testing.assert_close(matrix, raw["matrix"], rtol=1e-11, atol=1e-12)
    reconstructed[name] = matrix
for hidden in (2048, 3264):
    raw = torch.load(
        ROOT / "checks" / f"gelu_parity_h{hidden}.pt", map_location="cpu", weights_only=True
    )
    up, down = reconstructed[f"h{hidden}_up"], reconstructed[f"h{hidden}_down"]
    expected = raw["input"] @ up.T @ down.T
    tangent = 0.5 * raw["tangent"] @ up.T @ down.T
    torch.testing.assert_close(raw["difference"], expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(raw["linear"], expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(raw["jvp"], tangent, rtol=1e-11, atol=1e-12)
raw = torch.load(ROOT / "checks/swiglu_parity_separation.pt", map_location="cpu", weights_only=True)
torch.testing.assert_close(raw["even_sum"], raw["quadratic"], rtol=1e-11, atol=1e-12)
assert torch.count_nonzero(raw["jvp"]) == 0
for state in (raw["gelu_state"], raw["swiglu_state"]):
    for parameter in state.values():
        assert parameter[0, 0, 0] == 1 and torch.count_nonzero(parameter) == 1
for sign, label in ((1, "positive"), (-1, "negative")):
    t = sign * raw["witness_input"][:, 0]
    gelu = 0.5 * t * (1 + torch.erf(t / math.sqrt(2)))
    swiglu = t.square() / (1 + torch.exp(-t))
    torch.testing.assert_close(raw[f"gelu_{label}"][:, 0], gelu, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(raw[f"swiglu_{label}"][:, 0], swiglu, rtol=1e-12, atol=1e-12)
# Exact rational moment calculation is independent of the numerical quadrature.
variance = Fraction(1) + Fraction(4) + Fraction(9, 5)
residual = Fraction(4) + Fraction(9, 5) - 1
floor = residual / variance
assert variance == Fraction(34, 5) and residual == Fraction(24, 5) and floor == Fraction(12, 17)
raw = torch.load(ROOT / "checks/population_bound.pt", map_location="cpu", weights_only=True)
w = raw["weights"]
assert math.isclose(w.sum().item(), 1, rel_tol=1e-12) and (w > 0).all()
torch.testing.assert_close(
    raw["covariance"], torch.eye(4, dtype=torch.float64) * float(variance), rtol=1e-12, atol=1e-12
)
torch.testing.assert_close(
    raw["residual_covariance"],
    torch.eye(4, dtype=torch.float64) * float(residual),
    rtol=1e-12,
    atol=1e-12,
)
assert raw["linear_cross"].abs().max() <= 1e-12
ratio = float(floor)
assert math.isclose(
    r["checks"]["population_bound"]["relative_population_error_floor"],
    ratio,
    rel_tol=1e-12,
    abs_tol=1e-12,
)
counts = r["checks"]["counts_topology"]["counts"]
assert counts == {
    "gelu_same_width": 233472,
    "gelu_matched": 350208,
    "plain_swiglu": 350208,
    "full_gelu": 1179648,
    "full_swiglu": 1179648,
    "narrow_gelu": 350208,
}
record = {
    "status": "PASS",
    "scientific_verdict": r["scientific_verdict"],
    "qualification_checks": 18,
    "exact_raw_tensor_comparisons": 72,
    "independently_reconstructed_projection_matrices": 4,
    "exact_target_variance": str(variance),
    "exact_unavoidable_odd_residual_variance": str(residual),
    "exact_relative_population_error_floor": str(floor),
    "artifact_hashes": artifacts,
    "source_unchanged": True,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "research_goal_achieved": False,
    "single_bias_free_layer_scope": True,
}
write_json(OUT, record)
print(json.dumps({k: v for k, v in record.items() if k != "artifact_hashes"}))
