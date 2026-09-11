"""Regenerate data and score saved states without using Model.forward or fit scorer."""

# ruff: noqa: I001
from results.conjugated_ffn_v1.worker import dataset
from results.checkpoint_input_offload_v1.source.common import torch, boundary, tensor_hash
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import math

ROOT = Path("results/conjugated_ffn_v1")
F = torch.nn.functional


@torch.no_grad()
def independent(state, arm, x, y):
    weights = {k: v.cuda() for k, v in state.items()}

    def branch(v, prefix):
        u = F.linear(v, weights[prefix + "up.weight"], weights[prefix + "up.bias"])
        if prefix + "gate.weight" in weights:
            features = F.silu(u) * F.linear(
                v, weights[prefix + "gate.weight"], weights[prefix + "gate.bias"]
            )
        else:
            features = F.gelu(u)
        return F.linear(features, weights[prefix + "down.weight"], weights[prefix + "down.bias"])

    total = 0.0
    for start in range(0, len(x), 257):
        v = x[start : start + 257].cuda()
        if arm in ("full_gelu", "full_swiglu", "narrow_gelu"):
            prediction = branch(v, "first.")
        else:
            z = v + branch(v, "first.") / math.sqrt(2)
            prefix = "second." if arm == "untied_permuted" else "first."
            correction = branch(z.index_select(-1, weights["permutation"]), prefix).index_select(
                -1, weights["inverse"]
            )
            prediction = z + correction / math.sqrt(2) - v
        total += (
            (prediction.double() - y[start : start + 257].cuda().double()).square().sum().item()
        )
    return total / y.numel()


def run():
    p = read(ROOT / "protocol.json")
    seal = read(ROOT / "audit_protocol.json")
    hashes(seal["files"])
    hashes(p["sources"])
    hashes(p["maintained_files"])
    torch.backends.cuda.matmul.allow_tf32 = False
    boundaries = [boundary()]
    results = []
    for task in p["tasks"]:
        for seed in p["seeds"]:
            stored = torch.load(
                ROOT / f"data_{task}_{seed}.pt", map_location="cpu", weights_only=True
            )
            regenerated = dataset(task, seed)
            for key in ("x", "y", "stream"):
                assert torch.equal(stored[key], regenerated[key])
            for key in stored["stats"]:
                assert torch.equal(stored["stats"][key], regenerated["stats"][key])
            del regenerated
            for f in sorted(ROOT.glob("fit*.json")):
                row = read(f)
                if (row["task"], row["seed"]) != (task, seed):
                    continue
                assert row["stream_hash"] == tensor_hash(stored["stream"])
                assert len(row["history"]) == 300 and all(
                    math.isfinite(v["loss"]) and math.isfinite(v["grad_norm"])
                    for v in row["history"]
                )
                assert sha(row["state_path"]) == row["state_sha256"]
                state = torch.load(row["state_path"], map_location="cpu", weights_only=True)
                assert (
                    sum(v.numel() for k, v in state.items() if k not in ("permutation", "inverse"))
                    == row["parameters"]
                )
                scores = {}
                for name, lo, hi in [("validation", 4096, 6144), ("reporting", 6144, 8192)]:
                    value = independent(state, row["arm"], stored["x"][lo:hi], stored["y"][lo:hi])
                    error = abs(value - row[name]) / max(abs(row[name]), 1e-30)
                    assert error <= seal["score_relative_tolerance"], (row["index"], name, error)
                    scores[name] = dict(value=value, relative_error=error)
                results.append(dict(index=row["index"], scores=scores))
                boundaries.append(boundary())
            print("Audited", task, seed, flush=True)
    assert len(results) == 54 and len(boundaries) == 55
    write_json(
        ROOT / "audit.json",
        dict(
            passed=True,
            cases=results,
            boundaries=boundaries,
            regenerated_datasets=9,
            scores=108,
            training_updates=0,
            backwards=0,
        ),
    )


if __name__ == "__main__":
    run()
