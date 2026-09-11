"""Reuse independent NumPy AdamW and native score/batch checks."""

# ruff: noqa: I001
from results.optimizer_memory_v1.source import audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path

ROOT = Path("results/compact_training_v1")


def run():
    assert not (ROOT / "audit.json").exists()
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    numerical = []
    for row in r["cases"]:
        numerical.append(dict(arm=row["arm"], check=old.numerical(row["measurement"], p)))
    boundaries = [boundary()]
    scores = []
    for row in r["cases"]:
        scores.append(
            dict(arm=row["arm"], check=old.native_score_and_batches(row["measurement"], p))
        )
        boundaries.append(boundary())
    gradients = []
    for f in p["fixtures"]:
        cases = [row for row in r["cases"] if row["measurement"]["fixture"] == f]
        base, next_case = [
            next(row["measurement"] for row in cases if row["arm"] == arm)
            for arm in ("baseline", "combined")
        ]
        a, b = [
            old.arrays(old.load(c["first_gradients"]["path"])["raw"]) for c in (base, next_case)
        ]
        ge = old.error(a, b)["distance"]
        te = max(old.error({k: a[k]}, {k: b[k]})["distance"] for k in a)
        gradients.append(
            dict(
                fixture=f["label"],
                global_relative=ge,
                max_tensor_relative=te,
                passed=ge <= 1e-5 and te <= 1e-4,
            )
        )
    passed = all(row["check"]["passed"] for row in numerical + scores) and all(
        v["passed"] for v in gradients
    )
    old.write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            numerical=numerical,
            scores=scores,
            gradients=gradients,
            boundaries=boundaries,
            optimizer_updates=0,
            backwards=0,
            native_scores=4,
            batches=120,
        ),
    )
    print("Independent audit:", passed, flush=True)


if __name__ == "__main__":
    run()
