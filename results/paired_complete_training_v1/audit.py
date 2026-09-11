"""Independent inherited clipping/Adam, sampler and native-score audits."""

# ruff: noqa: I001
from results.optimizer_memory_v1.source import audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path

ROOT = Path("results/paired_complete_training_v1")


def run():
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    old.torch.use_deterministic_algorithms(True, warn_only=False)
    old.torch.backends.cudnn.deterministic = True
    old.torch.backends.cudnn.benchmark = False
    assert old.torch.backends.cuda.cublas_workspace_size(8 * 2**20) == 8 * 2**20
    numerical, scores, boundaries = [], [], [boundary()]
    for row in r["cases"]:
        numerical.append(dict(index=row["index"], check=old.numerical(row["measurement"], p)))
        scores.append(
            dict(index=row["index"], check=old.native_score_and_batches(row["measurement"], p))
        )
        boundaries.append(boundary())
        old.write_json(
            ROOT / "audit_progress.json",
            dict(numerical=numerical, scores=scores, boundaries=boundaries),
        )
        print("Audited", row["index"], flush=True)
    exact = []
    for f in p["fixtures"]:
        peers = [row for row in r["cases"] if row["fixture"] == f and row["arm"] != "ordinary"]
        base = next(row["measurement"] for row in peers if row["arm"] == "native")
        for row in peers:
            c, checks = row["measurement"], {}
            for key in ("first_state", "final_state"):
                for field in ("model_hash", "optimizer_hash", "sampler_hash"):
                    checks[key + "_" + field] = base[key][field] == c[key][field]
            for key in ("raw_hash", "clipped_hash"):
                checks[key] = base["first_gradients"][key] == c["first_gradients"][key]
            checks["loss_norm"] = all(
                (x["loss"], x["norm"]) == (y["loss"], y["norm"])
                for x, y in zip(base["updates"], c["updates"], strict=True)
            )
            checks["validation"] = all(
                base[key] == c[key] for key in ("before_score", "after_score")
            )
            exact.append(dict(index=row["index"], checks=checks, passed=all(checks.values())))
    passed = all(x["check"]["passed"] for x in numerical + scores) and all(
        x["passed"] for x in exact
    )
    old.write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            numerical=numerical,
            scores=scores,
            exact=exact,
            boundaries=boundaries,
            backwards=0,
            training_updates=0,
            native_scores=12,
            batches=360,
        ),
    )
    print("Independent audit", passed)


if __name__ == "__main__":
    run()
