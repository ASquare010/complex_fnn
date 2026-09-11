"""Independent numerical update, data and native-score audit."""

# ruff: noqa: I001
from results.optimizer_memory_v1.source import audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path
import os

ROOT = Path("results/native_buffer_training_v1")


def run():
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    for key in ("sources", "input_hashes", "maintained_files"):
        hashes(p[key])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    old.torch.use_deterministic_algorithms(True, warn_only=False)
    old.torch.backends.cudnn.deterministic = True
    old.torch.backends.cudnn.benchmark = False
    numerical = [
        dict(arm=row["arm"], check=old.numerical(row["measurement"], p)) for row in r["cases"]
    ]
    boundaries = [boundary()]
    scores = []
    for row in r["cases"]:
        scores.append(
            dict(arm=row["arm"], check=old.native_score_and_batches(row["measurement"], p))
        )
        boundaries.append(boundary())
    exact = []
    for f in p["fixtures"]:
        peers = {
            row["arm"]: row["measurement"]
            for row in r["cases"]
            if row["measurement"]["fixture"] == f
        }
        base = peers["native"]
        for arm in ("offload", "reuse"):
            c = peers[arm]
            checks = {}
            for key in ("first_state", "final_state"):
                for field in ("model_hash", "optimizer_hash", "sampler_hash"):
                    checks[key + "_" + field] = base[key][field] == c[key][field]
            for key in ("raw_hash", "clipped_hash"):
                checks[key] = base["first_gradients"][key] == c["first_gradients"][key]
            checks["step_loss_norm"] = all(
                (x["loss"], x["norm"]) == (y["loss"], y["norm"])
                for x, y in zip(base["updates"], c["updates"], strict=True)
            )
            checks["validation"] = all(
                base[key]["nll"] == c[key]["nll"] for key in ("before_score", "after_score")
            )
            exact.append(
                dict(dataset=f["dataset"], arm=arm, checks=checks, passed=all(checks.values()))
            )
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
            native_scores=8,
            batches=240,
        ),
    )
    print("Independent audit:", passed)


if __name__ == "__main__":
    run()
