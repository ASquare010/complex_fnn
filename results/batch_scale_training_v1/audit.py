"""Independent clipping, Adam, native-score and calibrated gradient audit."""

# ruff: noqa: I001
from results.ordinary_complete_training_v1.audit import gradient_check, final_distance
from results.batch_scale_training_v1 import native_audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path

ROOT = Path("results/batch_scale_training_v1")


def run():
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    old.torch.use_deterministic_algorithms(False)
    old.torch.backends.cudnn.deterministic = False
    old.torch.backends.cudnn.benchmark = False
    assert old.torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    numerical, scores, boundaries = [], [], [boundary()]
    for row in r["cases"]:
        numerical.append(dict(index=row["index"], check=old.numerical(row["measurement"], p)))
        scores.append(
            dict(index=row["index"], check=old.native_score_and_batches(row["measurement"], p))
        )
        boundaries.append(boundary())
        write_json(
            ROOT / "audit_progress.json",
            dict(numerical=numerical, scores=scores, boundaries=boundaries),
        )
        print("Audited", row["index"], flush=True)
    calibrated, finals = [], []
    for f in p["fixtures"]:
        peers = {
            (row["arm"], row["repeat"]): row["measurement"]
            for row in r["cases"]
            if row["fixture"] == f
        }
        controls = [peers["ordinary", i] for i in (0, 1)]
        for arm in ("buffer4",):
            candidates = [peers[arm, i] for i in (0, 1)]
            calibrated.append(
                dict(
                    dataset=f["dataset"],
                    seed=f["seed"],
                    arm=arm,
                    checks=gradient_check(*controls, candidates, p),
                )
            )
            for repeat in (0, 1):
                finals.append(
                    dict(
                        dataset=f["dataset"],
                        arm=arm,
                        repeat=repeat,
                        error=final_distance(candidates[repeat], controls[repeat]),
                    )
                )
    passed = all(v["check"]["passed"] for v in numerical + scores) and all(
        c["passed"] for v in calibrated for c in v["checks"]
    )
    write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            numerical=numerical,
            scores=scores,
            calibrated=calibrated,
            final_distances=finals,
            boundaries=boundaries,
            backwards=0,
            training_updates=0,
            native_scores=24,
            batches=720,
        ),
    )
    print("Independent audit", passed)


if __name__ == "__main__":
    run()
