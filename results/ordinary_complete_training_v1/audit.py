"""Independent native scoring, Adam checks and bounded gradient-noise calibration."""

# ruff: noqa: I001
from results.optimizer_memory_v1.source import audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read
from pathlib import Path

ROOT = Path("results/ordinary_complete_training_v1")


def distances(left, right):
    a, b = old.arrays(left), old.arrays(right)
    global_error = old.error(a, b)
    tensor = max(old.error({k: a[k]}, {k: b[k]})["distance"] for k in a)
    return dict(
        global_relative=global_error["distance"],
        tensor_relative=tensor,
        max_absolute=global_error["max_absolute"],
    )


def gradient_check(control0, control1, candidates, p):
    g0 = old.load(control0["first_gradients"]["path"])
    g1 = old.load(control1["first_gradients"]["path"])
    cfg = p["calibration"]
    checks = []
    for kind in ("raw", "clipped"):
        noise = distances(g0[kind], g1[kind])
        limits = {
            key: min(
                cfg[key + "_cap"],
                max(cfg[key + "_floor"], cfg["multiplier"] * noise[key + "_relative"]),
            )
            for key in ("global", "tensor")
        }
        noise_passed = all(noise[k + "_relative"] <= cfg[k + "_cap"] for k in limits)
        for repeat, candidate in enumerate(candidates):
            cg = old.load(candidate["first_gradients"]["path"])
            reference = g0 if repeat == 0 else g1
            error = distances(cg[kind], reference[kind])
            passed = noise_passed and all(error[k + "_relative"] <= limits[k] for k in limits)
            checks.append(
                dict(
                    kind=kind,
                    repeat=repeat,
                    noise=noise,
                    limits=limits,
                    error=error,
                    bitwise=candidate["first_gradients"][kind + "_hash"]
                    == [control0, control1][repeat]["first_gradients"][kind + "_hash"],
                    passed=passed,
                )
            )
    return checks


def final_distance(candidate, control):
    a, b = old.load(candidate["final_state"]["path"]), old.load(control["final_state"]["path"])
    return distances(a["model"], b["model"])


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
        old.write_json(
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
        candidates = [peers["reuse", i] for i in (0, 1)]
        for arm in ("ordinary", "native"):
            controls = [peers[arm, i] for i in (0, 1)]
            calibrated.append(
                dict(
                    dataset=f["dataset"],
                    control=arm,
                    checks=gradient_check(*controls, candidates, p),
                )
            )
            for repeat in (0, 1):
                finals.append(
                    dict(
                        dataset=f["dataset"],
                        control=arm,
                        repeat=repeat,
                        error=final_distance(candidates[repeat], controls[repeat]),
                    )
                )
            finals.append(
                dict(
                    dataset=f["dataset"],
                    control=arm,
                    repeat="native_noise",
                    error=final_distance(*controls),
                )
            )
    passed = all(v["check"]["passed"] for v in numerical + scores) and all(
        c["passed"] for v in calibrated for c in v["checks"]
    )
    old.write_json(
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
            native_scores=12,
            batches=360,
        ),
    )
    print("Independent audit", passed)


if __name__ == "__main__":
    run()
