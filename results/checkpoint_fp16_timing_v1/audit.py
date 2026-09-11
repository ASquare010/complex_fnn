"""Independent first-Adam arithmetic, native scoring, batches and raw gradients."""

# ruff: noqa: I001
from results.batch_scale_training_v1 import native_audit as old
from results.checkpoint_input_offload_v1.source.common import boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import math
import numpy as np

ROOT = Path("results/checkpoint_fp16_timing_v1")


def gradient(candidate, reference, approximate):
    a = old.load(candidate["first_gradients"]["path"])["raw"]
    b = old.load(reference["first_gradients"]["path"])["raw"]
    assert a.keys() == b.keys()
    ee = aa = bb = ab = 0.0
    per = {}
    for name in a:
        u, v = [s[name].numpy().astype(np.float64).ravel() for s in (a, b)]
        assert u.shape == v.shape and np.isfinite(u).all() and np.isfinite(v).all()
        delta = u - v
        e = float(delta @ delta)
        vn = float(v @ v)
        ee += e
        aa += float(u @ u)
        bb += vn
        ab += float(u @ v)
        per[name] = math.sqrt(e) / max(math.sqrt(vn), 1e-8)
    error = math.sqrt(ee) / max(math.sqrt(bb), 1e-12)
    cosine = ab / math.sqrt(aa * bb)
    loss = abs(candidate["updates"][0]["loss"] / reference["updates"][0]["loss"] - 1)
    passed = (
        error <= (0.002 if approximate else 1e-5)
        and max(per.values()) <= (0.02 if approximate else 1e-4)
        and loss <= 1e-6
        and (not approximate or cosine >= 0.99999)
    )
    return dict(
        global_relative_l2=error,
        max_tensor_relative_l2=max(per.values()),
        cosine=cosine,
        loss_relative_error=loss,
        per_tensor=per,
        passed=passed,
    )


def main():
    assert not (ROOT / "audit.json").exists()
    assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    cases = [read(ROOT / f"case{i:02d}" / "case.json") for i in range(36)]
    files = [*ROOT.glob("case*/case.json")]
    for row in cases:
        c = row["measurement"]
        for key in ("first_gradients", "first_state", "final_state"):
            assert sha(c[key]["path"]) == c[key]["sha256"]
            files.append(Path(c[key]["path"]))
    write_json(ROOT / "audit_protocol.json", dict(files={f.as_posix(): sha(f) for f in files}))
    old.torch.use_deterministic_algorithms(False)
    old.torch.backends.cudnn.benchmark = False
    old.torch.backends.cudnn.deterministic = False
    numerical = []
    scores = []
    bounds = [boundary()]
    for row in cases:
        c = row["measurement"]
        numerical.append(dict(index=row["index"], check=old.numerical(c, p)))
        scores.append(dict(index=row["index"], check=old.native_score_and_batches(c, p)))
        bounds.append(boundary())
        write_json(
            ROOT / "audit_progress.json",
            dict(numerical=numerical, scores=scores, boundaries=bounds),
        )
        print("audited", row["index"], flush=True)
    gradients = []
    for f in p["fixtures"]:
        peers = {(r["arm"], r["repeat"]): r["measurement"] for r in cases if r["fixture"] == f}
        for repeat in (0, 1):
            for arm in ("buffer4", "fp16"):
                gradients.append(
                    dict(
                        dataset=f["dataset"],
                        seed=f["seed"],
                        arm=arm,
                        repeat=repeat,
                        check=gradient(
                            peers[arm, repeat], peers["ordinary", repeat], arm == "fp16"
                        ),
                    )
                )
        gradients.append(
            dict(
                dataset=f["dataset"],
                seed=f["seed"],
                arm="native_repeat",
                repeat=1,
                check=gradient(peers["ordinary", 1], peers["ordinary", 0], False),
            )
        )
    hashes(read(ROOT / "audit_protocol.json")["files"])
    passed = all(r["check"]["passed"] for r in numerical + scores + gradients)
    write_json(
        ROOT / "audit.json",
        dict(
            status="EVIDENCE_VERIFIED",
            passed=passed,
            numerical=numerical,
            scores=scores,
            gradients=gradients,
            boundaries=bounds,
            native_scores=36,
            batches=1080,
            training_updates=0,
            backwards=0,
        ),
    )
    print("Independent audit", passed)


if __name__ == "__main__":
    main()
