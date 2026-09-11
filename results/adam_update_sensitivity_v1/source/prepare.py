"""Freeze H119 before loading gradient values or taking disposable steps."""

import json
import shutil
from pathlib import Path

from results.verification.training_variability_final_v1 import hashes, read, sha

ROOT = Path("results/adam_update_sensitivity_v1")
PREVIOUS = Path("results/training_variability_v1")
RECEIPT = Path("results/verification/training_variability_final_v1.json")


def run():
    assert not (ROOT / "protocol.json").exists()
    previous, receipt = read(PREVIOUS / "protocol.json"), read(RECEIPT)
    assert receipt["status"] == "PASS" and not receipt["broad_goal_achieved"]
    for field in ("files", "tensor_hashes", "local_metric_hashes"):
        hashes(receipt[field])
    for path, info in receipt["packed"].items():
        assert sha(path) == info["sha256"]
    hashes(previous["sources"])
    hashes(previous["maintained_files"])
    old_root = Path(previous["previous_root"])
    original = read(old_root / "result.json")
    cases = [c for c in original["cases"] if c["label"] in previous["original_case_labels"]]
    cases += read(PREVIOUS / "result.json")["cases"]
    assert len(cases) == 6 and len({c["initial_model_hash"] for c in cases}) == 1
    inputs = [RECEIPT, old_root / "result.json", Path(previous["original_fixture"]["checkpoint"])]
    inputs += [
        PREVIOUS / name for name in ("protocol.json", "result.json", "audit.json", "summary.json")
    ]
    inputs += [Path(c["initial_probe"]["path"]) for c in cases]
    library = [
        Path(".venv/Lib/site-packages/torch") / name
        for name in ("optim/adam.py", "optim/adamw.py", "nn/utils/clip_grad.py")
    ]
    sources = previous["sources"].copy()
    sources.update(
        {
            p.as_posix(): sha(p)
            for p in [
                *ROOT.glob("source/*.py"),
                Path("research/adam_update_sensitivity_plan.md"),
                Path("results/verification/training_variability_final_v1.py"),
            ]
        }
    )
    snapshots = {}
    (ROOT / "before_documents").mkdir(exist_ok=False)
    (ROOT / "runs").mkdir(exist_ok=False)
    for name in previous["before_documents"]:
        path = Path(name)
        target = (
            ROOT
            / "before_documents"
            / ("gitignore" if name == ".gitignore" else name.replace("/", "__"))
        )
        target.write_bytes(path.read_bytes())
        snapshots[name] = dict(sha256=sha(path), preserved_path=target.as_posix())
    assert shutil.disk_usage(ROOT).free > 3 * 2**30
    protocol = dict(
        study="H119",
        sources=sources,
        maintained_files=previous["maintained_files"],
        input_hashes={p.as_posix(): sha(p) for p in inputs},
        library_hashes={p.as_posix(): sha(p) for p in library},
        before_documents=snapshots,
        previous_receipt=RECEIPT.as_posix(),
        initial_path=previous["original_fixture"]["checkpoint"],
        initial_model_hash=cases[0]["initial_model_hash"],
        seed=101,
        cases=[dict(label=c["label"], policy=c["policy"], probe=c["initial_probe"]) for c in cases],
        devices=["cpu", "cuda"],
        epsilons=[1e-8, 1e-7, 1e-6],
        native_epsilon=1e-8,
        amplification_min=10,
        near_zero_share_min=0.5,
        conditioning_ratio_max=0.5,
        distortion_max=0.01,
        parameter_relative_tolerance=1e-6,
        parameter_absolute_tolerance=2e-7,
        clip_relative_tolerance=1e-6,
        metric_rtol=1e-8,
        metric_atol=1e-12,
        disposable_optimizer_steps=12,
        language_training_updates=0,
        forwards=0,
        backwards=0,
        training_targets=0,
        validation_scores=0,
        planned_artifacts=24,
        planned_audit_optimizer_steps=0,
        selected_failure_diagnosis=True,
        original_gate_requalified=False,
        broad_goal_achieved=False,
    )
    (ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            dict(
                frozen_sources=len(sources),
                maintained=len(previous["maintained_files"]),
                inputs=len(inputs),
                steps=12,
            )
        )
    )


if __name__ == "__main__":
    run()
