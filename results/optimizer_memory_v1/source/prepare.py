"""Freeze four fixtures and three native implementation modes before profiling."""

import copy
import json
import shutil
from pathlib import Path

from results.verification.adam_update_sensitivity_final_v1 import hashes, read, sha

ROOT = Path("results/optimizer_memory_v1")
PREVIOUS = Path("results/adam_update_sensitivity_v1")
RECEIPT = Path("results/verification/adam_update_sensitivity_final_v1.json")


def run():
    assert not (ROOT / "protocol.json").exists()
    prior, receipt = read(PREVIOUS / "protocol.json"), read(RECEIPT)
    for field in ("files", "tensor_hashes", "local_metric_hashes"):
        hashes(receipt[field])
    for path, info in receipt["packed"].items():
        assert sha(path) == info["sha256"]
    assert receipt["status"] == "PASS" and not receipt["scientific_numerical_audit_passed"]
    hashes(prior["sources"])
    hashes(prior["maintained_files"])
    training_root = Path("results/fp32_training_replication_recovery_v1")
    tp, tr = read(training_root / "protocol.json"), read(training_root / "result.json")
    chosen = [
        c
        for c in tr["cases"]
        if c["fixture"]["seed"] == 101 and c["policy"] == "fp32_default_chunks"
    ]
    assert len(chosen) == 2
    modes = {
        "default": dict(foreach=None, fused=None),
        "single": dict(foreach=False, fused=False),
        "fused": dict(foreach=None, fused=True),
    }
    fixtures, inputs = (
        [],
        [
            RECEIPT,
            PREVIOUS / "protocol.json",
            PREVIOUS / "recovery_protocol.json",
            training_root / "protocol.json",
            training_root / "result.json",
        ],
    )
    for case in chosen:
        for loss in ("fp32_default_native", "fp32_default_chunks"):
            f = copy.deepcopy(case["fixture"])
            f.update(
                label=f["dataset"] + "__" + loss,
                loss_policy=loss,
                source_step=800,
                checkpoint=case["checkpoint"]["path"],
                source_sha256=case["checkpoint"]["sha256"],
            )
            order = list(modes)
            shift = len(fixtures) % 3
            f["mode_order"] = order[shift:] + order[:shift]
            fixtures.append(f)
        inputs.append(Path(case["checkpoint"]["path"]))
    datasets = tp["datasets"]
    for dataset in datasets.values():
        folder = Path(dataset["path"])
        inputs += [
            folder / "manifest.json",
            *[folder / name for name in dataset["manifest"]["files"]],
        ]
        assert sha(folder / "manifest.json") == dataset["manifest_sha256"]
        hashes(
            {
                (folder / name).as_posix(): digest
                for name, digest in dataset["manifest"]["files"].items()
            }
        )
    before = {}
    (ROOT / "before_documents").mkdir(exist_ok=False)
    (ROOT / "runs").mkdir(exist_ok=False)
    for name in prior["before_documents"]:
        target = (
            ROOT
            / "before_documents"
            / ("gitignore" if name == ".gitignore" else name.replace("/", "__"))
        )
        target.write_bytes(Path(name).read_bytes())
        before[name] = dict(sha256=sha(name), preserved_path=target.as_posix())
    sources = prior["sources"].copy()
    sources.update(
        {
            f.as_posix(): sha(f)
            for f in [
                *ROOT.glob("source/*.py"),
                Path("research/optimizer_memory_plan.md"),
                Path("results/verification/adam_update_sensitivity_final_v1.py"),
            ]
        }
    )
    library = [
        Path(".venv/Lib/site-packages/torch") / f
        for f in ("optim/adam.py", "optim/adamw.py", "optim/optimizer.py", "nn/utils/clip_grad.py")
    ]
    assert shutil.disk_usage(ROOT).free > 6 * 2**30
    p = dict(
        study="H120",
        previous_goal_turn="progress",
        sources=sources,
        maintained_files=prior["maintained_files"],
        before_documents=before,
        previous_receipt=RECEIPT.as_posix(),
        input_hashes={f.as_posix(): sha(f) for f in inputs},
        library_hashes={f.as_posix(): sha(f) for f in library},
        fixtures=fixtures,
        datasets=datasets,
        modes=modes,
        steps=30,
        warmup=10,
        cases=12,
        training_updates=360,
        training_targets=1474560,
        training_backwards=360,
        audit_backwards=0,
        study_scores=24,
        audit_scores=12,
        tensor_artifacts=36,
        memory_ratio_max=0.9,
        time_ratio_max=1.10,
        timing_stability_max=1.15,
        nll_ratio_max=1.01,
        score_tolerance=1e-6,
        parameter_relative_tolerance=1e-6,
        parameter_absolute_tolerance=2e-7,
        moment_tolerance=1e-6,
        clip_tolerance=1e-6,
        gradient_global_tolerance=1e-5,
        gradient_tensor_tolerance=1e-4,
        broad_goal_achieved=False,
        original_quality_gate_requalified=False,
    )
    (ROOT / "protocol.json").write_text(
        json.dumps(p, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(dict(sources=len(sources), fixtures=4, cases=12, training_updates=360)))


if __name__ == "__main__":
    run()
