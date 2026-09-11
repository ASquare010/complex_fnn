"""Verify H117 evidence and freeze a four-run conditional diagnosis."""

import copy
import json
import shutil
from pathlib import Path

from results.verification.fp32_training_replication_final_v1 import hashes, read, sha

ROOT = Path("results/training_variability_v1")
PREVIOUS = Path("results/fp32_training_replication_recovery_v1")
RECEIPT = Path("results/verification/fp32_training_replication_final_v1.json")


def run() -> None:
    assert not (ROOT / "protocol.json").exists()
    previous, receipt = read(PREVIOUS / "protocol.json"), read(RECEIPT)
    assert receipt["status"] == "PASS" and not receipt["broad_goal_achieved"]
    for field in ("files", "tensor_hashes", "local_metric_hashes"):
        hashes(receipt[field])
    for path, info in receipt["packed"].items():
        assert sha(path) == info["sha256"]
    hashes(previous["sources"])
    hashes(previous["maintained_files"])
    assert len(receipt["tensor_hashes"]) == 78
    result = read(PREVIOUS / "result.json")
    base = next(f for f in previous["fixtures"] if f["label"] == "wikitext2_s101")
    policies = ["fp32_default_native", "fp32_default_chunks"]
    originals = [c for c in result["cases"] if c["fixture"] == base and c["policy"] in policies]
    assert len(originals) == 2 and all(c["optimizer_updates"] == 800 for c in originals)
    fixtures = []
    for repeat in (1, 2):
        fixture = copy.deepcopy(base)
        fixture.update(
            label=base["label"] + f"_r{repeat}",
            repetition=repeat,
            source_fixture_label=base["label"],
            policy_order=policies if repeat == 1 else policies[::-1],
        )
        fixtures.append(fixture)
    input_paths = [
        RECEIPT,
        *[
            PREVIOUS / n
            for n in (
                "protocol.json",
                "result.json",
                "audit.json",
                "qualification.json",
                "initializations.json",
            )
        ],
        Path(base["checkpoint"]),
    ]
    for case in originals:
        input_paths += [Path(case[k]["path"]) for k in ("initial_probe", "checkpoint")]
        input_paths += [Path(c["path"]) for c in case["intermediate_checkpoints"]]
    dataset = previous["datasets"]["wikitext2"]
    folder = Path(dataset["path"])
    assert sha(folder / "manifest.json") == dataset["manifest_sha256"]
    input_paths.append(folder / "manifest.json")
    for name, digest in dataset["manifest"]["files"].items():
        assert sha(folder / name) == digest
        input_paths.append(folder / name)
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
    sources = previous["sources"].copy()
    additions = [
        *ROOT.glob("source/*.py"),
        Path("research/training_variability_plan.md"),
        Path("results/verification/fp32_training_replication_final_v1.py"),
    ]
    sources.update({p.as_posix(): sha(p) for p in additions})
    disk_free = shutil.disk_usage(ROOT).free
    assert disk_free > 5 * 2**30
    protocol = dict(
        study="H118",
        previous_goal_turn="progress",
        sources=sources,
        maintained_files=previous["maintained_files"],
        previous_receipt_sha256=sha(RECEIPT),
        before_documents=snapshots,
        input_hashes={p.as_posix(): sha(p) for p in input_paths},
        previous_tensor_artifacts_verified=78,
        previous_root=PREVIOUS.as_posix(),
        original_case_labels=[c["label"] for c in originals],
        original_fixture=base,
        fixtures=fixtures,
        policies=policies,
        checkpoint_hashes={base["checkpoint"]: sha(base["checkpoint"])},
        datasets={"wikitext2": dataset},
        seed=101,
        repetitions=[1, 2],
        cases=4,
        steps_per_case=800,
        optimizer_updates=3200,
        training_targets=13107200,
        profile_backward_passes=3204,
        new_qualification_backward_passes=0,
        planned_audit_backward_passes=4,
        planned_native_scores=13,
        total_planned_new_backward_passes=3208,
        selected_failure_diagnosis=True,
        repeated_h117_runs_retained=True,
        original_gate_requalified=False,
        git_head=previous["git_head"],
        disk_free_bytes_before=disk_free,
        broad_goal_achieved=False,
    )
    (ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            dict(
                frozen_sources=len(sources),
                prior_tensors_verified=78,
                new_trials=4,
                updates=3200,
                disk_free_gib=disk_free / 2**30,
            )
        )
    )


if __name__ == "__main__":
    run()
