"""Freeze H121 source, prior evidence, checkpoints and installed hook implementation."""

import copy
import json
import shutil
from pathlib import Path

from results.verification.adam_update_sensitivity_final_v1 import hashes, read, sha

ROOT = Path("results/checkpoint_input_offload_v1")
PREVIOUS = Path("results/optimizer_memory_v1")
RECEIPT = Path("results/verification/optimizer_memory_final_v1.json")


def run():
    assert not (ROOT / "protocol.json").exists()
    prior, receipt = read(PREVIOUS / "protocol.json"), read(RECEIPT)
    assert receipt["status"] == "PASS" and receipt["all_numerical_audits_passed"]
    for field in ("files", "tensor_hashes", "local_metric_hashes"):
        hashes(receipt[field])
    for path, info in receipt["packed"].items():
        assert sha(path) == info["sha256"]
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(prior[field])
    fixtures = copy.deepcopy(prior["fixtures"])
    for i, fixture in enumerate(fixtures):
        fixture["mode_order"] = ["native", "offload"] if i % 2 == 0 else ["offload", "native"]
    (ROOT / "runs").mkdir(exist_ok=False)
    (ROOT / "before_documents").mkdir(exist_ok=False)
    before = {}
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
                Path("research/checkpoint_input_offload_plan.md"),
                Path("results/verification/optimizer_memory_final_v1.py"),
            ]
        }
    )
    inputs = prior["input_hashes"].copy()
    inputs.update(
        {
            f.as_posix(): sha(f)
            for f in [RECEIPT, PREVIOUS / "protocol.json", PREVIOUS / "result.json"]
        }
    )
    library = prior["library_hashes"].copy()
    library.update(
        {
            (".venv/Lib/site-packages/torch/" + f): sha(".venv/Lib/site-packages/torch/" + f)
            for f in (
                "autograd/graph.py",
                "utils/checkpoint.py",
                "cuda/memory.py",
                "distributed/algorithms/_checkpoint/checkpoint_wrapper.py",
            )
        }
    )
    assert shutil.disk_usage(ROOT).free > 4 * 2**30
    p = dict(
        study="H121",
        previous_goal_turn="progress",
        previous_receipt=RECEIPT.as_posix(),
        sources=sources,
        maintained_files=prior["maintained_files"],
        input_hashes=inputs,
        library_hashes=library,
        before_documents=before,
        fixtures=fixtures,
        datasets=prior["datasets"],
        repetitions=30,
        warmup=10,
        cases=8,
        qualification_backwards=6,
        probe_backwards=240,
        trace_backwards=8,
        replay_backwards=8,
        total_backwards=262,
        training_updates=0,
        diagnostic_targets=1048576,
        validation_scores=0,
        tensor_artifacts=24,
        memory_ratio_max=0.9,
        time_ratio_max=1.15,
        timing_stability_max=1.15,
        loss_tolerance=1e-6,
        gradient_global_tolerance=1e-5,
        gradient_tensor_tolerance=1e-4,
        boundary_payload_bytes=48 * 2**20,
        host_allocated_peak_max=128 * 2**20,
        broad_goal_achieved=False,
        maintained_defaults_changed=False,
    )
    (ROOT / "protocol.json").write_text(
        json.dumps(p, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(dict(sources=len(sources), cases=8, backwards=262, training_updates=0)))


if __name__ == "__main__":
    run()
