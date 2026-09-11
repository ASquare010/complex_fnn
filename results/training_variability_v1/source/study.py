"""Four conditional repeat executions through H117's unchanged common runner."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from results.fp32_classifier_profile_v1.source.common import clear_boundary  # noqa: E402
from results.fp32_training_replication_recovery_v1.source import study as shared  # noqa: E402
from src.core.reproducibility import environment, sha256, write_json  # noqa: E402

ROOT = Path("results/training_variability_v1")


def run() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for field in ("sources", "input_hashes"):
        for name, digest in protocol[field].items():
            assert sha256(Path(name)) == digest, name
    env = environment()
    env.update(
        tf32=False, threads=4, deterministic_algorithms=torch.are_deterministic_algorithms_enabled()
    )
    write_json(ROOT / "environment.json", env)
    cases, boundaries = [], [clear_boundary()]
    previous_root = shared.ROOT
    beginning = time.perf_counter()
    try:
        # Output routing is the only change to the imported runner's module state.
        shared.ROOT = ROOT
        for fixture in protocol["fixtures"]:
            for policy in fixture["policy_order"]:
                cases.append(shared.run_case(fixture, policy, protocol))
                boundaries.append(clear_boundary())
                write_json(
                    ROOT / "progress.json",
                    dict(completed=len(cases), optimizer_updates=800 * len(cases)),
                )
    finally:
        shared.ROOT = previous_root
    assert len(cases) == 4 and shared.ROOT == previous_root
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=cases,
            boundaries=boundaries,
            elapsed_seconds=time.perf_counter() - beginning,
            optimizer_updates=3200,
            training_targets=13107200,
            profile_backward_passes=3204,
            new_qualification_backward_passes=0,
            reused_qualification="results/fp32_training_replication_recovery_v1/qualification.json",
            fresh_executions=4,
            distinct_training_seeds=1,
            selected_failure_diagnosis=True,
            output_root_restored=True,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "failure.json", dict(traceback=traceback.format_exc()))
        raise
