"""Continue only the unexecuted cases using the unchanged native-step function."""

import time

from results.adam_update_sensitivity_v1.source.study import (
    ROOT,
    clear_boundary,
    environment,
    hashes,
    read,
    run_case,
    write_json,
)


def run():
    assert not (ROOT / "result.json").exists()
    p, recovery = read(ROOT / "protocol.json"), read(ROOT / "recovery_protocol.json")
    hashes(recovery["files"])
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])
    beginning = time.perf_counter()
    write_json(ROOT / "environment.json", environment())
    cases, boundaries = recovery["cpu_cases"].copy(), [clear_boundary()]
    for case in p["cases"]:
        cases.append(run_case(case, "cuda", p))
        boundaries.append(clear_boundary())
    assert len(cases) == 12 and len(boundaries) == 7
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=cases,
            boundaries=boundaries,
            recovery_wall_seconds=time.perf_counter() - beginning,
            original_stage_wall_seconds=None,
            completed_original_steps=6,
            recovery_steps=6,
            disposable_optimizer_steps=12,
            language_training_updates=0,
            training_targets=0,
            forwards=0,
            backwards=0,
            validation_scores=0,
            original_failure_preserved=True,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
