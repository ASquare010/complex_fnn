"""Freeze the completed CPU work and a no-repeat CUDA continuation."""

import json
from pathlib import Path

from results.adam_update_sensitivity_v1.source.prepare import ROOT, hashes, read, sha

assert not (ROOT / "recovery_protocol.json").exists()
p = read(ROOT / "protocol.json")
for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
    hashes(p[field])
assert (ROOT / "study_exit.txt").read_text().strip() == "1"
paths = [ROOT / n for n in ("protocol.json", "study.log", "study_exit.txt", "study_failure.json")]
assert "assert not torch.cuda.is_initialized()" in read(ROOT / "study_failure.json")["traceback"]
cases = []
for source in p["cases"]:
    folder = ROOT / "runs" / ("cpu__" + source["label"])
    case = read(folder / "result.json")
    assert case["disposable_optimizer_steps"] == 1 and case["state_steps_all_one"]
    cases.append(case)
    paths.append(folder / "result.json")
    for key in ("clipped", "updated"):
        assert sha(case[key]["path"]) == case[key]["sha256"]
        paths.append(Path(case[key]["path"]))
    assert not (ROOT / "runs" / ("cuda__" + source["label"])).exists()
sources = [
    ROOT / "source" / name
    for name in ("prepare_recovery.py", "recover.py", "prepare_audit_recovery.py")
]
sources.append(Path("research/adam_update_sensitivity_recovery_plan.md"))
sources.append(Path(".venv/Lib/site-packages/torch/optim/optimizer.py"))
(ROOT / "recovery_protocol.json").write_text(
    json.dumps(
        dict(
            files={f.as_posix(): sha(f) for f in paths + sources},
            cpu_cases=cases,
            completed_disposable_steps=6,
            remaining_disposable_steps=6,
            total_disposable_steps=12,
            forwards=0,
            backwards=0,
            language_training_updates=0,
            unchanged_numerical_protocol=True,
            original_failure_preserved=True,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Six completed CPU steps preserved; six CUDA steps allocated; no repeat.")
