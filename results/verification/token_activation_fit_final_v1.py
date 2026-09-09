"""H068 final preservation audit; a completed rejection is not the research goal."""

import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

from src.core.config import VARIANTS
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/token_activation_fit_v2")
OLD = Path("results/token_activation_fit_v1")
OUT = Path("results/verification/token_activation_fit_final_v1.json")
assert not OUT.exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return sha256(Path(p))


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


r, p, a = [
    read(n)
    for n in (
        ROOT / "result.json",
        ROOT / "protocol.json",
        Path("results/verification/token_activation_fit_analysis_v1.json"),
    )
]
process = read(ROOT / "process.json")
assert a["status"] == process["status"] == "PASS" and process["returncode"] == 0
assert process["source_unchanged"] and a["source_unchanged"]
assert r["scientific_verdict"] == a["scientific_verdict"] == "REJECTED_AT_THIS_FITTING_BUDGET"
assert (
    r["cells"] == 252
    and r["optimizer_updates"] == 75600
    and r["training_example_presentations"] == 19353600
)
assert (
    r["corpus_targets"] == r["full_model_resource_workers"] == 0 and not r["research_goal_achieved"]
)
assert all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/token_activation_fit_plan.md") == p["plan_sha256"]
assert sha("research/token_activation_fit_recovery_plan.md") == p["recovery_plan_sha256"]
assert sha(ROOT / "protocol.json") == process["protocol_sha256"] == r["protocol_sha256"]
assert sha(ROOT / "source.zip") == process["source_zip_sha256"]
assert sha(ROOT / "process.log") == process["log_sha256"]
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
assert read(ROOT / "data_reproduction.json")["tensor_data_and_streams_exact"]
assert sha(ROOT / "data.pt") == r["data_sha256"]
assert (
    len(a["cell_artifact_hashes"]) == 252
    and a["rescored_selected_checkpoints"] == 126
    and a["rescored_gate_ablations"] == 42
)
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "cells" / cell / n) == h for n, h in files.items())
assert a["max_cpu_gpu_relative_mse_error"] < 1e-5

rows = {(v["task"], v["form"], v["seed"]): v for v in r["selected_rows"]}
tasks = list(r["task_mean_mse"])
seeds = (17, 29, 43)


def ratio(form, ref, ts=tasks, ss=seeds):
    return gm(
        [rows[t, form, s]["heldout_mse"] / rows[t, ref, s]["heldout_mse"] for t in ts for s in ss]
    )


for form, count in (("static", 5473), ("dynamic", 5521)):
    expected = {
        "two_percent_over_plain": ratio(form, "plain") <= 0.98,
        "every_seed_better_than_plain": all(ratio(form, "plain", ss=(s,)) < 1 for s in seeds),
        "beats_calibrated_narrow": ratio(form, "narrow") < 1,
        "generic_regression_cap": all(
            ratio(form, "plain", ts=(t,)) <= 1.05 for t in tasks if not t.endswith("_teacher")
        ),
        "seventy_percent_reduction": count <= 0.3 * 18432,
        "finite_final_state": all(v["finite"] for v in r["selected_rows"] if v["form"] == form),
    }
    if form == "dynamic":
        expected["one_percent_over_static"] = ratio(form, "static") <= 0.99
    assert expected == r["gates"][form]["tests"]
    assert (
        not all(expected.values())
        and not r["gates"][form]["earns_full_model_resource_qualification"]
    )

interruption = read(OLD / "interruption.json")
assert interruption["status"] == "INCOMPLETE_WORKER_ABSENT" and interruption["returncode"] is None
assert (
    interruption["completed_endpoints"] == 0
    and interruption["unrecorded_optimizer_updates_upper_bound"] == 300
)
assert not list((OLD / "cells").rglob("*.*"))
assert sha(OLD / "fitting_process.json") == interruption["fitting_process_sha256"]
assert sha(OLD / "fitting.log") == interruption["fitting_log_sha256"]
assert sha(OLD / "interruption.json") == p["original_interruption_sha256"]
assert read(OLD / "harness_process.json")["status"] == "PASS"
assert "4 passed in 1.32s" in (OLD / "harness.log").read_text(encoding="utf-8")
assert sha(OLD / "harness.log") == read(OLD / "harness_process.json")["log_sha256"]
assert sha(OLD / "source.zip") == read(OLD / "harness_process.json")["source_zip_sha256"]
before = read(OLD / "before.json")
with zipfile.ZipFile(OLD / "before_docs.zip") as z:
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
prior = read("results/verification/token_activation_final_v1.json")
assert sha("results/verification/token_activation_final_v1.json") == before["h067_final_sha256"]
assert all(sha(n) == h for n, h in before["h067_sources"].items())
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
    ("process.json", "process_sha256"),
):
    assert sha(Path("results/token_activation_v1") / filename) == prior[key]
for n, h in prior["check_artifact_hashes"].items():
    assert sha(Path("results/token_activation_v1/checks") / n) == h
manifest = read("research/archive/h061_manifest.json")
for n, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(n).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 56 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
assert len(VARIANTS) == 6 and len(list(Path("configs").glob("*.json"))) == 9
assert len(list(Path("results/runs").glob("*/metrics.json"))) == 171
assert {n.name for n in Path("src").iterdir() if n.is_dir() and n.name.endswith("_ffn")} == {
    "dense_ffn",
    "blockshuffle_ffn",
    "rational_blockshuffle_ffn",
}

links, historical = 0, []
old_manifest = read("research/archive/h057_manifest.json")
for doc in [
    Path("README.md"),
    *Path("src").rglob("*.md"),
    *Path("doc").glob("*.md"),
    *Path("configs").glob("*.md"),
    *Path("research").rglob("*.md"),
]:
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        resolved = (doc.parent / clean).resolve()
        if resolved == OUT.resolve():
            continue
        if resolved.exists():
            links += 1
            continue
        assert doc.parent == Path("research") and doc.name.endswith("plan.md"), (doc, clean)
        intended = resolved.relative_to(Path.cwd()).as_posix()
        assert (
            intended in old_manifest["files"]
            and (Path("research/archive/retired") / intended).exists()
        )
        historical.append({"plan": doc.as_posix(), "target": clean})
assert len(historical) == 3
lint = subprocess.run(
    [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "src",
        "tests",
        str(OLD / "source"),
        str(ROOT / "source"),
        *map(str, Path("results/verification").glob("token_activation_fit*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/learnable_activation_domain.md"),
    Path("research/token_activation_results.md"),
    Path("research/token_activation_fit_plan.md"),
    Path("research/token_activation_fit_recovery_plan.md"),
    Path("research/token_activation_fit_results.md"),
    *Path("research/figures").glob("token_activation_fit.*"),
    *Path("results/verification").glob("token_activation_fit*.py"),
    Path("results/verification/token_activation_fit_analysis_v1.json"),
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_verdict": r["scientific_verdict"],
    "research_goal_achieved": False,
    "complete_fitting_cells": 252,
    "selected_checkpoints_independently_rescored": 126,
    "gate_ablations_independently_rescored": 42,
    "isolated_harness_checks_passed": 4,
    "prior_active_suite_tests": 108,
    "completed_optimizer_updates": 75600,
    "total_updates_including_interruption_range": [75600, 75900],
    "incomplete_attempts": 1,
    "documented_unchanged_recovery_attempts": 1,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "earns_resource_or_language_training": False,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "all_prior_source_configuration_tests_unchanged": True,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "frozen_plans": plans,
    "source_hashes": p["sources"],
    "local_links_verified": links,
    "historical_archive_links": historical,
    "result_sha256": sha(ROOT / "result.json"),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "process_sha256": sha(ROOT / "process.json"),
    "analysis_sha256": sha("results/verification/token_activation_fit_analysis_v1.json"),
    "documents": {n.as_posix(): sha(n) for n in support if n.suffix == ".md"},
    "figure_visually_checked": True,
    "lint": lint.stdout,
}
write_json(OUT, record)
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("frozen_plans", "source_hashes", "documents")}
    )
)
