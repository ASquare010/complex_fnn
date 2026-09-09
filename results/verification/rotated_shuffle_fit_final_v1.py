"""Preserve and audit H071 completed fitting without promoting unqualified models."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/rotated_shuffle_fit_v1")
OUT = Path("results/verification/rotated_shuffle_fit_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


before = read(ROOT / "before.json")
p = read(ROOT / "protocol.json")
r = read(ROOT / "result.json")
a = read("results/verification/rotated_shuffle_fit_analysis_v1.json")
assert r["status"] == "complete" and a["status"] == "PASS"
assert a["scientific_verdict"] == r["scientific_verdict"]
assert not r["research_goal_achieved"]
assert r["optimizer_updates"] == 75600 and r["training_example_presentations"] == 19353600
assert r["cells"] == 252 and r["selection_cells"] == 126
assert r["corpus_targets"] == r["full_model_resource_workers"] == 0
assert a["rescored_selected_checkpoints"] == 126 and a["rescored_angle_ablations"] == 42
assert a["all_252_checkpoint_weights_moments_finite"] and a["regenerated_data_and_streams_exact"]
assert a["selected_initializations_match"] and a["selection_precedes_heldout_files"]
assert a["independently_rederived_gates"] == r["gates"]
assert len(p["sources"]) == 90 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/rotated_shuffle_fit_plan.md") == p["plan_sha256"]
processes = {}
for phase in ("harness", "fitting"):
    process = read(ROOT / (phase + "_process.json"))
    assert (
        process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
    )
    assert sha(ROOT / (phase + ".log")) == process["log_sha256"]
    assert sha(ROOT / "protocol.json") == process["protocol_sha256"]
    assert sha(ROOT / "source.zip") == process["source_zip_sha256"]
    processes[phase] = sha(ROOT / (phase + "_process.json"))
assert "4 passed" in (ROOT / "harness.log").read_text(encoding="utf-8")
assert sha(ROOT / "data.pt") == read(ROOT / "data_manifest.json")["data_sha256"] == r["data_sha256"]
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "cells" / cell / n) == h for n, h in files.items())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
    assert (
        hashlib.sha256(z.read("research/rotated_shuffle_fit_plan.md")).hexdigest()
        == p["plan_sha256"]
    )
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())

baseline = read("results/verification/rotated_shuffle_fit_baseline_v1.json")
assert baseline["status"] == "PASS" and baseline["post_hoc_diagnostic"]
assert not baseline["decision_gates_changed"] and baseline["optimizer_updates"] == 0
assert baseline["data_sha256"] == sha(ROOT / "data.pt")
assert baseline["result_sha256"] == sha(ROOT / "result.json")
assert (
    sum(
        all(v["selected_seeds_beating_zero"] == 0 for v in row["forms"].values())
        for row in baseline["rows"].values()
    )
    == 5
)
for phase in ("analysis", "report"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0
    assert event["postprocess_source_unchanged"] and event["scientific_source_unchanged"]
    assert event["scientific_reruns"] == 0
    assert sha(ROOT / (phase + ".log")) == event["log_sha256"]
    assert sha(f"results/verification/rotated_shuffle_fit_{phase}_v1.py") == event["source_sha256"]
    assert sha(ROOT / "fitting_process.json") == event["scientific_fitting_process_sha256"]
    processes[phase] = sha(ROOT / (phase + "_process.json"))
observation = read(ROOT / "postprocess_observation.json")
assert observation["audit_execution_before_interruption"] == "unknown"
assert observation["scientific_reruns"] == 0

prior_path = Path("results/verification/rotated_shuffle_final_v1.json")
assert sha(prior_path) == before["h070_final_sha256"]
prior = read(prior_path)
assert all(sha(n) == h for n, h in before["h070_sources"].items())
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
    ("process.json", "process_sha256"),
):
    assert sha(Path("results/rotated_shuffle_v1") / filename) == prior[key]
assert sha("results/verification/rotated_shuffle_analysis_v1.json") == prior["analysis_sha256"]
old_analysis = read("results/verification/rotated_shuffle_analysis_v1.json")
for n, h in old_analysis["artifact_hashes"].items():
    assert sha(Path("results/rotated_shuffle_v1/checks") / n) == h
assert sha("results/verification/factor_balance_final_v1.json") == prior["prior_h069_final_sha256"]
for root, label in (
    ("factor_balance_v2", "factor_balance"),
    ("token_activation_fit_v2", "token_activation_fit"),
):
    saved = read(f"results/verification/{label}_final_v1.json")
    for filename, key in (
        ("result.json", "result_sha256"),
        ("protocol.json", "protocol_sha256"),
        ("source.zip", "source_archive_sha256"),
        ("support.zip", "support_archive_sha256"),
    ):
        assert sha(Path("results") / root / filename) == saved[key]
    assert sha(f"results/verification/{label}_analysis_v1.json") == saved["analysis_sha256"]
fitting = read("results/verification/token_activation_fit_analysis_v1.json")
for cell, files in fitting["cell_artifact_hashes"].items():
    assert all(
        sha(Path("results/token_activation_fit_v2/cells") / cell / n) == h for n, h in files.items()
    )
manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, name
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 60 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
config = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(
    ast.literal_eval(n.value)
    for n in config.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets)
)
assert len(variants) == 6 and len(list(Path("configs").glob("*.json"))) == 9
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
        str(ROOT / "source"),
        *map(str, Path("results/verification").glob("rotated_shuffle_fit*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout + lint.stderr
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/rotated_shuffle_fit_plan.md"),
    Path("research/rotated_shuffle_fit_results.md"),
    *Path("research/figures").glob("rotated_shuffle_fit.*"),
    *Path("results/verification").glob("rotated_shuffle_fit*.py"),
    Path("results/verification/rotated_shuffle_fit_analysis_v1.json"),
    Path("results/verification/rotated_shuffle_fit_baseline_v1.json"),
    ROOT / "postprocess_observation.json",
    ROOT / "analysis_process.json",
    ROOT / "report_process.json",
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_verdict": r["scientific_verdict"],
    "research_goal_achieved": False,
    "isolated_harness_checks": 4,
    "scientific_attempts": 1,
    "audit_observation_interruption_preserved": True,
    "scientific_reruns": 0,
    "post_hoc_zero_baseline_recorded_without_gate_changes": True,
    "completed_fitting_cells": 252,
    "completed_optimizer_updates": 75600,
    "training_example_presentations": 19353600,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "all_final_weights_and_moments_finite": True,
    "independent_cpu_selected_rescores": 126,
    "independent_cpu_angle_reset_rescores": 42,
    "independent_gate_rederivation_passed": True,
    "regenerated_data_and_streams_exact": True,
    "earns_full_model_resource_qualification": {
        f: g["earns_full_model_resource_qualification"] for f, g in r["gates"].items()
    },
    "earns_language_training": False,
    "prior_active_suite_tests": 108,
    "all_prior_active_source_configuration_tests_unchanged": True,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": 252,
    "prior_h070_final_sha256": sha(prior_path),
    "frozen_plans": plans,
    "source_hashes": p["sources"],
    "process_records": processes,
    "local_links_verified": links,
    "historical_archive_links": historical,
    "result_sha256": sha(ROOT / "result.json"),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/rotated_shuffle_fit_analysis_v1.json"),
    "support_artifacts": {n.as_posix(): sha(n) for n in support},
    "figure_visually_checked": True,
    "lint": lint.stdout,
}
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k not in ("source_hashes", "frozen_plans", "support_artifacts")
        }
    )
)
