"""Final H076 audit: language evidence, execution records and prior-state preservation."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/ungated_lm_screen_v1")
OUT = Path("results/verification/ungated_lm_screen_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


before, p, r = [read(ROOT / n) for n in ("before.json", "protocol.json", "result.json")]
a = read("results/verification/ungated_lm_screen_analysis_v1.json")
assert r["status"] == "complete" and a["status"] == "PASS"
assert a["completed_lm_trials"] == 21 and a["all_21_rescores_exact"]
assert a["all_checkpoints_moments_finite"] and a["all_initial_states_reconstructed_exactly"]
assert (
    a["all_step_logs_consistent"] and a["all_sampler_states_exact"] and a["validation_order_exact"]
)
assert a["rescore_validation_targets"] == 6776448 and a["rescore_optimizer_updates"] == 0
assert (
    a["selected_cells"] == r["selected_cells"] and a["independently_rederived_gates"] == r["gates"]
)
assert a["earns_longer_comparison"] == r["earns_longer_comparison"]
assert r["language_trials"] == 21 and r["language_optimizer_updates"] == 4200
assert (
    r["language_training_targets"] == 8601600
    and r["shared_validation_target_exposures"] == 47435136
)
assert r["qualification_cpu_updates"] == 8 and r["qualification_cpu_training_targets"] == 256
assert r["scientific_attempts"] == 1 and r["scientific_retries"] == 0
assert not r["official_test_scored"] and not r["research_goal_achieved"]
assert len(p["sources"]) == 106 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/ungated_lm_screen_plan.md") == p["plan_sha256"] == before["plan_sha256"]
assert all(sha(Path("data/wikitext2_v1") / n) == h for n, h in p["data_files"].items())
assert sha("data/wikitext2_v1/manifest.json") == p["data_manifest_sha256"]
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "runs" / cell / n) == h for n, h in files.items())
assert len(list((ROOT / "runs").glob("*/metrics.json"))) == 21
for label, h in a["process_hashes"].items():
    path = ROOT / "processes" / (label + ".json")
    assert sha(path) == h and sha(path.with_suffix(".log")) == read(path)["log_sha256"]
assert len(list((ROOT / "processes").glob("*.json"))) == 23
assert "5 passed" in (ROOT / "processes/harness.log").read_text(encoding="utf-8")
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
for phase in ("analysis", "report"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert event["scientific_source_unchanged"] and event["scientific_reruns"] == 0
    assert sha(f"results/verification/ungated_lm_screen_{phase}_v1.py") == event["source_sha256"]
    assert sha(ROOT / (phase + ".log")) == event["log_sha256"]
    assert sha(ROOT / "result.json") == event["scientific_result_sha256"]
    assert sha(ROOT / "processes/finish.json") == event["scientific_finish_process_sha256"]
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
    assert (
        hashlib.sha256(z.read("research/ungated_lm_screen_plan.md")).hexdigest() == p["plan_sha256"]
    )
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
priorpath = Path("results/verification/ungated_resource_recovery_final_v1.json")
assert sha(priorpath) == before["h075_final_sha256"] == p["h075_final_sha256"]
prior = read(priorpath)
oldroot = Path("results/ungated_resource_recovery_v1")
for name, key in (
    ("protocol.json", "protocol_sha256"),
    ("result.json", "result_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
):
    assert sha(oldroot / name) == prior[key]
assert (
    sha("results/verification/ungated_resource_recovery_analysis_v1.json")
    == prior["analysis_sha256"]
)
oldanalysis = read("results/verification/ungated_resource_recovery_analysis_v1.json")
oldprotocol = read(oldroot / "protocol.json")
for cell, files in oldanalysis["worker_artifact_hashes"].items():
    assert all(
        sha(Path(oldprotocol["origins"][cell]) / "workers" / cell / n) == h
        for n, h in files.items()
    )
assert len(oldanalysis["worker_artifact_hashes"]) == 21
assert (
    sha("results/verification/ungated_resource_final_v1.json") == prior["prior_h074_final_sha256"]
)
assert not list(Path("results/ungated_resource_v1/workers/gelu_same_block_inner").iterdir())
assert not Path("results/ungated_resource_v1/result.json").exists()
with zipfile.ZipFile(oldroot / "support.zip") as z:
    assert z.testzip() is None
    assert all(
        hashlib.sha256(z.read(n)).hexdigest() == h for n, h in prior["support_artifacts"].items()
    )
old_cells = 0
for label, folder in (
    ("ungated_fit", "ungated_fit_v1"),
    ("rotated_shuffle_fit", "rotated_shuffle_fit_v1"),
    ("token_activation_fit", "token_activation_fit_v2"),
):
    old_final = read(f"results/verification/{label}_final_v1.json")
    analysis_path = f"results/verification/{label}_analysis_v1.json"
    assert sha(analysis_path) == old_final["analysis_sha256"]
    old = read(analysis_path)
    for cell, files in old["cell_artifact_hashes"].items():
        assert all(
            sha(Path("results") / folder / "cells" / cell / n) == h for n, h in files.items()
        )
    old_cells += len(old["cell_artifact_hashes"])
assert old_cells == 798
manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 65 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
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
archived = read("research/archive/h057_manifest.json")
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
            intended in archived["files"] and (Path("research/archive/retired") / intended).exists()
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
        *map(str, Path("results/verification").glob("ungated_lm_screen*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout + lint.stderr
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/ungated_lm_screen_plan.md"),
    Path("research/ungated_lm_screen_results.md"),
    *Path("research/figures").glob("ungated_lm_screen.*"),
    *Path("results/verification").glob("ungated_lm_screen*.py"),
    Path("results/verification/ungated_lm_screen_analysis_v1.json"),
    ROOT / "result.json",
    ROOT / "coordinator_status.json",
    ROOT / "analysis_process.json",
    ROOT / "report_process.json",
    *[n for n in (ROOT / "report_mplconfig").rglob("*") if n.is_file()],
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
recorded = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_status": "COMPLETE_LANGUAGE_SCREEN",
    "research_goal_achieved": False,
    "new_lm_trials": 21,
    "language_optimizer_updates": 4200,
    "language_training_targets": 8601600,
    "shared_validation_target_exposures": 47435136,
    "qualification_cpu_updates": 8,
    "qualification_cpu_training_targets": 256,
    "independent_rescored_checkpoints": 21,
    "all_rescored_nlls_exact": True,
    "independent_validation_targets": 6776448,
    "all_checkpoints_and_moments_finite": True,
    "sampler_and_validation_order_verified": True,
    "scientific_attempts": 1,
    "scientific_retries": 0,
    "official_test_scored": False,
    "earns_longer_comparison": r["earns_longer_comparison"],
    "prior_h075_final_sha256": sha(priorpath),
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "prior_lm_profile_runs_preserved": 171,
    "isolated_new_lm_runs": 21,
    "prior_active_suite_tests": 108,
    "prior_active_source_tests_configurations_unchanged": True,
    "prior_resource_checkpoint_pairs_preserved": 21,
    "prior_fitting_checkpoint_artifacts_preserved": 798,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "source_hashes": p["sources"],
    "frozen_plans": plans,
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "result_sha256": sha(ROOT / "result.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/ungated_lm_screen_analysis_v1.json"),
    "support_artifacts": {n.as_posix(): sha(n) for n in support},
    "figure_visually_checked": True,
    "local_links_verified": links,
    "historical_archive_links": historical,
    "lint": lint.stdout,
}
OUT.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in recorded.items()
            if k not in ("source_hashes", "frozen_plans", "support_artifacts")
        }
    ),
    flush=True,
)
