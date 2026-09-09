"""Preserve the complete H075 recovery and all prior scientific evidence."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/ungated_resource_recovery_v1")
OLD = Path("results/ungated_resource_v1")
OUT = Path("results/verification/ungated_resource_recovery_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


before, p, r = [read(ROOT / n) for n in ("before.json", "protocol.json", "result.json")]
a = read("results/verification/ungated_resource_recovery_analysis_v1.json")
assert r["status"] == "complete" and r["recovery_completed"] and a["status"] == "PASS"
assert r["all_comparisons_exact"] == a["all_comparisons_exact"]
assert (
    r["gates"] == a["independently_rederived_gates"] and r["selected_modes"] == a["selected_modes"]
)
assert r["earns_language_screen"] == a["earns_language_screen"]
assert len(r["rows"]) == 21 and len(r["comparisons"]) == 14
assert a["checkpoint_pairs_inspected"] == 21 and a["direct_cpu_fidelity_comparisons"] == 14
assert r["new_optimizer_updates"] == 80 and r["combined_optimizer_updates"] == 420
assert (
    r["new_synthetic_training_targets"] == 163840
    and r["combined_synthetic_training_targets"] == 860160
)
assert r["new_initial_probe_targets"] == 8192 and r["combined_initial_probe_targets"] == 43008
assert r["construction_probes"] == 3 and r["construction_probe_updates"] == r["corpus_targets"] == 0
assert r["combined_worker_launches"] == 22 and r["explicit_preupdate_failed_cell_retries"] == 1
assert r["completed_cell_reruns"] == 0 and not r["research_goal_achieved"]
assert len(p["sources"]) == 103 and all(sha(n) == h for n, h in p["sources"].items())
assert (
    sha("research/ungated_resource_recovery_plan.md")
    == before["plan_sha256"]
    == p["recovery_plan_sha256"]
)
assert sha("research/ungated_resource_plan.md") == p["plan_sha256"]
assert sha(OLD / "tokens.pt") == p["token_file_sha256"]
for cell, files in a["worker_artifact_hashes"].items():
    folder = Path(p["origins"][cell]) / "workers" / cell
    assert all(sha(folder / n) == h for n, h in files.items())
    origin = r["origins"][cell]
    assert (
        origin["root"] == p["origins"][cell]
        and sha(folder / "result.json") == origin["result_sha256"]
    )
    assert sha(folder / "checkpoint.pt") == origin["checkpoint_sha256"]
for label, h in a["process_hashes"].items():
    origin = OLD if label == "harness" else Path(p["origins"].get(label, ROOT.as_posix()))
    path = origin / "processes" / (label + ".json")
    assert sha(path) == h and sha(path.with_suffix(".log")) == read(path)["log_sha256"]
assert len(list((ROOT / "processes").glob("*.json"))) == 8
assert len(list((ROOT / "workers").iterdir())) == 4
assert not list((OLD / "workers/gelu_same_block_inner").iterdir())
assert not (OLD / "result.json").exists()
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
for name in ("cpu_small", "full_1", "full_2"):
    probe = read(ROOT / "probes" / (name + ".json"))
    assert probe["status"] == "PASS" and all(
        probe[k] == 0 for k in ("forwards", "backwards", "optimizer_updates", "corpus_targets")
    )
    assert not probe["cause_identified"]
for phase in ("analysis", "report"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert event["scientific_source_unchanged"] and event["scientific_reruns"] == 0
    assert (
        sha(f"results/verification/ungated_resource_recovery_{phase}_v1.py")
        == event["source_sha256"]
    )
    assert sha(ROOT / (phase + ".log")) == event["log_sha256"]
    assert sha(ROOT / "result.json") == event["scientific_result_sha256"]
    assert sha(ROOT / "processes/finish.json") == event["scientific_finish_process_sha256"]
creation = read(ROOT / "report_creation_failure.json")
assert (
    not creation["generator_created"]
    and not creation["report_coordinator_launched"]
    and creation["scientific_reruns"] == 0
)
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
    for name, key in (
        ("research/ungated_resource_plan.md", "plan_sha256"),
        ("research/ungated_resource_recovery_plan.md", "recovery_plan_sha256"),
    ):
        assert hashlib.sha256(z.read(name)).hexdigest() == p[key]
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
priorpath = Path("results/verification/ungated_resource_final_v1.json")
assert sha(priorpath) == before["h074_final_sha256"] == p["h074_final_sha256"]
prior = read(priorpath)
for name, key in (
    ("protocol.json", "protocol_sha256"),
    ("failure.json", "failure_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
):
    assert sha(OLD / name) == prior[key]
assert (
    sha("results/verification/ungated_resource_partial_analysis_v1.json")
    == prior["analysis_sha256"]
)
assert sha(OLD / "failure.json") == before["h074_failure_sha256"] == p["h074_failure_sha256"]
for cell, files in p["preserved_h074_worker_hashes"].items():
    assert all(sha(OLD / "workers" / cell / n) == h for n, h in files.items())
assert len(p["preserved_h074_worker_hashes"]) == 17
with zipfile.ZipFile(OLD / "support.zip") as z:
    assert z.testzip() is None
    assert all(
        hashlib.sha256(z.read(n)).hexdigest() == h for n, h in prior["support_artifacts"].items()
    )
assert sha("results/verification/ungated_fit_final_v1.json") == prior["prior_h073_final_sha256"]
old_cells = 0
for label, folder in (
    ("ungated_fit", "ungated_fit_v1"),
    ("rotated_shuffle_fit", "rotated_shuffle_fit_v1"),
    ("token_activation_fit", "token_activation_fit_v2"),
):
    final = read(f"results/verification/{label}_final_v1.json")
    path = f"results/verification/{label}_analysis_v1.json"
    assert sha(path) == final["analysis_sha256"]
    old = read(path)
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
assert len(plans) == 64 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
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
        *map(str, Path("results/verification").glob("ungated_resource_recovery*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout + lint.stderr
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/ungated_resource_recovery_plan.md"),
    Path("research/ungated_resource_recovery_results.md"),
    *Path("research/figures").glob("ungated_resource_recovery.*"),
    *Path("results/verification").glob("ungated_resource_recovery*.py"),
    Path("results/verification/ungated_resource_recovery_analysis_v1.json"),
    ROOT / "result.json",
    ROOT / "coordinator_status.json",
    ROOT / "analysis_process.json",
    ROOT / "report_process.json",
    ROOT / "report_creation_failure.json",
    *(ROOT / "probes").glob("*.json"),
    *[n for n in (ROOT / "report_mplconfig").rglob("*") if n.is_file()],
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_status": "COMPLETE_RESOURCE_QUALIFICATION",
    "research_goal_achieved": False,
    "original_cells_preserved": 17,
    "recovered_cells": 4,
    "combined_completed_cells": 21,
    "direct_cpu_checkpoint_pairs": 21,
    "exact_execution_comparisons": 14,
    "construction_probes_passed": 3,
    "new_optimizer_updates": 80,
    "combined_optimizer_updates": 420,
    "new_synthetic_training_targets": 163840,
    "combined_synthetic_training_targets": 860160,
    "combined_initial_probe_targets": 43008,
    "corpus_targets": 0,
    "completed_cell_reruns": 0,
    "explicit_preupdate_failed_cell_retries": 1,
    "combined_worker_launches": 22,
    "earns_language_screen": r["earns_language_screen"],
    "original_failure_preserved": True,
    "failure_cause_identified": False,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "prior_active_suite_tests": 108,
    "prior_active_sources_tests_configurations_unchanged": True,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": 798,
    "source_hashes": p["sources"],
    "frozen_plans": plans,
    "prior_h074_final_sha256": sha(priorpath),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "result_sha256": sha(ROOT / "result.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/ungated_resource_recovery_analysis_v1.json"),
    "support_artifacts": {n.as_posix(): sha(n) for n in support},
    "figure_visually_checked": True,
    "local_links_verified": links,
    "historical_archive_links": historical,
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
    ),
    flush=True,
)
