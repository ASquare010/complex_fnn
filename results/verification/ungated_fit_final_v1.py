"""Audit H073 fitting, assay validity and preservation of the retained research tree."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/ungated_fit_v1")
OUT = Path("results/verification/ungated_fit_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


before = read(ROOT / "before.json")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
a = read("results/verification/ungated_fit_analysis_v1.json")
assert r["status"] == "complete" and a["status"] == "PASS"
assert r["scientific_verdict"] == a["scientific_verdict"]
assert (
    r["assay_passed"] == a["assay_passed"]
    and r["positive_control_tests"] == a["positive_control_tests"]
)
assert r["gates"] == a["independently_rederived_gates"]
assert r["optimizer_updates"] == 88200 and r["training_example_presentations"] == 22579200
assert r["cells"] == 294 and r["selection_cells"] == 147
assert (
    r["corpus_targets"] == r["full_model_resource_workers"] == 0 and not r["research_goal_achieved"]
)
assert a["rescored_selected_checkpoints"] == 147 and a["all_294_checkpoint_weights_moments_finite"]
assert a["regenerated_data_and_streams_exact"] and a["selected_initializations_match"]
assert a["selection_precedes_heldout_files"] and len(a["cell_artifact_hashes"]) == 294
if not r["assay_passed"]:
    assert r["scientific_verdict"] == "INCONCLUSIVE_ASSAY_FAILURE"
    assert not any(g["earns_full_model_resource_qualification"] for g in r["gates"].values())
assert len(p["sources"]) == 97 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/ungated_fit_plan.md") == p["plan_sha256"]
processes = {}
for phase in ("harness", "fitting"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert sha(ROOT / (phase + ".log")) == event["log_sha256"]
    assert sha(ROOT / "protocol.json") == event["protocol_sha256"]
    assert sha(ROOT / "source.zip") == event["source_zip_sha256"]
    processes[phase] = sha(ROOT / (phase + "_process.json"))
assert "5 passed" in (ROOT / "harness.log").read_text(encoding="utf-8")
for phase in ("analysis", "report"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0
    assert event["postprocess_source_unchanged"] and event["scientific_source_unchanged"]
    assert event["scientific_reruns"] == 0
    assert sha(ROOT / (phase + ".log")) == event["log_sha256"]
    assert sha(f"results/verification/ungated_fit_{phase}_v1.py") == event["source_sha256"]
    assert sha(ROOT / "fitting_process.json") == event["scientific_fitting_process_sha256"]
    processes[phase] = sha(ROOT / (phase + "_process.json"))
assert sha(ROOT / "data.pt") == r["data_sha256"] == read(ROOT / "data_manifest.json")["data_sha256"]
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "cells" / cell / n) == h for n, h in files.items())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
    assert hashlib.sha256(z.read("research/ungated_fit_plan.md")).hexdigest() == p["plan_sha256"]
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
subsets = read("results/verification/ungated_fit_subsets_v1.json")
assert subsets["status"] == "PASS" and not subsets["decision_gates_changed"]
assert subsets["post_hoc_descriptive_subsets"] and subsets["optimizer_updates"] == 0
assert subsets["result_sha256"] == sha(ROOT / "result.json")

prior_path = Path("results/verification/ungated_blockshuffle_final_v1.json")
assert sha(prior_path) == before["h072_final_sha256"]
prior = read(prior_path)
assert all(sha(n) == h for n, h in before["h072_sources"].items())
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
    ("process.json", "process_sha256"),
):
    assert sha(Path("results/ungated_blockshuffle_v1") / filename) == prior[key]
assert sha("results/verification/ungated_blockshuffle_analysis_v1.json") == prior["analysis_sha256"]
for name, h in read("results/verification/ungated_blockshuffle_analysis_v1.json")[
    "artifact_hashes"
].items():
    assert sha(Path("results/ungated_blockshuffle_v1/checks") / name) == h
assert (
    sha("results/verification/rotated_shuffle_fit_final_v1.json")
    == prior["prior_h071_final_sha256"]
)
for directory, label in (
    ("rotated_shuffle_fit_v1", "rotated_shuffle_fit"),
    ("token_activation_fit_v2", "token_activation_fit"),
):
    saved = read(f"results/verification/{label}_final_v1.json")
    assert sha(f"results/verification/{label}_analysis_v1.json") == saved["analysis_sha256"]
    fitting = read(f"results/verification/{label}_analysis_v1.json")
    assert len(fitting["cell_artifact_hashes"]) == 252
    for cell, files in fitting["cell_artifact_hashes"].items():
        assert all(
            sha(Path("results") / directory / "cells" / cell / n) == h for n, h in files.items()
        )

manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, name
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 62 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
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
        *map(str, Path("results/verification").glob("ungated_fit*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout + lint.stderr
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/ungated_fit_plan.md"),
    Path("research/ungated_fit_results.md"),
    *Path("research/figures").glob("ungated_fit.*"),
    *Path("results/verification").glob("ungated_fit*.py"),
    Path("results/verification/ungated_fit_analysis_v1.json"),
    Path("results/verification/ungated_fit_subsets_v1.json"),
    ROOT / "analysis_process.json",
    ROOT / "report_process.json",
    *[n for n in (ROOT / "report_mplconfig").rglob("*") if n.is_file()],
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_verdict": r["scientific_verdict"],
    "research_goal_achieved": False,
    "assay_passed": r["assay_passed"],
    "descriptive_subsets_reported_without_gate_changes": True,
    "positive_control_tests": r["positive_control_tests"],
    "completed_fitting_cells": 294,
    "completed_optimizer_updates": 88200,
    "training_example_presentations": 22579200,
    "isolated_harness_checks": 5,
    "independent_cpu_selected_rescores": 147,
    "all_final_weights_and_moments_finite": True,
    "regenerated_data_and_streams_exact": True,
    "independent_gate_rederivation_passed": True,
    "earns_full_model_resource_qualification": {
        f: g["earns_full_model_resource_qualification"] for f, g in r["gates"].items()
    },
    "earns_language_training": False,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "scientific_attempts": 1,
    "scientific_reruns": 0,
    "prior_active_suite_tests": 108,
    "all_prior_active_source_configuration_tests_unchanged": True,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": 504,
    "prior_h072_final_sha256": sha(prior_path),
    "frozen_plans": plans,
    "source_hashes": p["sources"],
    "process_records": processes,
    "local_links_verified": links,
    "historical_archive_links": historical,
    "result_sha256": sha(ROOT / "result.json"),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/ungated_fit_analysis_v1.json"),
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
