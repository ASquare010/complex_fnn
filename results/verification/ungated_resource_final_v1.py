"""Preserve H074 partial evidence and the complete prior research state."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/ungated_resource_v1")
OUT = Path("results/verification/ungated_resource_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


before, protocol, failure = [
    read(ROOT / n) for n in ("before.json", "protocol.json", "failure.json")
]
audit = read("results/verification/ungated_resource_partial_analysis_v1.json")
assert audit["status"] == "PASS" and audit["completed_comparisons_exact"]
assert audit["checkpoint_pairs_inspected"] == 17 and audit["direct_cpu_fidelity_comparisons"] == 11
assert audit["failure_sha256"] == sha(ROOT / "failure.json")
assert failure["status"] == "INCOMPLETE_RUNTIME_FAILURE" and failure["scientific_attempts"] == 1
assert (
    failure["scientific_retries"]
    == failure["failed_worker_optimizer_updates"]
    == failure["corpus_targets"]
    == 0
)
assert (
    failure["completed_optimizer_updates"] == 340
    and failure["completed_synthetic_training_targets"] == 696320
)
assert (
    failure["completed_initial_probe_targets"] == 34816 and len(failure["unlaunched_workers"]) == 3
)
assert not any(failure["earns_language_screen"].values()) and not (ROOT / "result.json").exists()
assert not list((ROOT / "workers" / failure["failed_worker"]).iterdir())
assert len(protocol["sources"]) == 100 and all(sha(n) == h for n, h in protocol["sources"].items())
assert sha("research/ungated_resource_plan.md") == protocol["plan_sha256"] == before["plan_sha256"]
assert sha(ROOT / "tokens.pt") == protocol["token_file_sha256"]
for cell, files in audit["worker_artifact_hashes"].items():
    assert all(sha(ROOT / "workers" / cell / n) == h for n, h in files.items())
for label, h in audit["process_hashes"].items():
    path = ROOT / "processes" / (label + ".json")
    assert sha(path) == h and sha(path.with_suffix(".log")) == read(path)["log_sha256"]
assert "5 passed" in (ROOT / "processes/harness.log").read_text(encoding="utf-8")
assert sha(ROOT / "processes/gelu_same_block_inner.json") == failure["failure_process_sha256"]
assert sha(ROOT / "processes/gelu_same_block_inner.log") == failure["failure_log_sha256"]
assert sha(ROOT / "coordinator.log") == failure["coordinator_log_sha256"]
assert sha(ROOT / "coordinator_error.log") == failure["coordinator_error_log_sha256"]
assert len(list((ROOT / "processes").glob("*.json"))) == 19
style = read(ROOT / "report_style_fix.json")
assert sha(style["executed_source"]) == style["executed_source_sha256"]
assert sha("results/verification/ungated_resource_report_v1.py") == style["current_source_sha256"]
assert (
    not style["scientific_source_changed"]
    and style["scientific_reruns"] == style["report_rerenders"] == 0
)
for phase in ("partial_analysis", "report"):
    event = read(ROOT / (phase + "_process.json"))
    assert event["status"] == "PASS" and event["returncode"] == 0
    assert (
        event["source_unchanged"]
        and event["scientific_source_unchanged"]
        and event["scientific_reruns"] == 0
    )
    source = (
        style["executed_source"]
        if phase == "report"
        else f"results/verification/ungated_resource_{phase}_v1.py"
    )
    assert sha(source) == event["source_sha256"]
    assert sha(ROOT / (phase + ".log")) == event["log_sha256"]
    assert sha(ROOT / "failure.json") == event["scientific_failure_sha256"]
probe = read("results/verification/ungated_resource_tokenizer_probe_v1.json")
assert probe["status"] == "PASS" and not probe["torch_imported"] and probe["optimizer_updates"] == 0
assert all(sha(n) == v["sha256"] for n, v in probe["files"].items())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in protocol["sources"].items())
    assert (
        hashlib.sha256(z.read("research/ungated_resource_plan.md")).hexdigest()
        == protocol["plan_sha256"]
    )
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
priorpath = Path("results/verification/ungated_fit_final_v1.json")
assert sha(priorpath) == before["h073_final_sha256"]
prior = read(priorpath)
for n, k in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
):
    assert sha(Path("results/ungated_fit_v1") / n) == prior[k]
assert sha("results/verification/ungated_fit_analysis_v1.json") == prior["analysis_sha256"]
assert (
    sha("results/ungated_fit_v1/data.pt")
    == read("results/ungated_fit_v1/result.json")["data_sha256"]
)
assert (
    sha("results/verification/ungated_blockshuffle_final_v1.json")
    == prior["prior_h072_final_sha256"]
)
old_cells = 0
for label, folder in (
    ("ungated_fit", "ungated_fit_v1"),
    ("rotated_shuffle_fit", "rotated_shuffle_fit_v1"),
    ("token_activation_fit", "token_activation_fit_v2"),
):
    old_final = read(f"results/verification/{label}_final_v1.json")
    old_analysis_path = f"results/verification/{label}_analysis_v1.json"
    assert sha(old_analysis_path) == old_final["analysis_sha256"]
    old = read(old_analysis_path)
    for cell, files in old["cell_artifact_hashes"].items():
        assert all(
            sha(Path("results") / folder / "cells" / cell / n) == h for n, h in files.items()
        )
    old_cells += len(old["cell_artifact_hashes"])
assert old_cells == 798
manifest = read("research/archive/h061_manifest.json")
for n, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(n).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 63 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
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
        *map(str, Path("results/verification").glob("ungated_resource*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout + lint.stderr
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/ungated_resource_plan.md"),
    Path("research/ungated_resource_results.md"),
    *Path("research/figures").glob("ungated_resource.*"),
    *Path("results/verification").glob("ungated_resource*.py"),
    Path("results/verification/ungated_resource_partial_analysis_v1.json"),
    Path("results/verification/ungated_resource_tokenizer_probe_v1.json"),
    ROOT / "failure.json",
    ROOT / "partial_analysis_process.json",
    ROOT / "report_process.json",
    ROOT / "report_style_fix.json",
    Path(style["executed_source"]),
    ROOT / "processes/gelu_same_block_inner.json",
    ROOT / "processes/gelu_same_block_inner.log",
    *[n for n in (ROOT / "report_mplconfig").rglob("*") if n.is_file()],
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_status": "INCOMPLETE_RUNTIME_FAILURE",
    "completed_workers": 17,
    "failed_before_updates": 1,
    "unlaunched_workers": 3,
    "completed_optimizer_updates": 340,
    "synthetic_training_targets": 696320,
    "initial_probe_targets": 34816,
    "corpus_targets": 0,
    "scientific_attempts": 1,
    "scientific_retries": 0,
    "isolated_harness_checks": 5,
    "independent_cpu_checkpoint_pairs": 17,
    "exact_available_execution_comparisons": 11,
    "earns_language_screen": {"gelu_same": False, "gelu_matched": False},
    "research_goal_achieved": False,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "prior_active_suite_tests": 108,
    "prior_active_sources_tests_configurations_unchanged": True,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": old_cells,
    "frozen_plans": plans,
    "source_hashes": protocol["sources"],
    "prior_h073_final_sha256": sha(priorpath),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "failure_sha256": sha(ROOT / "failure.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/ungated_resource_partial_analysis_v1.json"),
    "support_artifacts": {n.as_posix(): sha(n) for n in support},
    "local_links_verified": links,
    "historical_archive_links": historical,
    "figure_visually_checked": True,
    "lint": lint.stdout,
}
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k not in ("frozen_plans", "source_hashes", "support_artifacts")
        }
    ),
    flush=True,
)
