"""Audit H070 qualification, report-only recovery and preserved research evidence."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/rotated_shuffle_v1")
OUT = Path("results/verification/rotated_shuffle_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


before = read(ROOT / "before.json")
p = read(ROOT / "protocol.json")
r = read(ROOT / "result.json")
a = read("results/verification/rotated_shuffle_analysis_v1.json")
process = read(ROOT / "process.json")
assert r["status"] == "complete" and r["scientific_verdict"] == "LOCALLY_QUALIFIED"
assert r["all_checks_pass"] and all(c["passed"] for c in r["checks"].values())
assert len(r["checks"]) == p["qualification_checks"] == a["qualification_checks"] == 21
assert a["status"] == "PASS" and a["source_unchanged"]
assert a["exact_raw_tensor_comparisons"] == 140
assert a["finite_nonzero_angle_gradient_vectors"] == 36
assert a["witness_rank"] == 7 and a["original_block_rank_bound"] == 6
assert r["earns_separate_fitting_resource_comparison"]
assert r["optimizer_updates"] == r["corpus_targets"] == r["full_model_resource_workers"] == 0
assert not r["research_goal_achieved"] and p["new_active_variants"] == 0
assert not r["checks"]["rank_witness"]["full_ffn_separation_claimed"]
assert not r["checks"]["isometry"]["whole_ffn_gradient_bound_claimed"]
assert process["status"] == "PASS" and process["returncode"] == 0
assert process["source_unchanged"] and sha(ROOT / "process.log") == process["log_sha256"]
assert "21 passed in 3.97s" in (ROOT / "process.log").read_text(encoding="utf-8")
assert sha(ROOT / "protocol.json") == process["protocol_sha256"]
assert sha(ROOT / "source.zip") == process["source_zip_sha256"]
assert sha("research/rotated_shuffle_plan.md") == p["plan_sha256"] == r["plan_sha256"]
assert sha("research/rotated_shuffle_theory.md") == p["theory_sha256"] == r["theory_sha256"]
assert len(p["sources"]) == 87 and all(sha(n) == h for n, h in p["sources"].items())
assert all(sha(ROOT / "checks" / n) == h for n, h in a["artifact_hashes"].items())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
    assert (
        hashlib.sha256(z.read("research/rotated_shuffle_plan.md")).hexdigest() == p["plan_sha256"]
    )
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())

prior_path = Path("results/verification/factor_balance_final_v1.json")
assert sha(prior_path) == before["h069_final_sha256"]
prior = read(prior_path)
assert all(sha(n) == h for n, h in before["h069_sources"].items())
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
):
    assert sha(Path("results/factor_balance_v2") / filename) == prior[key]
assert sha("results/verification/factor_balance_analysis_v1.json") == prior["analysis_sha256"]
old_analysis = read("results/verification/factor_balance_analysis_v1.json")
for name, h in old_analysis["state_artifact_hashes"].items():
    assert sha(Path("results/factor_balance_v2/states") / name) == h
for name, files in old_analysis["check_artifact_hashes"].items():
    for extension in ("json", "pt"):
        assert (
            sha(Path("results/factor_balance_v2/checks") / (name + "." + extension))
            == files[extension]
        )
for name, spec in prior["process_records"].items():
    directory, phase = name.split("/")
    assert sha(Path("results") / directory / (phase + "_process.json")) == spec["sha256"]

h069_before = read("results/factor_balance_v1/before.json")
assert (
    sha("results/verification/token_activation_fit_final_v1.json")
    == h069_before["h068_final_sha256"]
)
fitting = read("results/verification/token_activation_fit_final_v1.json")
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
    ("process.json", "process_sha256"),
):
    assert sha(Path("results/token_activation_fit_v2") / filename) == fitting[key]
assert (
    sha("results/verification/token_activation_fit_analysis_v1.json") == fitting["analysis_sha256"]
)
fitting_analysis = read("results/verification/token_activation_fit_analysis_v1.json")
assert len(fitting_analysis["cell_artifact_hashes"]) == 252
for cell, files in fitting_analysis["cell_artifact_hashes"].items():
    assert all(
        sha(Path("results/token_activation_fit_v2/cells") / cell / n) == h for n, h in files.items()
    )

manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, name
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 59 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
# Parse the literal registry without loading scientific dependencies for bookkeeping.
config_ast = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(
    ast.literal_eval(n.value)
    for n in config_ast.body
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

failure_root = ROOT / "report_generation"
failure = read(failure_root / "import_failure.json")
render = read(failure_root / "completion.json")
assert failure["exit_code"] == 1 and failure["cause"] == "unknown"
assert not failure["full_stderr_retained"] and failure["scientific_reruns"] == 0
assert sha(failure_root / "failed_source.py") == failure["source_sha256"]
assert render["status"] == "PASS" and render["observed_tool_exit_code"] == 0
assert render["scientific_reruns"] == 0
assert sha(failure_root / "rendered_source.py") == render["report_source_sha256"]
assert sha(failure_root / "rendered_report.md") == render["report_sha256"]
assert sha("research/figures/rotated_shuffle.png") == render["figure_png_sha256"]
assert sha("research/figures/rotated_shuffle.svg") == render["figure_svg_sha256"]
report = Path("research/rotated_shuffle_results.md").read_text(encoding="utf-8")
assert sha(ROOT / "result.json") in report and "tok_backup" in report
assert "16 identically zero" in Path("research/token_activation_fit_results.md").read_text(
    encoding="utf-8"
)

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
        *map(str, Path("results/verification").glob("rotated_shuffle*.py")),
        "results/verification/token_activation_fit_report_v1.py",
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout + lint.stderr

support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/literature.md"),
    Path("research/rotated_shuffle_plan.md"),
    Path("research/rotated_shuffle_theory.md"),
    Path("research/rotated_shuffle_results.md"),
    Path("research/token_activation_fit_results.md"),
    *Path("research/figures").glob("rotated_shuffle.*"),
    *Path("results/verification").glob("rotated_shuffle*.py"),
    Path("results/verification/rotated_shuffle_analysis_v1.json"),
    Path("results/verification/token_activation_fit_report_v1.py"),
    *[n for n in failure_root.iterdir() if n.is_file()],
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_verdict": r["scientific_verdict"],
    "research_goal_achieved": False,
    "isolated_qualification_checks": 21,
    "exact_raw_tensor_comparisons": 140,
    "finite_nonzero_angle_gradient_vectors": 36,
    "projection_family_expansion_locally_qualified": True,
    "full_ffn_separation_claimed": False,
    "learning_advantage_measured": False,
    "whole_network_gradient_bound_claimed": False,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "earns_separate_fitting_resource_comparison": True,
    "earns_language_training": False,
    "scientific_attempts": 1,
    "report_import_failures_preserved": 1,
    "report_only_dependency_correction_completed": True,
    "scientific_reruns": 0,
    "report_prose_spacing_polished_after_render": True,
    "prior_active_suite_tests": 108,
    "all_prior_active_source_configuration_tests_unchanged": True,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": 252,
    "prior_h069_final_sha256": sha(prior_path),
    "frozen_plans": plans,
    "source_hashes": p["sources"],
    "local_links_verified": links,
    "historical_archive_links": historical,
    "result_sha256": sha(ROOT / "result.json"),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "process_sha256": sha(ROOT / "process.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "before_documents_archive_sha256": sha(ROOT / "before_docs.zip"),
    "analysis_sha256": sha("results/verification/rotated_shuffle_analysis_v1.json"),
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
