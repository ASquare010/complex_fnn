"""Final H072 local qualification and historical-evidence preservation audit."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/ungated_blockshuffle_v1")
OUT = Path("results/verification/ungated_blockshuffle_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


before = read(ROOT / "before.json")
p, r, process = [read(ROOT / n) for n in ("protocol.json", "result.json", "process.json")]
a = read("results/verification/ungated_blockshuffle_analysis_v1.json")
assert r["status"] == "complete" and r["scientific_verdict"] == "LOCALLY_QUALIFIED"
assert (
    r["all_checks_pass"]
    and len(r["checks"]) == 18
    and all(c["passed"] for c in r["checks"].values())
)
assert a["status"] == "PASS" and a["exact_raw_tensor_comparisons"] == 72
assert a["independently_reconstructed_projection_matrices"] == 4
assert a["exact_relative_population_error_floor"] == "12/17"
assert not r["research_goal_achieved"] and r["earns_separate_learning_comparison"]
assert r["optimizer_updates"] == r["corpus_targets"] == r["full_model_resource_workers"] == 0
assert p["new_active_variants"] == 0
assert len(p["sources"]) == 94 and all(sha(n) == h for n, h in p["sources"].items())
assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
assert sha(ROOT / "process.log") == process["log_sha256"]
assert "18 passed in 5.79s" in (ROOT / "process.log").read_text(encoding="utf-8")
assert sha(ROOT / "protocol.json") == process["protocol_sha256"]
assert sha(ROOT / "source.zip") == process["source_zip_sha256"]
assert sha("research/ungated_blockshuffle_plan.md") == p["plan_sha256"] == r["plan_sha256"]
assert sha("research/ungated_blockshuffle_theory.md") == p["theory_sha256"] == r["theory_sha256"]
assert all(sha(ROOT / "checks" / n) == h for n, h in a["artifact_hashes"].items())
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
    assert (
        hashlib.sha256(z.read("research/ungated_blockshuffle_plan.md")).hexdigest()
        == p["plan_sha256"]
    )
with zipfile.ZipFile(ROOT / "before_docs.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
prior_path = Path("results/verification/rotated_shuffle_fit_final_v1.json")
assert sha(prior_path) == before["h071_final_sha256"]
prior = read(prior_path)
assert all(sha(n) == h for n, h in before["h071_sources"].items())
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
):
    assert sha(Path("results/rotated_shuffle_fit_v1") / filename) == prior[key]
assert sha("results/verification/rotated_shuffle_fit_analysis_v1.json") == prior["analysis_sha256"]
for phase, h in prior["process_records"].items():
    assert sha(Path("results/rotated_shuffle_fit_v1") / (phase + "_process.json")) == h
assert sha("results/verification/rotated_shuffle_final_v1.json") == prior["prior_h070_final_sha256"]
for directory, label in (
    ("rotated_shuffle_fit_v1", "rotated_shuffle_fit"),
    ("token_activation_fit_v2", "token_activation_fit"),
):
    fitting = read(f"results/verification/{label}_analysis_v1.json")
    assert len(fitting["cell_artifact_hashes"]) == 252
    for cell, files in fitting["cell_artifact_hashes"].items():
        assert all(
            sha(Path("results") / directory / "cells" / cell / n) == h for n, h in files.items()
        )
render_root = ROOT / "report_generation"
failure = read(render_root / "failure.json")
completion = read(render_root / "completion.json")
label = read(render_root / "label_fix.json")
assert (
    failure["exit_code"] == 1
    and failure["cause"] == "unknown"
    and not failure["full_stderr_retained"]
)
assert (
    sha(render_root / "failed_source.py") == failure["source_sha256"] == completion["source_sha256"]
)
assert completion["source_identical_to_failed_attempt"] and completion["observed_exit_code"] == 0
assert sha(render_root / "retry.log") == completion["retry_log_sha256"]
assert sha("research/ungated_blockshuffle_results.md") == completion["report_sha256"]
for ext in ("png", "svg"):
    assert sha(render_root / ("before_label_fix." + ext)) == completion[f"figure_{ext}_sha256"]
    assert sha("research/figures/ungated_blockshuffle." + ext) == label[f"figure_{ext}_sha256"]
assert (
    sha("results/verification/ungated_blockshuffle_report_v1.py") == label["report_source_sha256"]
)
assert sha(render_root / "label_fix.log") == label["log_sha256"]
assert label["observed_exit_code"] == 0
assert (
    failure["scientific_reruns"]
    == completion["scientific_reruns"]
    == label["scientific_reruns"]
    == 0
)

manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, name
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 61 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
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
        *map(str, Path("results/verification").glob("ungated_blockshuffle*.py")),
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
    Path("research/ungated_blockshuffle_plan.md"),
    Path("research/ungated_blockshuffle_theory.md"),
    Path("research/ungated_blockshuffle_results.md"),
    *Path("research/figures").glob("ungated_blockshuffle.*"),
    *Path("results/verification").glob("ungated_blockshuffle*.py"),
    Path("results/verification/ungated_blockshuffle_analysis_v1.json"),
    *[n for n in render_root.rglob("*") if n.is_file()],
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_verdict": r["scientific_verdict"],
    "research_goal_achieved": False,
    "existing_conventional_control_only": True,
    "isolated_qualification_checks": 18,
    "exact_raw_tensor_comparisons": 72,
    "independent_projection_matrix_reconstructions": 4,
    "exact_relative_population_error_floor": "12/17",
    "single_bias_free_layer_scope": True,
    "finite_heldout_error_floor_claimed": False,
    "whole_network_gradient_bound_claimed": False,
    "novel_architecture_claimed": False,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "full_model_resource_workers": 0,
    "earns_separate_learning_comparison": True,
    "earns_language_training": False,
    "scientific_attempts": 1,
    "scientific_reruns": 0,
    "report_only_failure_preserved": True,
    "source_identical_report_recovery_completed": True,
    "figure_label_polished_after_render": True,
    "prior_active_suite_tests": 108,
    "all_prior_active_source_configuration_tests_unchanged": True,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": 504,
    "prior_h071_final_sha256": sha(prior_path),
    "frozen_plans": plans,
    "source_hashes": p["sources"],
    "local_links_verified": links,
    "historical_archive_links": historical,
    "result_sha256": sha(ROOT / "result.json"),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "process_sha256": sha(ROOT / "process.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/ungated_blockshuffle_analysis_v1.json"),
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
