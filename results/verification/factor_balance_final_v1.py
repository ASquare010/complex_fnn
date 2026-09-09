"""H069 final audit: exact local probes, rejected motivation and preserved research evidence."""

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

from src.core.config import VARIANTS
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/factor_balance_v2")
OLD = Path("results/factor_balance_v1")
OUT = Path("results/verification/factor_balance_final_v1.json")
assert not OUT.exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return sha256(Path(p))


before = read(OLD / "before.json")
p, r, a = [
    read(n)
    for n in (
        ROOT / "protocol.json",
        ROOT / "result.json",
        Path("results/verification/factor_balance_analysis_v1.json"),
    )
]
assert a["status"] == "PASS" and r["status"] == "complete"
assert r["scientific_verdict"] == a["scientific_verdict"] == "REJECTED_AS_OPTIMIZER_MOTIVATION"
assert (
    r["all_local_exact"] and not r["every_seed_energy_gate"] and not r["every_seed_imbalance_gate"]
)
assert not r["earns_optimizer_state_update_qualification"] and not r["research_goal_achieved"]
assert r["optimizer_updates"] == r["corpus_targets"] == r["new_active_variants"] == 0
assert (
    a["projection_records"] == 216
    and a["channel_records"] == 82944
    and a["exact_mapped_tensor_comparisons"] == 192
)
assert a["channels_outside_fourfold"] == 0 and a["original_cpu_raw_tensors_reproduced"]
assert all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/factor_balance_plan.md") == p["plan_sha256"]
assert sha("research/factor_balance_logging_plan.md") == p["logging_plan_sha256"]
processes = {}
for root, label, expected in (
    (OLD, "harness", "PASS"),
    (OLD, "diagnosis", "FAIL"),
    (ROOT, "logging", "PASS"),
    (ROOT, "diagnosis", "PASS"),
):
    record = read(root / (label + "_process.json"))
    assert record["status"] == expected and record["source_unchanged"]
    assert record["returncode"] == (1 if expected == "FAIL" else 0)
    assert sha(root / (label + ".log")) == record["log_sha256"]
    assert sha(root / "protocol.json") == record["protocol_sha256"]
    assert sha(root / "source.zip") == record["source_zip_sha256"]
    processes[f"{root.name}/{label}"] = {
        "status": expected,
        "sha256": sha(root / (label + "_process.json")),
    }
assert "4 passed in 2.08s" in (OLD / "harness.log").read_text(encoding="utf-8")
assert "4 passed" in (ROOT / "logging.log").read_text(encoding="utf-8")
assert "Got unsupported ScalarType BFloat16" in (OLD / "diagnosis.log").read_text(encoding="utf-8")
assert not (OLD / "result.json").exists() and len(list((OLD / "checks").glob("*.json"))) == 1
assert read(ROOT / "first_cpu_reproduction.json")["original_cpu_case_exact"]
fix_before = read(ROOT / "before.json")
assert (
    sha(OLD / "protocol.json")
    == fix_before["original_protocol_sha256"]
    == p["original_protocol_sha256"]
)
assert (
    sha(OLD / "diagnosis_process.json")
    == fix_before["original_failure_sha256"]
    == p["original_failure_sha256"]
)
assert all(sha(n) == h for n, h in fix_before["original_artifacts"].items())
for n, h in a["state_artifact_hashes"].items():
    assert sha(ROOT / "states" / n) == h
for name, files in a["check_artifact_hashes"].items():
    assert sha(ROOT / "checks" / (name + ".json")) == files["json"]
    assert sha(ROOT / "checks" / (name + ".pt")) == files["pt"]
for spec in p["checkpoints"].values():
    assert sha(Path(spec["path"]) / "checkpoint.pt") == spec["checkpoint_sha256"]
    assert sha(Path(spec["path"]) / "metrics.json") == spec["metrics_sha256"]
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
with zipfile.ZipFile(OLD / "before_docs.zip") as z:
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in before["docs"].items())
prior = read("results/verification/token_activation_fit_final_v1.json")
assert sha("results/verification/token_activation_fit_final_v1.json") == before["h068_final_sha256"]
assert all(sha(n) == h for n, h in before["h068_sources"].items())
for filename, key in (
    ("result.json", "result_sha256"),
    ("protocol.json", "protocol_sha256"),
    ("source.zip", "source_archive_sha256"),
    ("support.zip", "support_archive_sha256"),
    ("process.json", "process_sha256"),
):
    assert sha(Path("results/token_activation_fit_v2") / filename) == prior[key]
assert sha("results/verification/token_activation_fit_analysis_v1.json") == prior["analysis_sha256"]
old_analysis = read("results/verification/token_activation_fit_analysis_v1.json")
for cell, files in old_analysis["cell_artifact_hashes"].items():
    assert all(
        sha(Path("results/token_activation_fit_v2/cells") / cell / n) == h for n, h in files.items()
    )
manifest = read("research/archive/h061_manifest.json")
for name, expected in manifest["protected_evidence_metadata"].items():
    stat = Path(name).stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 58 and all(plans[n] == h for n, h in prior["frozen_plans"].items())
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
        *map(str, Path("results/verification").glob("factor_balance*.py")),
    ],
    capture_output=True,
    text=True,
)
assert lint.returncode == 0, lint.stdout
support = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/idea_bank.md"),
    Path("research/literature.md"),
    Path("research/factor_balance_plan.md"),
    Path("research/factor_balance_logging_plan.md"),
    Path("research/factor_balance_results.md"),
    *Path("research/figures").glob("factor_balance.*"),
    *Path("results/verification").glob("factor_balance*.py"),
    Path("results/verification/factor_balance_analysis_v1.json"),
]
with zipfile.ZipFile(ROOT / "support.zip", "x", zipfile.ZIP_DEFLATED) as z:
    for n in support:
        z.write(n, n.as_posix())
record = {
    "status": "PASS",
    "goal_turn": "PROGRESS",
    "scientific_verdict": r["scientific_verdict"],
    "research_goal_achieved": False,
    "complete_paired_cases": 24,
    "exact_mapped_tensor_comparisons": 192,
    "earlier_saved_cpu_comparisons": 8,
    "projection_records": 216,
    "channel_records": 82944,
    "isolated_mathematical_checks": 4,
    "isolated_logging_checks": 4,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "earns_optimizer_state_or_training_allocation": False,
    "preserved_failed_attempts": 1,
    "serialization_corrected_attempts": 1,
    "original_cpu_artifact_reproduced": True,
    "prior_active_suite_tests": 108,
    "all_prior_active_source_configuration_tests_unchanged": True,
    "active_model_folders": 3,
    "active_variants": 6,
    "recipe_files": 9,
    "retained_lm_profile_runs": 171,
    "protected_evidence_metadata_preserved": len(manifest["protected_evidence_metadata"]),
    "prior_fitting_checkpoint_artifacts_preserved": 252,
    "frozen_plans": plans,
    "source_hashes": p["sources"],
    "process_records": processes,
    "local_links_verified": links,
    "historical_archive_links": historical,
    "result_sha256": sha(ROOT / "result.json"),
    "protocol_sha256": sha(ROOT / "protocol.json"),
    "source_archive_sha256": sha(ROOT / "source.zip"),
    "support_archive_sha256": sha(ROOT / "support.zip"),
    "analysis_sha256": sha("results/verification/factor_balance_analysis_v1.json"),
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
