"""Seal H083 evidence and verify retained scope; no training or model execution."""

import ast
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/latent_activation_recovery_v1")
OUT = Path("results/verification/latent_activation_recovery_final_v1.json")


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def archive(path, files):
    assert len(files) == len(set(files))
    with path.open("xb") as f:
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
            for n in files:
                z.write(n, n.as_posix())
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)
        assert all(hashlib.sha256(z.read(n.as_posix())).hexdigest() == sha(n) for n in files)


assert not OUT.exists() and not (ROOT / "support.zip").exists()
p, r, before = [read(ROOT / n) for n in ("protocol.json", "result.json", "before.json")]
a_path = Path("results/verification/latent_activation_recovery_analysis_v1.json")
a = read(a_path)
assert a["status"] == "PASS" and r["status"] == "complete" and not r["research_goal_achieved"]
assert a["scientific_verdict"] == r["scientific_verdict"] == "REJECTED_AT_THIS_FITTING_BUDGET"
assert a["gates"] == r["gates"] and not any(g["earns_full_model_resource_qualification"] for g in r["gates"].values())
assert a["selection_rescores_exact"] == a["checkpoint_records_verified"] == r["cells"] == 192
assert a["selected_heldout_rescores_exact"] == r["selection_cells"] == 96
assert a["reset_ablations_exact"] == r["post_training_reset_ablations"] == 36
assert a["optimizer_updates_verified"] == r["optimizer_updates"] == 115200
assert a["analysis_optimizer_updates"] == 0 and r["training_example_presentations"] == 29491200
assert a["result_sha256"] == sha(ROOT / "result.json")
assert a["protocol_sha256"] == sha(ROOT / "protocol.json")
assert a["source_archive_sha256"] == sha(ROOT / "source.zip")
assert a["data_sha256"] == sha(ROOT / "data.pt")
assert len(p["sources"]) == 128 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/latent_activation_recovery_plan.md") == p["plan_sha256"] == before["plan_sha256"]
assert sha("research/latent_activation_plan.md") == p["original_plan_sha256"] == before["original_plan_sha256"]
assert sha("research/latent_activation_theory.md") == p["theory_sha256"] == before["theory_sha256"]
assert all(sha(n) == h for n, h in before["prior_plans"].items())
assert len(list(Path("research").glob("*plan.md"))) == 72
anchors = {
    "h081_final_sha256": "results/verification/blast_learning_screen_final_v1.json",
    "h081_result_sha256": "results/blast_learning_screen_v1/result.json",
    "h081_source_sha256": "results/blast_learning_screen_v1/source.zip",
    "h082_failure_audit_sha256": "results/verification/latent_activation_preflight_failure_v1.json",
    "h082_source_sha256": "results/latent_activation_v1/source.zip",
    "h082_protocol_sha256": "results/latent_activation_v1/protocol.json",
    "h082_preflight_log_sha256": "results/latent_activation_v1/processes/preflight.log",
    "h082_report_sha256": "research/latent_activation_results.md",
}
assert all(sha(path) == before[k] for k, path in anchors.items())
assert len(before["h082_files"]) == 27
for name, old in before["h082_files"].items():
    f = Path(name)
    assert sha(f) == old["sha256"] and f.stat().st_size == old["size"]
    assert f.stat().st_mtime_ns == old["mtime_ns"]
assert len(a["cell_artifact_hashes"]) == 192
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "cells" / cell / n) == h for n, h in files.items())
assert len(a["selection_artifact_hashes"]) == 96
assert all(sha(ROOT / "selections" / n) == h for n, h in a["selection_artifact_hashes"].items())
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
for phase in ("preflight", "fitting"):
    e = read(ROOT / "processes" / (phase + ".json"))
    assert e["status"] == "PASS" and e["returncode"] == 0 and e["source_unchanged"]
    assert sha(ROOT / "processes" / (phase + ".log")) == e["log_sha256"]
for phase in ("analysis", "report"):
    e = read(ROOT / (phase + "_process.json"))
    assert e["status"] == "PASS" and e["returncode"] == 0
    assert e["source_unchanged"] and e["scientific_sources_unchanged"]
    assert e["training_repetitions"] == e["optimizer_updates"] == 0
    assert sha(ROOT / (phase + ".log")) == e["log_sha256"]
    assert sha(f"results/verification/latent_activation_recovery_{phase}_v1.py") == e["source_sha256"]
assert p["explicit_preflight_repetitions"] == a["explicit_preflight_repetitions"] == 1
assert p["total_preflight_attempts"] == 2
assert p["fitting_repetitions"] == a["fitting_repetitions"] == 0
assert {n.name for n in Path("src").iterdir() if n.is_dir() and n.name.endswith("_ffn")} == {
    "dense_ffn", "blockshuffle_ffn", "rational_blockshuffle_ffn"}
tree = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets))
assert len(variants) == 6 and len(list(Path("configs").glob("*.json"))) == 9
docs = [Path(n) for n in ("README.md", "research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md",
                          "research/idea_bank.md", "research/learnable_activation_domain.md",
                          "research/latent_activation_recovery_results.md")]
links = 0
for doc in docs:
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        assert (doc.parent / clean).resolve().exists(), (doc, clean)
        links += 1
metadata = sorted({*ROOT.glob("*.json"), *ROOT.glob("*.log"), *(ROOT / "processes").glob("*"),
                   *(ROOT / "preflight").glob("*.json"), *(ROOT / "cells").glob("*/*.json"),
                   *(ROOT / "selections").glob("*.json")})
archive(ROOT / "evidence_metadata.zip", metadata)
support = sorted({*docs, Path("research/latent_activation_plan.md"), Path("research/latent_activation_recovery_plan.md"),
                  Path("research/latent_activation_theory.md"), a_path,
                  *Path("results/verification").glob("latent_activation_recovery*.py"),
                  ROOT / "evidence_metadata.zip", ROOT / "source.zip", ROOT / "before_docs.zip"})
archive(ROOT / "support.zip", support)
record = {
    "status": "PASS", "goal_turn": "PROGRESS", "previous_goal_turn": "VERIFIED_WAIT",
    "research_goal_achieved": False, "scientific_verdict": r["scientific_verdict"],
    "fitting_cells": 192, "fitting_optimizer_updates": 115200,
    "training_example_presentations": 29491200, "selection_rescores_exact": 192,
    "selected_heldout_rescores_exact": 96, "reset_ablations_exact": 36,
    "analysis_optimizer_updates": 0, "corpus_targets": 0, "full_model_resource_workers": 0,
    "qualification_checks": 8, "qualification_optimizer_updates": 0,
    "qualification_gradient_pairs_exact": 120, "explicit_preflight_repetitions": 1,
    "total_preflight_attempts": 2, "fitting_repetitions": 0, "prior_h082_files_unchanged": 27,
    "prior_h082_observations_exact": 7, "prior_h082_tensor_payloads_exact": 6,
    "active_model_folders": 3, "active_variants": 6, "recipe_files": 9,
    "prior_h081_and_h082_anchors_verified": True, "unrelated_studies_not_rerun": True,
    "source_hashes": p["sources"],
    "frozen_plans": {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")},
    "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
    "analysis_sha256": sha(a_path), "source_archive_sha256": sha(ROOT / "source.zip"),
    "data_sha256": sha(ROOT / "data.pt"), "support_archive_sha256": sha(ROOT / "support.zip"),
    "evidence_metadata_archive_sha256": sha(ROOT / "evidence_metadata.zip"),
    "support_artifacts": {n.as_posix(): sha(n) for n in support},
    "metadata_artifacts": {n.as_posix(): sha(n) for n in metadata},
    "gates": r["gates"], "local_links_verified": links,
}
payload = (json.dumps(record, indent=2) + "\n").encode()
with OUT.open("xb") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())
assert OUT.read_bytes() == payload
print(json.dumps({k: v for k, v in record.items() if k not in
                  ("source_hashes", "frozen_plans", "support_artifacts", "metadata_artifacts")}), flush=True)
