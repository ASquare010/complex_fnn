"""Seal H084 without repeating model execution or altering prior evidence."""

import ast
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/latent_affine_mechanism_v1")
OUT = Path("results/verification/latent_affine_mechanism_final_v1.json")


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def archive(path, files):
    with path.open("xb") as f:
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
            for n in files:
                z.write(n, n.as_posix())
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(files)
        assert all(hashlib.sha256(z.read(n.as_posix())).hexdigest() == sha(n) for n in files)


assert not OUT.exists() and not (ROOT / "support.zip").exists()
p, r, b = [read(ROOT / n) for n in ("protocol.json", "result.json", "before.json")]
a_path = Path("results/verification/latent_affine_mechanism_analysis_v1.json")
a = read(a_path)
assert a["status"] == "PASS" and r["status"] == "COMPLETE"
assert not r["research_goal_achieved"] and not r["earns_training"]
assert a["checkpoint_cases"] == r["checkpoint_cases"] == len(r["rows"]) == 12
assert a["new_reset_rescores_exact"] == a["prediction_hashes_exact"] == 24
assert a["fp64_fold_comparisons"] == r["fp64_fold_comparisons"] == 24
assert a["origin_cases_verified"] == r["origin_mode_cases"] == 48
assert a["original_and_reset_both_anchors_exact"] == 24 and a["witness_verified"]
assert a["residual_moment_records_verified"] == r["reporting_passes"] == 72
assert a["optimizer_updates"] == r["optimizer_updates"] == r["corpus_targets"] == 0
assert r["training_repetitions"] == 0 and r["reporting_examples"] == 294912
assert a["independent_reporting_examples"] == 98304
assert a["protocol_sha256"] == sha(ROOT / "protocol.json") == r["protocol_sha256"]
assert a["result_sha256"] == sha(ROOT / "result.json")
assert a["source_archive_sha256"] == sha(ROOT / "source.zip")
assert a["artifact_hashes"] == r["artifact_hashes"]
assert a["reset_to_original_ratios"] == r["reset_to_original_ratios"]
assert a["maximum_fold_mse_drift"] == r["maximum_fold_mse_drift"] <= 1e-5
assert len(p["sources"]) == 131 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/latent_affine_mechanism_plan.md") == p["plan_sha256"] == b["plan_sha256"]
assert all(sha(n) == h for n, h in b["prior_plans"].items())
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 73
assert all(sha(n) == h for n, h in b["anchors"].items())
assert len(b["checkpoint_files"]) == 60
for name, old in b["checkpoint_files"].items():
    n = Path(name)
    assert sha(n) == old["sha256"] and n.stat().st_size == old["size"]
    assert n.stat().st_mtime_ns == old["mtime_ns"]
assert sha(ROOT / "witness.pt") == r["witness_sha256"]
for cell, files in r["artifact_hashes"].items():
    assert all(sha(ROOT / "cells" / cell / n) == h for n, h in files.items())
process = read(ROOT / "process.json")
assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
assert sha(ROOT / "study.log") == process["log_sha256"]
for phase in ("analysis", "report"):
    e = read(ROOT / (phase + "_process.json"))
    assert e["status"] == "PASS" and e["returncode"] == 0
    assert e["source_unchanged"] and e["scientific_sources_unchanged"]
    assert e["optimizer_updates"] == e["training_repetitions"] == 0
    assert sha(ROOT / (phase + ".log")) == e["log_sha256"]
    assert sha(f"results/verification/latent_affine_mechanism_{phase}_v1.py") == e["source_sha256"]
assert {n.name for n in Path("src").iterdir() if n.is_dir() and n.name.endswith("_ffn")} == {
    "dense_ffn", "blockshuffle_ffn", "rational_blockshuffle_ffn"}
tree = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets))
assert len(variants) == 6 and len(list(Path("configs").glob("*.json"))) == 9
docs = [Path(n) for n in ("README.md", "research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md",
                          "research/idea_bank.md", "research/learnable_activation_domain.md",
                          "research/latent_affine_mechanism_results.md")]
links = 0
for doc in docs:
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        assert (doc.parent / clean).resolve().exists(), (doc, clean)
        links += 1
metadata = sorted({*ROOT.glob("*.json"), *ROOT.glob("*.log"), *(ROOT / "cells").glob("*/*.json")})
archive(ROOT / "evidence_metadata.zip", metadata)
support = sorted({*docs, Path("research/latent_affine_mechanism_plan.md"), a_path,
                  *Path("results/verification").glob("latent_affine_mechanism*.py"),
                  ROOT / "evidence_metadata.zip", ROOT / "source.zip", ROOT / "before_docs.zip"})
archive(ROOT / "support.zip", support)
record = {"status": "PASS", "goal_turn": "PROGRESS", "previous_goal_turn": "PROGRESS",
          "research_goal_achieved": False, "earns_training": False,
          "scientific_verdict": r["scientific_verdict"], "checkpoint_cases": 12,
          "reporting_passes": 72, "reporting_examples": 294912, "optimizer_updates": 0,
          "corpus_targets": 0, "training_repetitions": 0, "independent_new_reset_rescores_exact": 24,
          "independent_reporting_examples": 98304, "fp64_fold_comparisons": 24,
          "origin_cases": 48, "origin_jvp_directions_per_case": 8,
          "origin_witness_verified": True, "prior_h083_checkpoint_files_unchanged": 60,
          "prior_h083_anchors_verified": True, "active_model_folders": 3,
          "active_variants": 6, "recipe_files": 9, "source_hashes": p["sources"],
          "frozen_plans": plans, "maximum_fold_mse_drift": r["maximum_fold_mse_drift"],
          "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
          "analysis_sha256": sha(a_path), "source_archive_sha256": sha(ROOT / "source.zip"),
          "support_archive_sha256": sha(ROOT / "support.zip"),
          "evidence_metadata_archive_sha256": sha(ROOT / "evidence_metadata.zip"),
          "support_artifacts": {n.as_posix(): sha(n) for n in support},
          "metadata_artifacts": {n.as_posix(): sha(n) for n in metadata},
          "local_links_verified": links, "unrelated_studies_not_rerun": True}
payload = (json.dumps(record, indent=2)+"\n").encode()
with OUT.open("xb") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())
assert OUT.read_bytes() == payload
print(json.dumps({k: v for k, v in record.items() if k not in
                  ("source_hashes", "frozen_plans", "support_artifacts", "metadata_artifacts")}), flush=True)
