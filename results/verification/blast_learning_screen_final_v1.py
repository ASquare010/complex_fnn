"""Seal completed H081 evidence without rerunning training or unrelated studies."""

import ast
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/blast_learning_screen_v1")
OUT = Path("results/verification/blast_learning_screen_final_v1.json")
assert not OUT.exists() and not (ROOT / "support.zip").exists()


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
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)
        assert all(hashlib.sha256(z.read(n.as_posix())).hexdigest() == sha(n) for n in files)


p, r, before = [read(ROOT / n) for n in ("protocol.json", "result.json", "before.json")]
a_path = Path("results/verification/blast_learning_screen_analysis_v1.json")
a = read(a_path)
assert a["status"] == "PASS" and r["status"] == "COMPLETE" and not r["research_goal_achieved"]
assert a["checkpoint_rescores_exact"] == r["language_trials"] == 24
assert a["language_optimizer_updates"] == r["language_optimizer_updates"] == 4800
assert a["analysis_optimizer_updates"] == 0 and r["scientific_retries"] == 0
assert r["qualification_updates"] == 24 and r["qualification_targets"] == 24960
assert a["result_sha256"] == sha(ROOT / "result.json")
assert a["protocol_sha256"] == sha(ROOT / "protocol.json")
assert a["source_archive_sha256"] == sha(ROOT / "source.zip")
assert len(p["sources"]) == 121 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/blast_learning_screen_plan.md") == p["plan_sha256"] == before["plan_sha256"]
assert sha("results/verification/blast_operator_recovery_final_v1.json") == before["h080_final_sha256"]
assert sha("results/blast_operator_recovery_v1/source.zip") == before["h080_source_sha256"]
assert sha("results/blast_operator_recovery_v1/result.json") == before["h080_result_sha256"]
assert all(sha(n) == h for n, h in before["prior_plans"].items())
assert len(list(Path("research").glob("*plan.md"))) == 70
assert len(a["cell_artifact_hashes"]) == 24
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "runs" / cell / n) == h for n, h in files.items())
for label in ("preflight", *p["cells"], "finish"):
    e = read(ROOT / "processes" / (label + ".json"))
    assert e["status"] == "PASS" and e["returncode"] == 0 and e["source_unchanged"]
    assert sha(ROOT / "processes" / (label + ".log")) == e["log_sha256"]
for label in ("analysis", "report"):
    e = read(ROOT / (label + "_process.json"))
    assert e["status"] == "PASS" and e["returncode"] == 0
    assert e["source_unchanged"] and e["scientific_sources_unchanged"]
    assert e["training_repetitions"] == e["optimizer_updates"] == 0
    assert sha(ROOT / (label + ".log")) == e["log_sha256"]
    assert sha(f"results/verification/blast_learning_screen_{label}_v1.py") == e["source_sha256"]
assert all(sha(Path("data/wikitext2_v1") / n) == h for n, h in p["data_files"].items())
assert sha("data/wikitext2_v1/manifest.json") == p["data_manifest_sha256"]
assert {n.name for n in Path("src").iterdir() if n.is_dir() and n.name.endswith("_ffn")} == {
    "dense_ffn", "blockshuffle_ffn", "rational_blockshuffle_ffn"}
tree = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets))
assert len(variants) == 6 and len(list(Path("configs").glob("*.json"))) == 9
docs = [Path(n) for n in ("README.md", "research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md",
                          "research/idea_bank.md", "research/blast_learning_screen_results.md")]
links = 0
for doc in docs:
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        assert (doc.parent / clean).resolve().exists(), (doc, clean)
        links += 1
metadata = sorted({*ROOT.glob("*.json"), *ROOT.glob("*.log"),
                   *(ROOT / "processes").glob("*"), *(ROOT / "preflight").glob("*.json"),
                   *(ROOT / "runs").glob("*/*.json"), *(ROOT / "runs").glob("*/*.jsonl")})
archive(ROOT / "evidence_metadata.zip", metadata)
support = sorted({*docs, Path("research/blast_learning_screen_plan.md"), a_path,
                  *Path("results/verification").glob("blast_learning_screen*.py"),
                  ROOT / "evidence_metadata.zip", ROOT / "source.zip", ROOT / "before_docs.zip"})
archive(ROOT / "support.zip", support)
record = {"status": "PASS", "goal_turn": "PROGRESS", "research_goal_achieved": False,
          "language_trials": 24, "language_optimizer_updates": 4800,
          "language_training_targets": 9830400, "checkpoint_rescores_exact": 24,
          "independent_validation_targets": 7744512, "analysis_optimizer_updates": 0,
          "qualification_checks": 9, "qualification_optimizer_updates": 24,
          "qualification_gradient_pairs_exact": a["qualification_gradient_pairs_exact"],
          "scientific_retries": 0, "active_model_folders": 3, "active_variants": 6, "recipe_files": 9,
          "prior_h080_anchor_verified": True, "unrelated_studies_not_rerun": True,
          "prior_h080_final_sha256": before["h080_final_sha256"],
          "source_hashes": p["sources"], "frozen_plans": {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")},
          "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
          "analysis_sha256": sha(a_path), "source_archive_sha256": sha(ROOT / "source.zip"),
          "support_archive_sha256": sha(ROOT / "support.zip"),
          "evidence_metadata_archive_sha256": sha(ROOT / "evidence_metadata.zip"),
          "support_artifacts": {n.as_posix(): sha(n) for n in support},
          "metadata_artifacts": {n.as_posix(): sha(n) for n in metadata},
          "selected_cells": r["selected_cells"], "earns_longer_comparison": r["earns_longer_comparison"],
          "local_links_verified": links}
payload = (json.dumps(record, indent=2) + "\n").encode()
with OUT.open("xb") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())
assert OUT.read_bytes() == payload
print(json.dumps({k: v for k, v in record.items()
                  if k not in ("source_hashes", "frozen_plans", "support_artifacts", "metadata_artifacts")}), flush=True)
