"""Seal H085 controlled-training evidence after independent verification."""

import ast
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("results/latent_offset_fit_v1")
OUT = Path("results/verification/latent_offset_fit_final_v1.json")


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
p, r, b, manifest = [read(ROOT / n) for n in ("protocol.json", "result.json", "before.json", "data_manifest.json")]
a_path = Path("results/verification/latent_offset_fit_analysis_v1.json")
a = read(a_path)
assert a["status"] == "PASS" and r["status"] == "complete" and not r["research_goal_achieved"]
assert a["scientific_verdict"] == r["scientific_verdict"] and a["gates"] == r["gates"]
assert a["robustness"] == r["robustness"]
assert a["checkpoint_records_verified"] == a["selection_rescores_exact"] == a["all_rate_reporting_rescores_exact"] == r["cells"] == 240
assert a["selected_reporting_rescores_exact"] == r["selection_cells"] == 120
assert a["reset_ablations_exact"] == r["post_training_reset_ablations"] == 48
assert a["optimizer_updates_verified"] == r["optimizer_updates"] == 144000
assert a["analysis_optimizer_updates"] == 0 and r["training_example_presentations"] == 36864000
assert a["additional_unselected_reporting_passes"] == 120
assert a["analysis_selection_examples"] == 983040 and a["analysis_reporting_and_reset_examples"] == 1179648
assert a["result_sha256"] == sha(ROOT / "result.json")
assert a["protocol_sha256"] == r["protocol_sha256"] == manifest["protocol_sha256"] == sha(ROOT / "protocol.json")
assert a["source_archive_sha256"] == sha(ROOT / "source.zip")
assert a["data_sha256"] == r["data_sha256"] == manifest["data_sha256"] == sha(ROOT / "data.pt")
assert a["fresh_data_regenerated_exact"] and a["initial_states_reconstructed"] == 30 and a["rng_states_exact"] == 240
assert a["qualification_checks"] == 8 and a["qualification_exact_gradient_pairs"] == 80
assert len(p["sources"]) == 135 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/latent_offset_fit_plan.md") == p["plan_sha256"] == b["plan_sha256"]
assert all(sha(n) == h for n, h in b["prior_plans"].items())
assert all(sha(n) == h for n, h in b["anchors"].items())
plans = {n.as_posix(): sha(n) for n in Path("research").glob("*plan.md")}
assert len(plans) == 74
assert len(a["cell_artifact_hashes"]) == 240 and len(a["selection_artifact_hashes"]) == 120
for cell, files in a["cell_artifact_hashes"].items():
    assert all(sha(ROOT / "cells" / cell / n) == h for n, h in files.items())
assert all(sha(ROOT / "selections" / n) == h for n, h in a["selection_artifact_hashes"].items())
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
for phase in ("preflight", "fitting"):
    e = read(ROOT / "processes" / (phase+".json"))
    assert e["status"] == "PASS" and e["returncode"] == 0 and e["source_unchanged"]
    assert sha(ROOT / "processes" / (phase+".log")) == e["log_sha256"]
for phase in ("analysis", "report"):
    e = read(ROOT / (phase+"_process.json"))
    assert e["status"] == "PASS" and e["returncode"] == 0
    assert e["source_unchanged"] and e["scientific_sources_unchanged"]
    assert e["optimizer_updates"] == e["training_repetitions"] == 0
    assert sha(ROOT / (phase+".log")) == e["log_sha256"]
    assert sha(f"results/verification/latent_offset_fit_{phase}_v1.py") == e["source_sha256"]
observer = read(ROOT / "observation_shell_failure.json")
assert observer["status"] == "OBSERVER_FAILURE_RECOVERED" and not observer["experiment_restarted"]
assert observer["observer_exit_code"] == -1073740940
assert p["scientific_retries"] == a["scientific_retries"] == 0
assert {n.name for n in Path("src").iterdir() if n.is_dir() and n.name.endswith("_ffn")} == {
    "dense_ffn", "blockshuffle_ffn", "rational_blockshuffle_ffn"}
tree = ast.parse(Path("src/core/config.py").read_text(encoding="utf-8"))
variants = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets))
assert len(variants) == 6 and len(list(Path("configs").glob("*.json"))) == 9
docs = [Path(n) for n in ("README.md", "research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md",
                          "research/idea_bank.md", "research/learnable_activation_domain.md",
                          "research/latent_offset_fit_results.md")]
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
support = sorted({*docs, Path("research/latent_offset_fit_plan.md"), a_path,
                  *Path("results/verification").glob("latent_offset_fit*.py"),
                  ROOT / "evidence_metadata.zip", ROOT / "source.zip", ROOT / "before_docs.zip"})
archive(ROOT / "support.zip", support)
record = {"status": "PASS", "goal_turn": "PROGRESS", "previous_goal_turn": "PROGRESS",
          "research_goal_achieved": False, "scientific_verdict": r["scientific_verdict"],
          "fitting_cells": 240, "fitting_optimizer_updates": 144000,
          "training_example_presentations": 36864000, "selection_rescores_exact": 240,
          "all_rate_reporting_rescores_exact": 240, "selected_reporting_rescores_exact": 120,
          "reset_ablations_exact": 48, "analysis_optimizer_updates": 0,
          "additional_unselected_reporting_passes": 120, "corpus_targets": 0,
          "full_model_resource_workers": 0, "qualification_checks": 8,
          "qualification_optimizer_updates": 0, "qualification_gradient_pairs_exact": 80,
          "scientific_retries": 0, "observer_shell_failure_preserved": True,
          "active_model_folders": 3, "active_variants": 6, "recipe_files": 9,
          "prior_h083_and_h084_anchors_verified": True, "unrelated_studies_not_rerun": True,
          "source_hashes": p["sources"], "frozen_plans": plans,
          "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
          "analysis_sha256": sha(a_path), "source_archive_sha256": sha(ROOT / "source.zip"),
          "data_sha256": sha(ROOT / "data.pt"), "support_archive_sha256": sha(ROOT / "support.zip"),
          "evidence_metadata_archive_sha256": sha(ROOT / "evidence_metadata.zip"),
          "support_artifacts": {n.as_posix(): sha(n) for n in support},
          "metadata_artifacts": {n.as_posix(): sha(n) for n in metadata},
          "gates": r["gates"], "robustness": r["robustness"], "local_links_verified": links}
payload = (json.dumps(record, indent=2)+"\n").encode()
with OUT.open("xb") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())
assert OUT.read_bytes() == payload
print(json.dumps({k: v for k, v in record.items() if k not in
                  ("source_hashes", "frozen_plans", "support_artifacts", "metadata_artifacts")}), flush=True)
