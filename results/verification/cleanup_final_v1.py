"""Audit the concrete cleanup against preserved source, evidence and active tests."""

import ast
import hashlib
import importlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

import torch

from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.transformer import Transformer

ROOT = Path.cwd().resolve()
OUT = ROOT / "results/verification/cleanup_final_v1.json"
assert not OUT.exists()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


manifest = json.loads((ROOT / "research/archive/h057_manifest.json").read_text())
archive_path = ROOT / "research/archive/h057_source.zip"
assert sha(archive_path) == manifest["archive_sha256"]
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()} == manifest[
        "files"
    ]

for name, expected in manifest["protected_evidence_metadata"].items():
    actual = (ROOT / name).stat()
    assert {"size": actual.st_size, "mtime_ns": actual.st_mtime_ns} == expected, name
plans = {p.as_posix(): sha(p) for p in Path("research").glob("*plan.md")}
assert len(plans) == 47
assert all(manifest["files"][name] == digest for name, digest in plans.items())

suite = json.loads(Path("results/verification/cleanup_tests_v1.json").read_text())
assert suite["status"] == "PASS" and suite["returncode"] == 0 and suite["source_unchanged"]
assert all(sha(Path(name)) == digest for name, digest in suite["source_hashes"].items())
log = Path("results/verification/cleanup_tests_v1.log")
assert sha(log) == suite["log_sha256"] and "89 passed" in log.read_text()
initial = json.loads(Path("results/verification/cleanup_before_signatures_v1.json").read_text())
assert len(initial) == 12
for name in ("cleanup_after_signatures_v1.json", "cleanup_final_signatures_v1.json"):
    assert json.loads((Path("results/verification") / name).read_text()) == initial

expected_variants = {
    "gelu",
    "swiglu",
    "gelu_narrow",
    "swiglu_narrow",
    "blockshuffle_swiglu",
    "blockshuffle_swiglu_rational",
}
assert set(VARIANTS) == expected_variants
folders = sorted(
    p.name for p in Path("src").iterdir() if p.is_dir() and p.name not in ("core", "__pycache__")
)
assert folders == ["blockshuffle_ffn", "dense_ffn", "rational_blockshuffle_ffn"]
moves = json.loads(Path("results/verification/cleanup_moves_v1.json").read_text())
for name in moves["directories"] + moves["files"]:
    assert not Path(name).exists(), name
    assert (Path(moves["destination"]) / name).exists(), name
configs = list(Path("configs").glob("*.json"))
assert len(configs) == 9
for p in configs:
    assert sha(p) == manifest["files"][p.as_posix()]
    raw = json.loads(p.read_text())
    ModelConfig(**raw["model"]).validate()
    TrainConfig(**raw["training"]).validate()

active_modules = []
for p in sorted(Path("src").rglob("*.py")):
    ast.parse(p.read_text())
    module = p.with_suffix("").as_posix().replace("/", ".").removesuffix(".__init__")
    importlib.import_module(module)
    active_modules.append(module)
commands = {}
for label, args in {
    "lint": ["-m", "ruff", "check", "src", "tests"],
    "cli": ["-m", "src.core.cli", "counts"],
    "entrypoint": ["main.py", "--help"],
}.items():
    completed = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert completed.returncode == 0, (label, completed.stderr, completed.stdout)
    commands[label] = completed.stdout
assert {json.loads(line)["variant"] for line in commands["cli"].splitlines()} == expected_variants

# Real retained checkpoints must remain loadable, not just freshly initialized fixtures.
runs = sorted(Path("results/runs").glob("*/metrics.json"))
assert len(runs) == 168
selected = {}
for path in runs:
    raw = json.loads(path.read_text())
    variant = raw["model"]["variant"]
    if variant in expected_variants and (path.parent / "checkpoint.pt").exists():
        selected[variant] = path.parent
assert set(selected) == expected_variants
torch.set_num_threads(2)
checkpoints = {}
for variant, run in selected.items():
    path = run / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = Transformer(cfg, checkpoint["training_config"]["seed"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        logits = model(torch.zeros((1, 2), dtype=torch.long))
    assert torch.isfinite(logits).all()
    checkpoints[variant] = {
        "run": run.as_posix(),
        "checkpoint_sha256": sha(path),
        "strict_load": True,
        "synthetic_forward_finite": True,
    }
    del model, checkpoint

links = 0
frozen_links = []
for p in [
    Path("README.md"),
    *Path("doc").glob("*.md"),
    *Path("src").rglob("*.md"),
    *Path("configs").glob("*.md"),
    *Path("research").rglob("*.md"),
]:
    for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
        clean = unquote(target.strip("<>").split("#")[0])
        if not clean or re.match(r"^[a-zA-Z]+://", clean):
            continue
        if (p.parent / clean).exists():
            links += 1
        else:
            assert p.parent == Path("research") and p.name.endswith("plan.md"), (p, clean)
            intended = (ROOT / p.parent / clean).resolve().relative_to(ROOT).as_posix()
            assert (
                intended in manifest["files"]
                and (ROOT / "research/archive/retired" / intended).exists()
            )
            frozen_links.append(
                {
                    "plan": p.as_posix(),
                    "original_target": clean,
                    "resolution": "complete source snapshot and archive guide",
                }
            )
assert len(frozen_links) == 3

record = {
    "status": "PASS",
    "cleanup_complete": True,
    "research_target_achieved": False,
    "goal_turn": "PROGRESS",
    "registered_variants": list(VARIANTS),
    "model_folders": folders,
    "core_python_files": len(list(Path("src/core").glob("*.py"))),
    "active_python_modules_imported": len(active_modules),
    "active_test_files": len(list(Path("tests").glob("*.py"))),
    "tests_passed": 89,
    "retained_recipe_files": len(configs),
    "retired_folders": len(moves["directories"]),
    "retired_individual_files": len(moves["files"]),
    "source_snapshot_files": len(manifest["files"]),
    "source_archive_sha256": sha(archive_path),
    "all_archived_bytes_verified": True,
    "protected_result_data_files_metadata_unchanged": len(manifest["protected_evidence_metadata"]),
    "retained_lm_profile_runs": len(runs),
    "real_checkpoints_loaded": checkpoints,
    "exact_cpu_gpu_signature_cases": len(initial),
    "signature_scope": "six variants, CPU FP32/CUDA BF16, initial weights, nonzero shape controls, outputs, loss, gradients, optimizer groups, two updates, diagnostics",
    "tested_sources_still_exact": True,
    "source_hashes": suite["source_hashes"],
    "frozen_plans_unchanged": plans,
    "local_links_verified": links,
    "immutable_historical_links_resolve_from_snapshot": frozen_links,
    "commands": commands,
    "corpus_training_or_scoring_added": False,
    "next_hypothesis": "H058 additive block/low-rank; unimplemented, no training allocation frozen",
    "documents": {
        p.as_posix(): sha(p)
        for p in [
            Path("README.md"),
            Path("research/CURRENT_STATE.md"),
            Path("research/cleanup_results.md"),
            Path("research/archive/README.md"),
            Path("research/idea_bank.md"),
            *Path("src").rglob("*.md"),
            Path("configs/README.md"),
        ]
    },
}
OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k
            not in (
                "source_hashes",
                "frozen_plans_unchanged",
                "commands",
                "documents",
                "real_checkpoints_loaded",
            )
        }
    ),
    flush=True,
)
