"""Final bounded CPU artifact checks for the completed joint experiment."""

import ast
import json
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from src.core.conditioning import floor_square_blocks
from src.core.joint_conditioning_report import report
from src.core.reproducibility import provenance, sha256, write_json

torch.set_num_threads(4)
folder = Path("results/joint_conditioning_scale384_v1")
summary = report()
raw = json.loads((folder / "result.json").read_text())
tests = json.loads(Path("results/verification/joint_test_run_v1.json").read_text())
assert tests["returncode"] == 0 and "129 passed" in tests["stdout"]
transforms = {}
for seed in (17, 29, 43):
    row = raw["records"][f"s{seed}_floor"]
    run = Path("results/runs") / row["run"]
    original = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    adjusted = torch.load(
        folder / f"s{seed}_floor/transformed_factors.pt", map_location="cpu", weights_only=True
    )
    transforms[seed] = {}
    for name, value in adjusted.items():
        assert torch.equal(value, floor_square_blocks(original[name], 0.1))
        s = torch.linalg.svdvals(value.double())
        condition = (s.max() / s.min()).item()
        assert condition <= 10.0001
        transforms[seed][name] = {
            "bitwise_reconstruction_matches": True,
            "square_factor_condition": condition,
        }
with zipfile.ZipFile(folder / "source.zip") as archive:
    before = ast.parse(archive.read("src/core/joint_conditioning.py").decode())
after = ast.parse(Path("src/core/joint_conditioning.py").read_text())


class WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


assert ast.dump(WithoutImports().visit(before)) == ast.dump(WithoutImports().visit(after))
checks = {}
for command in (
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "-m", "ruff", "format", "--check", "."],
    ["git", "diff", "--check"],
):
    r = subprocess.run(command, capture_output=True, text=True)
    checks[" ".join(command)] = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    assert r.returncode == 0, checks
paths = [
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
    Path("research/joint_conditioning_results.md"),
    Path("research/joint_conditioning_plan.md"),
    Path("research/broader_corpus_design.md"),
    Path("src/blockshuffle_ffn/README.md"),
]
links = []
for path in paths:
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if target.startswith(("https://", "http://", "#")):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        if resolved == Path("results/verification/joint_checks_v1.json").resolve():
            continue
        assert resolved.exists(), (path, target)
        links.append({"document": path.as_posix(), "target": target})
record = {
    "created_utc": datetime.now(UTC).isoformat(),
    "provenance": provenance(),
    "verification_script_sha256": sha256(Path(__file__)),
    "all_joint_gates_pass": summary["all_gates_pass"],
    "report_artifact_checks_pass": summary["artifact_checks_pass"],
    "transform_reconstruction": transforms,
    "post_measurement_change": "Only optional imports relocated into the worker; all non-import AST nodes match archived source.",
    "test_run_sha256": sha256(Path("results/verification/joint_test_run_v1.json")),
    "tests_passed": 129,
    "checks": checks,
    "local_links": links,
    "visually_inspected_figure": "results/plots/joint_conditioning_scale384.png",
    "failure_disclosure": "Import-time access violation retained; five unmeasured cases completed after explicit unchanged continuation.",
    "interpretation": "Checks support this fixed six-case numerical/quality and projection-conditioning result, not broad superiority, convergence, combined serving speed or a full-network stability proof.",
}
write_json(Path("results/verification/joint_checks_v1.json"), record)
print(
    json.dumps(
        {
            "all_joint_gates_pass": record["all_joint_gates_pass"],
            "tests_passed": record["tests_passed"],
            "local_links": len(links),
            "transformed_factors_checked": sum(len(v) for v in transforms.values()),
        }
    )
)
