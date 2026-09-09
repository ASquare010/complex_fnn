"""H056 final audit: reproduce decisions from artifacts and verify frozen sources."""

import ast
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


root = Path("results/overcomplete_screen_v1")
result, protocol, pre = [read(root / n) for n in ("result.json", "protocol.json", "preflight.json")]
assert result["status"] == "complete" and not result["research_target_passes"]
plan = Path("research/overcomplete_screen_plan.md")
assert sha(plan) == protocol["plan_sha256"] == result["plan_sha256"]
with zipfile.ZipFile(root / "source.zip") as archive:
    for n, h in protocol["provenance"]["source_files"].items():
        assert sha(n) == h == hashlib.sha256(archive.read(n)).hexdigest()
    assert hashlib.sha256(archive.read(plan.as_posix())).hexdigest() == sha(plan)
q = read(root / "worker_qualification.json")
assert q["protocol_sha256"] == sha(root / "protocol.json") and q["plan_sha256"] == sha(plan)
assert q["critical_source_hashes"] == protocol["provenance"]["source_files"]
phases = {}
for label in ("focused_tests", "preflight", "lr300", "lr600", "lr1200", "finish", "tests"):
    p = Path(f"results/verification/overcomplete_screen_{label}_v2.json")
    d = read(p)
    assert d["returncode"] == 0 and d["source_unchanged"] and d["plan_sha256"] == sha(plan)
    assert all(sha(n) == h for n, h in d["source_files"].items())
    phases[label] = {"record": p.as_posix(), "sha256": sha(p), "seconds": d["seconds"]}
assert (
    "245 passed in 37.29s"
    in Path("results/verification/overcomplete_screen_tests_v2.log").read_text()
)
failed = read("results/verification/overcomplete_screen_preflight_v1.json")
assert (
    failed["returncode"] == 1 and failed["source_unchanged"] and failed["plan_sha256"] == sha(plan)
)
assert (
    sha("results/verification/overcomplete_screen_before_preflight_fix_v1.py")
    == failed["source_files"]["src/core/overcomplete_screen.py"]
)
assert (
    "AssertionError: src/core/diagnostics.py"
    in Path("results/verification/overcomplete_screen_preflight_v1.log").read_text()
)
prior = read("results/verification/overcomplete_recompute_final_v1.json")
tested = read("results/verification/overcomplete_recompute_tests_v1.json")
assert all(sha(n) == h for n, h in tested["source_files"].items())
assert sha("results/overcomplete_recompute_v1/result.json") == prior["result_sha256"]
assert pre["retained_exact_variant_count"] == 32
assert pre["validation_stream_exact"] and pre["control_diagnostic_values_exact_on_cpu_probe"]
for n, h in pre["data_hashes"].items():
    assert sha(Path("data/wikitext2_v1") / n) == h
all_rows = {}
for row in pre["controls"]:
    p = Path("results/runs") / row["run"]
    for name, key in (
        ("metrics.json", "metrics_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"),
        ("source.zip", "source_archive_sha256"),
    ):
        assert sha(p / name) == row[key]
    all_rows.setdefault(row["recipe"], []).append(read(p / "metrics.json"))
for row in result["trials"]:
    p = Path("results/runs") / row["run"]
    assert all(sha(p / n) == h for n, h in row["files"].items())
    m = read(p / "metrics.json")
    assert m["model"] == protocol["configurations"][p.name]["model"]
    assert m["training"] == protocol["configurations"][p.name]["training"]
    assert m["optimizer_parameter_groups"] == pre["candidate_optimizer_groups"]
    all_rows.setdefault(row["recipe"], []).append(m)
selected = {
    recipe: min(rows, key=lambda m: (m["validation_loss"], m["training"]["learning_rate"]))
    for recipe, rows in all_rows.items()
}
c = selected["overcomplete_headwise"]
assert c["run"] == result["selected"]
assert all(selected[r]["run"] == run for r, run in result["references"].items())
gates = {
    "at_least_70_percent_fewer_ffn_weights": c["ffn_reduction_percent"] >= 70,
    "beats_calibrated_narrow": c["validation_loss"]
    < selected["calibrated_narrow"]["validation_loss"],
}
for recipe in ("full_swiglu", "full_gelu"):
    gates[f"within_one_percent_{recipe}"] = (
        c["validation_loss"] <= 1.01 * selected[recipe]["validation_loss"]
    )
    gates[f"memory_within_ten_percent_{recipe}"] = (
        c["peak_allocated_vram_bytes"] <= 1.1 * selected[recipe]["peak_allocated_vram_bytes"]
    )
assert gates == result["gates"] and not any(
    gates[n]
    for n in (
        "beats_calibrated_narrow",
        "within_one_percent_full_swiglu",
        "within_one_percent_full_gelu",
    )
)
assert not result["earns_separate_longer_comparison"]
for r, run in result["references"].items():
    assert result["selected_relative_nll_percent"][r] == 100 * (
        c["validation_loss"] / selected[r]["validation_loss"] - 1
    )
for row in result["trials"]:
    m = next(m for m in all_rows["overcomplete_headwise"] if m["run"] == row["run"])
    for recipe in result["references"]:
        ref = next(v for v in all_rows[recipe] if v["training"]["learning_rate"] == row["rate"])
        assert result["same_rate_relative_nll_percent"][str(round(row["rate"] * 1e6))][
            recipe
        ] == 100 * (m["validation_loss"] / ref["validation_loss"] - 1)
for p in ("results/verification/overcomplete_screen_plot_v1.json", root / "observations.json"):
    d = read(p)
    assert all(sha(n) == h for n, h in d["inputs"].items())
    if "outputs" in d:
        assert all(sha(n) == h for n, h in d["outputs"].items())
plans = {**prior["plans_unchanged"], "overcomplete_screen_v1": plan.name}
for folder, name in plans.items():
    assert read(Path("results") / folder / "protocol.json")["plan_sha256"] == sha(
        Path("research") / name
    )
assert read("results/wikitext_screen_v1/result.json")["protocol"]["plan_sha256"] == sha(
    "research/wikitext_screen_plan.md"
)
runs = list(Path("results/runs").glob("*/metrics.json"))
counts = Counter(read(p)["training"]["steps"] for p in runs)
assert len(runs) == 168 and dict(counts) == {200: 86, 800: 66, 3200: 13, 20: 3}
config = ast.parse(Path("src/core/config.py").read_text())
variants = next(
    ast.literal_eval(n.value)
    for n in config.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets)
)
assert len(variants) == 32
files = [
    Path(n)
    for n in (
        "README.md",
        "research/CURRENT_STATE.md",
        "research/idea_bank.md",
        "research/overcomplete_screen_plan.md",
        "research/overcomplete_screen_results.md",
        "src/multihead_ffn/overcomplete.md",
    )
]
links = 0
for p in files:
    for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
        target = target.strip("<>").split("#")[0]
        if not target or re.match(r"^[a-zA-Z]+://", target):
            continue
        assert (p.parent / unquote(target)).exists(), (p, target)
        links += 1
record = {
    "status": "PASS",
    "research_target_passes": False,
    "full_tests_passed": 245,
    "full_test_record": "results/verification/overcomplete_screen_tests_v2.json",
    "all_tested_sources_exact": True,
    "existing_computation_unchanged_from_h055": True,
    "retained_exact_signature_variants": 32,
    "registered_variants": 32,
    "retained_lm_profile_runs": len(runs),
    "run_step_counts": dict(counts),
    "new_training_tokens": 1228800,
    "validation_targets_per_evaluation": 322688,
    "official_test_scored": False,
    "numerical_failures": 0,
    "training_process_failures": 0,
    "preflight_failures": 1,
    "all_data_and_trial_artifact_hashes_exact": True,
    "independently_recomputed_decision": gates,
    "earns_separate_longer_comparison": False,
    "candidate_selected_nll": c["validation_loss"],
    "candidate_peak_mib": c["peak_allocated_vram_bytes"] / 2**20,
    "plans_unchanged": plans,
    "phases": phases,
    "local_links_checked": links,
    "figure_visually_inspected": True,
    "result_sha256": sha(root / "result.json"),
    "source_archive_sha256": sha(root / "source.zip"),
    "documents": {p.as_posix(): sha(p) for p in files},
}
p = Path("results/verification/overcomplete_screen_final_v1.json")
assert not p.exists()
p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("documents", "plans_unchanged", "phases")}
    ),
    flush=True,
)
