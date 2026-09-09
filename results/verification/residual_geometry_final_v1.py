"""Independent H057 final artifact and interpretation audit."""

import ast
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


root = Path("results/residual_geometry_v1")
protocol = read(root / "protocol.json")
result = read(root / "result.json")
assert (
    result["status"] == "complete"
    and not result["research_target_passes"]
    and not result["remedy_earned"]
)
assert result["optimizer_updates"] == result["validation_or_test_targets_scored"] == 0
assert not result["causal_explanation_established"]
assert sha("research/residual_geometry_plan.md") == result["plan_sha256"] == protocol["plan_sha256"]
assert all(sha(root / n) == h for n, h in result["files"].items())
with zipfile.ZipFile(root / "source.zip") as archive:
    for n, h in protocol["provenance"]["source_files"].items():
        assert sha(n) == h == hashlib.sha256(archive.read(n)).hexdigest()
    for n in ("research/residual_geometry_plan.md", "research/normalization_geometry.md"):
        assert archive.read(n) == Path(n).read_bytes()
prior = read("results/verification/overcomplete_screen_final_v1.json")
assert prior["status"] == "PASS" and not prior["earns_separate_longer_comparison"]
assert sha("results/overcomplete_screen_v1/result.json") == prior["result_sha256"]
old_sources = read("results/overcomplete_screen_v1/protocol.json")["provenance"]["source_files"]
assert all(sha(n) == h for n, h in old_sources.items())
assert (
    protocol["environment"] == read("results/overcomplete_screen_v1/preflight.json")["environment"]
)
phases = {}
for label in ("focused_tests", "preflight", "cpu", "gpu", "finish", "tests"):
    p = Path(f"results/verification/residual_geometry_{label}_v1.json")
    record = read(p)
    assert record["returncode"] == 0 and record["source_unchanged"]
    assert record["plan_sha256"] == result["plan_sha256"]
    assert all(sha(n) == h for n, h in record["source_files"].items())
    phases[label] = {"record": p.as_posix(), "sha256": sha(p), "seconds": record["seconds"]}
assert (
    "248 passed in 35.78s"
    in Path("results/verification/residual_geometry_tests_v1.log").read_text()
)
for n, h in protocol["data_hashes"].items():
    assert sha(Path("data/wikitext2_v1") / n) == h
for ref in protocol["references"].values():
    assert all(sha(Path(ref["path"]) / n) == h for n, h in ref["files"].items())
cpu = read(root / "cpu.json")
gpu = read(root / "gpu.json")
assert len(cpu["records"]) == 14 and len(gpu["records"]) == 6
assert (
    cpu["batch_sha256"] == gpu["batch_sha256"]
    and cpu["sampler_unchanged"]
    and gpu["sampler_unchanged"]
)
errors = []
for phase, data in (("cpu", cpu), ("gpu", gpu)):
    assert data["optimizer_updates"] == 0
    for row in data["records"].values():
        assert row["weights_and_cpu_rng_unchanged"] and row["all_parameter_gradients_finite"]
        assert len(row["norms"]) == 17 and len(row["ffns"]) == 8
        for norm in row["norms"].values():
            assert norm["local_vjp_relative_error"] <= 5e-5
            errors.append(norm["local_vjp_relative_error"])
            for key in (
                "input_rms",
                "inverse_scale",
                "unweighted_radial_gain",
                "local_adjoint_gain",
            ):
                values = norm[key]
                assert 0 <= values["p10"] <= values["median"] <= values["p90"]
                assert all(math.isfinite(v) for v in values.values())
    gains = {
        recipe: math.exp(
            sum(
                math.log(
                    data["records"][recipe + "_final"]["norms"][f"blocks.{i}.ffn_norm"][
                        "local_adjoint_gain"
                    ]["median"]
                )
                for i in range(1, 8)
            )
            / 7
        )
        for recipe in protocol["selected"]
    }
    assert gains == result["geometric_mean_later_ffn_norm_median_vjp_gain"][phase]
    ratios = {
        r: gains["overcomplete_headwise"] / gains[r] for r in gains if r != "overcomplete_headwise"
    }
    assert ratios == result["candidate_to_control_local_gain_ratio"][phase]
    assert gains["full_gelu"] < gains["overcomplete_headwise"] < gains["full_swiglu"]
assert len(errors) == 340
assert result["ordering_agrees_cpu_bf16"]
gauge = cpu["records"]["overcomplete_headwise_final"]["gauge_qualification"]
assert (
    gauge["logit_relative_error"]
    == gauge["loss_absolute_error"]
    == gauge["maximum_predicted_gradient_relative_error"]
    == 0
)
assert gauge["original_weights_restored_exactly"] and not gauge["modified_checkpoint_written"]
assert len(gauge["gradient_norm_ratios"]) == 16
assert all(
    v == (0.125 if n.endswith(".value") else 8.0) for n, v in gauge["gradient_norm_ratios"].items()
)
for row in cpu["records"].values():
    if "factor_geometry" in row:
        assert len(row["factor_geometry"]) == 8
        assert all(
            m["numerical_rank"] == 384
            for layer in row["factor_geometry"].values()
            for m in layer["maps"].values()
        )
plot = read("results/verification/residual_geometry_plot_v1.json")
assert plot["status"] == "PASS"
assert all(sha(n) == h for kind in ("inputs", "outputs") for n, h in plot[kind].items())
plans = {**prior["plans_unchanged"], "residual_geometry_v1": "residual_geometry_plan.md"}
for folder, name in plans.items():
    assert read(Path("results") / folder / "protocol.json")["plan_sha256"] == sha(
        Path("research") / name
    )
config = ast.parse(Path("src/core/config.py").read_text())
variants = next(
    ast.literal_eval(n.value)
    for n in config.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "VARIANTS" for t in n.targets)
)
assert len(variants) == 32 and not any("additive_block" in n for n in variants)
assert len(list(Path("results/runs").glob("*/metrics.json"))) == 168
budgets = {}
for d in (24, 384):
    h = 8 * d // 3
    rank = d // 8
    p = 3 * (d * h // 8 + rank * (d + h))
    assert p == 19 * d * d // 8 and 1 - p / (8 * d * d) == 0.703125
    budgets[str(d)] = {
        "width": d,
        "hidden": h,
        "groups": 8,
        "rank": rank,
        "ffn_parameters_per_layer": p,
    }
assert budgets["384"]["ffn_parameters_per_layer"] == 350208
files = [
    Path(n)
    for n in (
        "README.md",
        "research/CURRENT_STATE.md",
        "research/idea_bank.md",
        "research/residual_geometry_plan.md",
        "research/normalization_geometry.md",
        "research/residual_geometry_results.md",
        "research/additive_block_lowrank_proposal.md",
        "src/multihead_ffn/overcomplete.md",
    )
]
links = 0
for path in files:
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = target.strip("<>").split("#")[0]
        if not target or re.match(r"^[a-zA-Z]+://", target):
            continue
        assert (path.parent / unquote(target)).exists(), (path, target)
        links += 1
record = {
    "status": "PASS",
    "research_target_passes": False,
    "full_tests_passed": 248,
    "full_test_record": "results/verification/residual_geometry_tests_v1.json",
    "all_tested_sources_exact": True,
    "all_existing_computation_and_configs_unchanged": True,
    "retained_exact_signature_variants": 32,
    "registered_variants": 32,
    "retained_lm_profile_runs": 168,
    "cpu_probes": 14,
    "native_bf16_probes": 6,
    "additional_disposable_gauge_checks": 1,
    "optimizer_updates": 0,
    "validation_or_test_targets_scored": 0,
    "norm_vjps_verified": 340,
    "maximum_vjp_relative_error": max(errors),
    "gauge_logit_loss_and_predicted_gradient_errors": 0.0,
    "all_original_checkpoints_exact": True,
    "causal_explanation_established": False,
    "remedy_earned": False,
    "numerical_failures": 0,
    "process_failures": 0,
    "plans_unchanged": plans,
    "phases": phases,
    "local_links_checked": links,
    "figure_visually_inspected": True,
    "next_proposal_only_analytic_budgets_verified": budgets,
    "next_proposal_implemented_or_qualified": False,
    "result_sha256": sha(root / "result.json"),
    "source_archive_sha256": sha(root / "source.zip"),
    "documents": {p.as_posix(): sha(p) for p in files},
}
p = Path("results/verification/residual_geometry_final_v1.json")
assert not p.exists()
p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("documents", "plans_unchanged", "phases")}
    ),
    flush=True,
)
