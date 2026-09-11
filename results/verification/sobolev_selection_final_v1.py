"""Verify H108's decisions/provenance and losslessly package complete evidence."""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path("results/sobolev_selection_v1")
PREVIOUS = Path("results/affine_residual_fit_v1")
DESTINATION = Path("results/verification/sobolev_selection_final_v1.json")
METRICS = ("value_nmse", "derivative_relative_mse")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


def run():
    protocol = read(ROOT / "protocol.json")
    previous = read(PREVIOUS / "protocol.json")
    frozen = {}
    for p in (protocol, previous, read("results/affine_residual_capacity_v1/protocol.json")):
        for path, digest in p["sources"].items():
            assert sha(path) == digest, path
            frozen[path] = digest
    maintained = protocol["provenance"]["source_files"]
    assert maintained == previous["provenance"]["source_files"]
    for path, digest in maintained.items():
        assert sha(path) == digest, path
    prior_receipt = read("results/verification/affine_residual_final_v1.json")
    assert prior_receipt["status"] == "PASS" and prior_receipt["pytest_passed"] == 116
    assert "116 passed" in (PREVIOUS / "tests_recovery.log").read_text()
    assert (PREVIOUS / "tests_recovery_exit.txt").read_text().strip() == "0"

    qualified = read(ROOT / "qualification.json")
    assert qualified["passed"] and len(qualified["checks"]) == 13
    result, summary, audit, ablation = (
        read(ROOT / name) for name in ("result.json", "summary.json", "audit.json", "ablation.json")
    )
    rows, cases = result["rows"], result["cases"]
    assert result["status"] == "COMPLETE" and len(rows) == 432 and len(cases) == 18
    assert result["neural_updates"] == summary["neural_updates"] == audit["neural_updates"] == 0
    assert not result["breakthrough"] and not summary["breakthrough"]
    assert all(r["all_finite"] for r in rows)
    assert audit["passed"] and len(audit["statistics_cases"]) == 18
    assert len(audit["scores"]) == audit["score_count"] == 864
    assert len(audit["greedy_marginals"]) == audit["checkpoints"] == 432
    assert all(
        c["autograd_reference_values_and_derivatives"] and c["orders"]
        for c in audit["statistics_cases"]
    )
    assert all(c["maximal_gain_or_variance"] for c in audit["greedy_marginals"])
    assert audit["max_score_difference"] == max(s["absolute_difference"] for s in audit["scores"])
    assert audit["max_score_difference"] < 1e-8 and audit["max_gain_difference"] < 1e-12
    assert audit["max_gain_difference"] == max(
        s["gain_difference"] for s in audit["greedy_marginals"]
    )

    before = read(ROOT / "audit_recovery_before.json")
    assert not before["training_repeated"]
    for path, digest in before["files"].items():
        assert sha(path) == digest, path
    for name in ("study", "audit_recovery", "analyze", "plot", "ablation"):
        assert (ROOT / f"{name}_exit.txt").read_text().strip() == "0", name
    assert (ROOT / "audit_exit.txt").read_text().strip() == "1"
    assert (ROOT / "audit_failure.json").is_file()

    lookup = {(r["teacher"], r["layer"], r["seed"], r["budget"], r["method"]): r for r in rows}
    assert len(lookup) == 432
    for r in rows:
        matrices = 2 if r["teacher"] == "gelu" else 3
        assert r["parameters"] == matrices * 384 * r["hidden"] + 384
        assert r["parameter_bytes"] == 4 * r["parameters"]
        close(r["parameter_reduction"], 1 - r["parameters"] / 1179648)
        assert (r["parameter_reduction"] >= 0.7) == (r["budget"] == "primary")
        assert len(set(r["selected"])) == r["hidden"]
        assert r["solve"]["stationarity_max"] < 1e-8
    groups = {(r["teacher"], r["budget"], r["method"]): r for r in summary["groups"]}
    assert len(groups) == 48
    for key, group in groups.items():
        peers = [r for r in rows if (r["teacher"], r["budget"], r["method"]) == key]
        assert len(peers) == 9
        for metric, description in group.items():
            if isinstance(description, dict) and "sample_variance" in description:
                values = [r[metric] for r in peers]
                assert description == {
                    "mean": st.mean(values),
                    "median": st.median(values),
                    "sample_variance": st.variance(values),
                    "min": min(values),
                    "max": max(values),
                    "n": 9,
                }
        for category, values, field in (
            ("seed_means", (71, 83, 97), "seed"),
            ("depth_means", (0, 3, 7), "layer"),
        ):
            for value in values:
                for metric in METRICS:
                    close(
                        group[category][str(value)][metric],
                        st.mean(r[metric] for r in peers if r[field] == value),
                    )

    for budget, decision in summary["component_decisions"].items():
        assert decision["verdict"] == "COMPONENT_QUALIFIES"
        for teacher, data in decision["teachers"].items():
            peers = [
                r
                for r in rows
                if (r["teacher"], r["budget"], r["method"]) == (teacher, budget, "sobolev_greedy")
            ]
            for seed in (None, 71, 83, 97):
                subset = [r for r in peers if seed is None or r["seed"] == seed]
                recorded = (
                    data["ratios_to_value_greedy"]
                    if seed is None
                    else data["seed_ratios"][str(seed)]
                )
                for metric in METRICS:
                    independent = math.prod(
                        r[metric]
                        / lookup[teacher, r["layer"], r["seed"], budget, "value_greedy"][metric]
                        for r in subset
                    ) ** (1 / len(subset))
                    close(independent, recorded[metric])
                    assert independent <= (1.01 if metric == "value_nmse" else 0.95)
            dominators = []
            candidate = groups[teacher, budget, "sobolev_greedy"]
            for method in ("random", "weight_norm", "fluctuation"):
                control = groups[teacher, budget, method]
                if all(control[k]["mean"] <= candidate[k]["mean"] for k in METRICS) and any(
                    control[k]["mean"] < candidate[k]["mean"] for k in METRICS
                ):
                    dominators.append(method)
            assert data["static_dominators"] == dominators == []
            assert data["passes"] and all(data["gates"].values())

    for budget, methods in summary["allocation"].items():
        for method, decision in methods.items():
            assert (
                decision["verdict"] == "NO_LANGUAGE_ALLOCATION" and decision["component_eligible"]
            )
            assert decision["original_seventy_percent_parameter_target"] == (budget == "primary")
            actual_passes = []
            for teacher, data in decision["teachers"].items():
                g = groups[teacher, budget, method]
                references = [c["teacher_resources"] for c in cases if c["teacher"] == teacher]
                memory = g["peak_allocated_bytes"]["mean"] / st.mean(
                    c["peak_allocated_bytes"] for c in references
                )
                latency = g["inference_ms"]["mean"] / st.mean(c["inference_ms"] for c in references)
                close(memory, data["memory_ratio_to_full"])
                close(latency, data["latency_ratio_to_full"])
                gates = {
                    "mean_value_error_below_five_percent": g["value_nmse"]["mean"] <= 0.05,
                    "mean_derivative_error_below_ten_percent": g["derivative_relative_mse"]["mean"]
                    <= 0.1,
                    "every_seed_meets_both_absolute_errors": all(
                        s["value_nmse"] <= 0.05 and s["derivative_relative_mse"] <= 0.1
                        for s in g["seed_means"].values()
                    ),
                    "ten_percent_inference_memory_saving": memory <= 0.9,
                    "latency_within_ten_percent": latency <= 1.1,
                    "at_least_twenty_percent_fewer_parameters": g["parameter_reduction"] >= 0.2,
                }
                assert gates == data["gates"]
                assert data["passes"] == all(gates.values())
                actual_passes.append(all(gates.values()))
            assert not all(actual_passes)

    with gzip.open(ROOT / "metrics.csv.gz", "rt", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 432
    for record in table:
        row = lookup[
            record["teacher"],
            int(record["layer"]),
            int(record["seed"]),
            record["budget"],
            record["method"],
        ]
        for key, value in record.items():
            if isinstance(row[key], str):
                assert value == row[key]
            else:
                assert float(value) == row[key]

    fractions = []
    assert ablation["posthoc_attribution_of_predeclared_ablations"] and len(ablation["rows"]) == 6
    for row in ablation["rows"]:
        teacher, budget = row["teacher"], row["budget"]
        for metric, values in row["metrics"].items():
            a, b, c, d = [
                groups[teacher, budget, method][metric]["mean"]
                for method in (
                    "value_greedy",
                    "value_select_sobolev_fit",
                    "sobolev_select_value_fit",
                    "sobolev_greedy",
                )
            ]
            for key, value in zip(
                ("baseline", "readout_only", "selection_only", "both"), (a, b, c, d)
            ):
                close(values[key], value)
            close(values["readout_fraction_of_combined_absolute_gain"], (a - b) / (a - d))
            close(values["selection_fraction_of_combined_absolute_gain"], (a - c) / (a - d))
            close(values["interaction"], d - b - c + a)
            if metric == "derivative_relative_mse":
                fractions.append((a - b) / (a - d))
    assert 0.883 < min(fractions) < max(fractions) < 0.902

    packed = {}
    for name in ("result.json", "audit.json", "study.log", "audit.log", "audit_recovery.log"):
        original = ROOT / name
        target = original.with_name(original.name + ".gz")
        target.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == original.read_bytes()
        packed[str(target)] = {
            "sha256": sha(target),
            "original_sha256": sha(original),
            "bytes": target.stat().st_size,
        }

    report = Path("research/sobolev_selection_results.md")
    guide = ROOT / "source/README.md"
    for document in (report, guide):
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if not link.startswith("https:"):
                target = (document.parent / link).resolve()
                assert target == DESTINATION.resolve() or target.exists(), (document, link)
    public = [
        report,
        guide,
        Path("research/sobolev_selection_plan.md"),
        Path("research/figures/sobolev_selection.png"),
        ROOT / "protocol.json",
        ROOT / "qualification.json",
        ROOT / "summary.json",
        ROOT / "ablation.json",
        ROOT / "metrics.csv.gz",
        ROOT / "audit_failure.json",
        ROOT / "audit_recovery_before.json",
        *sorted((ROOT / "source").glob("*.py")),
        Path(__file__),
    ]
    for path in public + [Path(p) for p in packed]:
        assert path.stat().st_size < 5 * 2**20, path
    receipt = {
        "status": "PASS",
        "h108_frozen_sources_verified": len(protocol["sources"]),
        "total_h106_h107_h108_frozen_files_verified": len(frozen),
        "maintained_files_unchanged_since_h107": len(maintained),
        "qualification_groups": 13,
        "compressed_models": 432,
        "independent_scores": 864,
        "independent_statistics_cases": 18,
        "independent_export_readout_checks": 432,
        "sampled_actual_greedy_gain_checks": 432,
        "max_score_difference": audit["max_score_difference"],
        "max_gain_difference": audit["max_gain_difference"],
        "neural_updates": 0,
        "component_passes_all_three_widths_both_teachers": True,
        "all_six_language_allocations_unearned": True,
        "readout_fraction_of_combined_derivative_gain_range": [min(fractions), max(fractions)],
        "original_audit_failure_preserved": True,
        "audit_recovery_passed": True,
        "training_repeated": False,
        "previous_maintained_tests_passed": 116,
        "maintained_tests_rerun_in_h108": False,
        "previous_test_receipt": "results/verification/affine_residual_final_v1.json",
        "broad_goal_achieved": False,
        "packed": packed,
        "public_bytes_excluding_this_receipt": sum(p.stat().st_size for p in public)
        + sum(v["bytes"] for v in packed.values()),
        "files": {str(p): sha(p) for p in public},
    }
    DESTINATION.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("packed", "files")}, indent=2))


if __name__ == "__main__":
    run()
