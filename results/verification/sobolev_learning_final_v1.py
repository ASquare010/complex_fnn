"""Check H109 evidence, decisions and unchanged scientific sources; pack losslessly."""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path("results/sobolev_learning_v1")
DESTINATION = Path("results/verification/sobolev_learning_final_v1.json")
METRICS = ("reporting_mse", "derivative_relative_mse")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def run():
    protocol = read(ROOT / "protocol.json")
    for path, digest in protocol["sources"].items():
        assert sha(path) == digest, path
    previous = read("results/sobolev_selection_v1/protocol.json")
    assert protocol["provenance"]["source_files"] == previous["provenance"]["source_files"]
    for path, digest in protocol["provenance"]["source_files"].items():
        assert sha(path) == digest, path
    old_receipt = read("results/verification/sobolev_selection_final_v1.json")
    assert old_receipt["status"] == "PASS"
    for path, digest in old_receipt["files"].items():
        assert sha(path) == digest, path
    assert old_receipt["previous_maintained_tests_passed"] == 116
    assert read("results/verification/affine_residual_final_v1.json")["pytest_passed"] == 116
    for stage in ("study_recovery", "audit", "analyze", "plot", "inference_only"):
        assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0", stage
    assert (ROOT / "study_exit.txt").read_text().strip() == "1"
    before = read(ROOT / "preflight_recovery.json")
    assert (
        before["neural_updates_before_failure"] == 0
        and not before["protocol_written_before_failure"]
    )
    for path, digest in before["before"].items():
        assert sha(path) == digest, path
    q = read(ROOT / "qualification.json")
    assert q["passed"] and len(q["checks"]) == 4
    result, summary, audit = [
        read(ROOT / name) for name in ("result.json", "summary.json", "audit.json")
    ]
    rows, cases = result["rows"], result["cases"]
    assert result["status"] == "COMPLETE" and len(rows) == 72 and len(cases) == 18
    assert result["neural_updates"] == summary["neural_updates"] == 86400
    assert result["profile_updates"] == summary["profile_updates"] == 540
    assert not result["breakthrough"] and not summary["breakthrough"]
    assert audit["passed"] and audit["endpoint_states"] == 216 and audit["selected"] == 72
    assert (
        len(audit["scores"]) == 648
        and len(audit["training"]) == 144
        and len(audit["datasets"]) == 18
    )
    assert audit["audit_neural_updates"] == 0
    assert audit["max_score_difference"] == max(r["absolute_difference"] for r in audit["scores"])
    assert audit["max_score_difference"] < 2e-8
    assert all(
        d["exact_input_recapture"]
        and d["exact_smooth_targets"]
        and d["exact_stream_and_old_normalizers"]
        for d in audit["datasets"]
    )
    assert all(t["initial_loss_matches"] and t["finite_steps"] == 600 for t in audit["training"])
    lookup = {(r["teacher"], r["layer"], r["seed"], r["method"]): r for r in rows}
    assert len(lookup) == 72
    for r in rows:
        assert len(r["endpoints"]) == 3 and r["selected_index"] == 1 and r["all_finite"]
        close(r["parameter_reduction"], 1 - r["parameters"] / 1179648)
        assert r["parameter_reduction"] >= 0.7
        assert r["parameters"] == (344448 if r["teacher"] == "gelu" else 343680)
        assert r["selected_index"] == min(
            range(3), key=lambda i: (r["endpoints"][i]["selection_mse"], i)
        )
        for metric in ("selection_mse", *METRICS):
            assert r[metric] == r["endpoints"][1][metric]
    groups = {(g["teacher"], g["method"]): g for g in summary["groups"]}
    assert len(groups) == 8
    for key, g in groups.items():
        peers = [r for r in rows if (r["teacher"], r["method"]) == key]
        for metric, data in g.items():
            if isinstance(data, dict) and "sample_variance" in data:
                values = [r[metric] for r in peers]
                assert data == {
                    "mean": st.mean(values),
                    "median": st.median(values),
                    "sample_variance": st.variance(values),
                    "min": min(values),
                    "max": max(values),
                    "n": 9,
                }
        for metric in METRICS:
            close(
                g["selected_over_initial_ratios"][metric],
                math.prod(r[metric] / r["endpoints"][0][metric] for r in peers) ** (1 / 9),
            )
        for field, values, name in (
            ("seed", (71, 83, 97), "seed_means"),
            ("layer", (0, 3, 7), "depth_means"),
        ):
            for value in values:
                for metric in METRICS:
                    close(
                        g[name][str(value)][metric],
                        st.mean(r[metric] for r in peers if r[field] == value),
                    )
        assert g["training"]["clip_fraction"] == 0 and g["training"]["all_finite"]
    for method, decision in summary["components"].items():
        assert decision["verdict"] == "FIXED_INITIALIZER_LEARNING_RECIPE_CLOSED"
        for teacher, data in decision["teachers"].items():
            peers = [r for r in rows if (r["teacher"], r["method"]) == (teacher, method)]
            for label, endpoint_index in (("selected", None), ("0", 0), ("1", 1), ("2", 2)):
                for metric in METRICS:
                    values = []
                    for r in peers:
                        c = lookup[teacher, r["layer"], r["seed"], "value_greedy"]
                        a, b = (
                            (r, c)
                            if endpoint_index is None
                            else (r["endpoints"][endpoint_index], c["endpoints"][endpoint_index])
                        )
                        values.append(a[metric] / b[metric])
                    close(math.prod(values) ** (1 / 9), data["paired_ratios"][label][metric])
            for seed, ratios in data["seed_ratios"].items():
                for metric in METRICS:
                    values = [
                        r[metric] / lookup[teacher, r["layer"], r["seed"], "value_greedy"][metric]
                        for r in peers
                        if r["seed"] == int(seed)
                    ]
                    close(math.prod(values) ** (1 / 3), ratios[metric])
            selected = data["paired_ratios"]["selected"]
            assert data["gates"]["five_percent_derivative_gain"] == (
                selected["derivative_relative_mse"] <= 0.95
            )
            assert not data["gates"]["five_percent_derivative_gain"]
            assert data["gates"]["at_most_one_percent_value_regression"] == (
                selected["reporting_mse"] <= 1.01
            )
            assert data["gates"]["every_seed_meets_both"] == all(
                v["reporting_mse"] <= 1.01 and v["derivative_relative_mse"] <= 0.95
                for v in data["seed_ratios"].values()
            )
            assert data["passes"] == all(data["gates"].values())
            assert not data["passes"]
    for method, decision in summary["allocation"].items():
        assert decision["verdict"] == "NO_LANGUAGE_ALLOCATION"
        assert decision["component_eligible"] == (method == "value_greedy")
        for teacher, data in decision["teachers"].items():
            g = groups[teacher, method]
            assert data["gates"]["mean_value_error"] == (g["reporting_mse"]["mean"] <= 0.05)
            assert not data["gates"]["mean_value_error"]
            assert data["gates"]["mean_derivative_error"] == (
                g["derivative_relative_mse"]["mean"] <= 0.1
            )
            assert not data["gates"]["mean_derivative_error"]
            refs = [c for c in cases if c["teacher"] == teacher]
            close(
                data["ratios"]["training_memory"],
                g["peak_allocated_bytes"]["mean"]
                / st.mean(c["full_profile"]["peak_allocated_bytes"] for c in refs),
            )
            close(
                data["ratios"]["inference_memory"],
                g["inference"]["peak_allocated_bytes"]["mean"]
                / st.mean(c["full_inference"]["peak_allocated_bytes"] for c in refs),
            )
            close(
                data["ratios"]["inference_time"],
                g["inference"]["inference_ms"]["mean"]
                / st.mean(c["full_inference"]["inference_ms"] for c in refs),
            )
            assert data["passes"] == all(data["gates"].values())
            assert not data["passes"]
    with gzip.open(ROOT / "metrics.csv.gz", "rt", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 216 and sum(r["selected"] == "True" for r in table) == 72
    for record in table:
        r = lookup[record["teacher"], int(record["layer"]), int(record["seed"]), record["method"]]
        endpoint = next(e for e in r["endpoints"] if e["rate"] == float(record["rate"]))
        for metric in ("selection_mse", *METRICS):
            assert float(record[metric]) == endpoint[metric]
        assert record["checkpoint_sha256"] == endpoint["checkpoint"]["sha256"]
    diagnostics = json.loads(gzip.decompress((ROOT / "diagnostics.json.gz").read_bytes()))
    assert len(diagnostics) == 144
    deployment = read(ROOT / "inference_only.json")
    assert deployment["posthoc"] and deployment["original_gates_unchanged"]
    assert (
        not deployment["gradients_or_optimizers_in_process"] and deployment["neural_updates"] == 0
    )
    assert len(deployment["students"]) == 72 and len(deployment["full_teachers"]) == 18
    for r in deployment["students"]:
        expected = lookup[r["teacher"], r["layer"], r["seed"], r["method"]]
        assert r["checkpoint"] == expected["endpoints"][1]["checkpoint"]
        assert r["parameters"] == expected["parameters"]
    packed = {}
    for name in ("result.json", "audit.json", "study.log", "study_recovery.log", "audit.log"):
        original = ROOT / name
        target = original.with_name(original.name + ".gz")
        target.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == original.read_bytes()
        packed[str(target)] = {
            "sha256": sha(target),
            "original_sha256": sha(original),
            "bytes": target.stat().st_size,
        }
    documents = [Path("research/sobolev_learning_results.md"), ROOT / "source/README.md"]
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if not link.startswith("https:"):
                target = (document.parent / link).resolve()
                assert target == DESTINATION.resolve() or target.exists(), (document, link)
    public = [
        *documents,
        Path("research/sobolev_learning_plan.md"),
        Path("research/figures/sobolev_learning.png"),
        ROOT / "protocol.json",
        ROOT / "summary.json",
        ROOT / "qualification.json",
        ROOT / "data_manifest.json",
        ROOT / "metrics.csv.gz",
        ROOT / "diagnostics.json.gz",
        ROOT / "inference_only.json",
        ROOT / "failure.json",
        ROOT / "preflight_recovery.json",
        ROOT / "preflight_failure/study.py",
        *sorted((ROOT / "source").glob("*.py")),
        Path(__file__),
    ]
    assert all(p.stat().st_size < 5 * 2**20 for p in public + [Path(p) for p in packed])
    receipt = {
        "status": "PASS",
        "frozen_sources_verified": len(protocol["sources"]),
        "maintained_files_unchanged": len(protocol["provenance"]["source_files"]),
        "qualification_groups": 4,
        "fresh_fits": 144,
        "neural_updates": 86400,
        "profile_updates": 540,
        "endpoint_states": 216,
        "independent_metrics": 648,
        "selected_models": 72,
        "exact_recaptured_datasets": 18,
        "max_score_difference": audit["max_score_difference"],
        "audit_neural_updates": 0,
        "both_initializer_learning_recipes_closed": True,
        "all_language_allocations_unearned": True,
        "preflight_failure_preserved": True,
        "scientific_training_repeated": False,
        "previous_maintained_tests_passed": 116,
        "maintained_tests_rerun_in_h109": False,
        "fresh_process_inference_profiles": 90,
        "broad_goal_achieved": False,
        "packed": packed,
        "files": {str(p): sha(p) for p in public},
    }
    DESTINATION.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("packed", "files")}, indent=2))


if __name__ == "__main__":
    run()
