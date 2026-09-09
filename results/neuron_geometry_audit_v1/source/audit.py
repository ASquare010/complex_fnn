"""Independently reduce H090 evidence and rescore checkpoints; never train."""

import hashlib
import itertools
import json
import math
import statistics as stats
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import equal_payload, read, sha, write_json
from results.neuron_geometry_recovery_v1.source.recovery import PrecisionFFN

ROOT = Path("results/neuron_geometry_recovery_v1")
OUT = Path("results/neuron_geometry_audit_v1")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS, RATES = (17, 29, 43), (0.001, 0.003)
FORMS = (
    "full_gelu",
    "full_relu",
    "full_swiglu",
    "narrow_gelu",
    "narrow_relu",
    "narrow_swiglu",
    "groupsort",
    "gelu_offset",
    "twist_fixed",
    "twist_learned",
    "twist_identity",
    "bezier_1p",
    "bump_2p",
    "linear",
)
CANDIDATES = ("twist_learned", "twist_identity", "bezier_1p", "bump_2p")
CONTROLS = ("narrow_gelu", "narrow_relu", "narrow_swiglu", "groupsort", "gelu_offset")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def close_structure(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            close_structure(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            close_structure(a, b)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def tensor_file(path, expected_hash=None):
    if expected_hash is not None:
        assert sha(path) == expected_hash, path
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    return torch.load(path, map_location="cpu", weights_only=True)


def audit_predecessors():
    root = Path("results/neuron_precision_v1")
    result = read(root / "result.json")
    before = read(root / "before.json")
    assert result["evaluations"] == 32 and result["optimizer_updates"] == 0
    for row in result["rows"]:
        modes = tensor_file(row["tensor_file"], row["tensor_sha256"])
        for key, first, second, tol in (
            ("original_reference", "native", "reference", row["atol"]),
            ("island_reference", "island", "reference_island", row["atol"]),
            ("native_vs_island", "native", "island", 0),
        ):
            independently_reduced = {}
            for name, value in modes[first].items():
                other = modes[second][name]
                independently_reduced[name] = {
                    "elements": value.numel(),
                    "mismatches": int((~torch.isclose(value, other, atol=tol, rtol=tol)).sum()),
                    "max_abs": (value.double() - other.double()).abs().max().item(),
                    "exact": torch.equal(value, other),
                }
            assert independently_reduced == row[key]
        assert all(v["mismatches"] == 0 for v in row["island_reference"].values())
        if row["precision"] == "fp32":
            assert all(v["exact"] for v in row["native_vs_island"].values())
    for name, record in before["h088_files"].items():
        path = Path(name)
        assert sha(path) == record["sha256"]
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (
            record["bytes"],
            record["mtime_ns"],
        )
    assert not Path("results/neuron_geometry_v1/data.pt").exists()
    assert not list(Path("results/neuron_geometry_v1/cells").iterdir())
    return {
        "diagnostic_payloads": 8,
        "diagnostic_tensor_comparisons": 120,
        "fp32_unchanged_pairs": 20,
        "h088_files_preserved": len(before["h088_files"]),
    }


def audit_qualification():
    observations, payloads, reference_pairs, checkpoint_pairs = 0, 0, 0, 0
    for path in (ROOT / "qualification").glob("*.json"):
        row = read(path)
        assert row["status"] == "PASS" and row["optimizer_updates"] == 0
        observations += 1
        if "tensor_file" not in row:
            continue
        data = tensor_file(path.parent / row["tensor_file"], row["tensor_sha256"])
        payloads += 1
        if "actual" in data:
            for a, b in zip(data["actual"], data["expected"]):
                tolerance = (0.02, 0.02) if row["kind"] == "cuda" else (1e-11, 1e-10)
                torch.testing.assert_close(a, b, atol=tolerance[0], rtol=tolerance[1])
                reference_pairs += 1
            for a, b in zip(data["actual"], data.get("checkpoint", [])):
                assert torch.equal(a, b)
                checkpoint_pairs += 1
        if row["kind"] == "geometry":
            origin = torch.tensor([row["center"], 0], dtype=torch.float64)
            torch.testing.assert_close(data["inverse"], data["input"], atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(
                (data["output"] - origin).square().sum(-1),
                (data["input"] - origin).square().sum(-1),
                atol=1e-10,
                rtol=1e-10,
            )
            assert (torch.linalg.det(data["jacobian"]) - 1).abs().max() <= 1e-10
            s = torch.linalg.svdvals(data["jacobian"])
            torch.testing.assert_close(s, data["singular_values"], atol=1e-10, rtol=1e-10)
            assert s.min() >= math.sqrt(2) - 1 - 1e-10 and s.max() <= math.sqrt(2) + 1 + 1e-10
    for path in Path("results/neuron_geometry_v1/qualification").glob("*.json"):
        old, new = read(path), read(ROOT / "qualification" / path.name)
        if "tensor_file" in old:
            equal_payload(
                tensor_file(path.parent / old["tensor_file"], old["tensor_sha256"]),
                tensor_file(ROOT / "qualification" / new["tensor_file"], new["tensor_sha256"]),
            )
            old.pop("tensor_sha256")
            new.pop("tensor_sha256")
        assert old == new
    assert (observations, payloads, reference_pairs, checkpoint_pairs) == (27, 11, 34, 20)
    return {
        "observations": observations,
        "tensor_payloads": payloads,
        "independent_reference_pairs": reference_pairs,
        "exact_checkpoint_pairs": checkpoint_pairs,
    }


def audit_data(data):
    x = (2 * torch.rand(73728, 384, generator=torch.Generator().manual_seed(9814)) - 1) * math.sqrt(
        3
    )
    assert torch.equal(x, data["x"])
    permutation = torch.randperm(384, generator=torch.Generator().manual_seed(9283))
    rotation = torch.linalg.qr(
        torch.randn(384, 384, generator=torch.Generator().manual_seed(9282))
    ).Q
    assert torch.equal(permutation, data["permutation"]) and torch.equal(rotation, data["rotation"])
    a, b, c, d = (torch.roll(x[:, permutation], -i, -1) for i in range(4))
    formulas = (
        lambda: torch.sin(2 * a) + b.square() + torch.exp(0.5 * c) - d,
        lambda: torch.sin(3 * a) * torch.cos(2 * b) + 0.5 * torch.sin(4 * c + d),
        lambda: a * b + 2 * b * c * d + a.square() * d,
        lambda: F.relu(a + b) - 0.7 * (c - d).abs() + torch.where(a > 0, b, c),
    )
    for task, formula in zip(TASKS, formulas):
        rotated = formula() @ rotation
        scale = rotated[:65536].std(0, correction=0)
        assert torch.equal(scale, data["scales"][task])
        assert torch.equal(rotated / scale, data["targets"][task])
    for seed in SEEDS:
        stream = torch.randint(
            65536, (600, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        assert torch.equal(stream, data["streams"][seed])


@torch.inference_mode()
def rescore(model, x, y):
    accumulated = 0.0
    for begin in range(0, x.shape[0], 256):
        accumulated += F.mse_loss(
            model(x[begin : begin + 256]), y[begin : begin + 256], reduction="sum"
        ).item()
    return accumulated / y.numel()


def reduce_summary(selected, rows, zero):
    lookup = {(r["task"], r["form"], r["seed"]): r for r in selected}
    rate_lookup = {(r["task"], r["form"], r["seed"], r["rate"]): r for r in rows}

    def ratio(form, reference, seeds=SEEDS, rate=None):
        pairs = ((task, seed) for task in TASKS for seed in seeds)
        if rate is None:
            values = [
                lookup[t, form, s]["reporting_mse"] / lookup[t, reference, s]["reporting_mse"]
                for t, s in pairs
            ]
        else:
            values = [
                rate_lookup[t, form, s, rate]["reporting_mse"]
                / rate_lookup[t, reference, s, rate]["reporting_mse"]
                for t, s in pairs
            ]
        return math.exp(math.fsum(map(math.log, values)) / len(values))

    per_task, resources = {}, {}
    for form in FORMS:
        per_task[form] = {}
        for task in TASKS:
            values = [lookup[task, form, seed]["reporting_mse"] for seed in SEEDS]
            mean = math.fsum(values) / 3
            variance = math.fsum((v - mean) ** 2 for v in values) / 2
            per_task[form][task] = {
                "mean": mean,
                "median": sorted(values)[1],
                "variance": variance,
                "sample_sd": math.sqrt(variance),
                "values": values,
            }
        subset = [lookup[t, form, s] for t in TASKS for s in SEEDS]
        resources[form] = {
            f"mean_median_{phase}_ms": math.fsum(r[f"median_{phase}_ms"] for r in subset) / 12
            for phase in ("update", "forward", "backward")
        }
        resources[form].update(
            max_peak_bytes=max(r["peak_bytes"] for r in subset),
            mean_clipped_fraction=math.fsum(r["clipped_fraction"] for r in subset) / 12,
            upper_rate_selected=sum(r["rate"] == 0.003 for r in subset),
        )
    assay = sum(per_task["full_gelu"][t]["mean"] <= 0.98 * zero[t] for t in TASKS) >= 2
    decisions = {}
    for form in CANDIDATES:
        ratios = {
            ref: ratio(form, ref) for ref in (*CONTROLS, "twist_fixed", "full_gelu", "full_swiglu")
        }
        by_seed = {ref: [ratio(form, ref, (s,)) for s in SEEDS] for ref in CONTROLS}
        gates = {
            "positive_assay": assay,
            "complete_finite_grid": len(rows) == 336 and all(r["all_finite"] for r in rows),
            "seventy_percent_reduction": 1
            - sum(p.numel() for p in PrecisionFFN(form).parameters()) / 1179648
            >= 0.7,
            "two_percent_over_each_narrow_control": all(ratios[r] <= 0.98 for r in CONTROLS),
            "every_seed_better_than_each_control": all(
                v < 1 for row in by_seed.values() for v in row
            ),
            "per_task_regression_cap": all(
                per_task[form][t]["mean"] <= 1.05 * min(per_task[r][t]["mean"] for r in CONTROLS)
                for t in TASKS
            ),
            "learned_twist_beats_fixed": not form.startswith("twist_")
            or ratios["twist_fixed"] <= 0.99,
        }
        decisions[form] = {
            "gates": gates,
            "selected_ratios": ratios,
            "seed_ratios": by_seed,
            "matched_rate_ratios": {
                str(rate): {ref: ratio(form, ref, rate=rate) for ref in CONTROLS} for rate in RATES
            },
            "verdict": "EARNS_RESOURCE_STUDY" if all(gates.values()) else "REJECTED_AT_THIS_BUDGET",
        }
    return {
        "positive_assay": assay,
        "per_task": per_task,
        "resources": resources,
        "decisions": decisions,
        "all_ratios_vs_narrow_gelu": {form: ratio(form, "narrow_gelu") for form in FORMS},
    }


def main():
    before = read(OUT / "before.json")
    assert all(sha(p) == h for p, h in before["source_hashes"].items())
    assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
    for phase in ("collection", "qualification", "fitting"):
        process = read(ROOT / f"{phase}_process.json")
        assert process["status"] == "PASS" and process["returncode"] == 0
        assert sha(ROOT / f"{phase}.log") == process["log_sha256"]
    result, protocol = read(ROOT / "result.json"), read(ROOT / "protocol.json")
    assert (result["runs"], result["updates"], result["training_example_presentations"]) == (
        336,
        201600,
        51609600,
    )
    assert sha(ROOT / "protocol.json") == result["protocol_sha256"]
    assert sha(ROOT / "source.zip") == result["source_archive_sha256"]
    for p, h in {**protocol["source_hashes"], **protocol["anchors"]}.items():
        assert sha(p) == h
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.testzip() is None
        for p, h in {**protocol["source_hashes"], **protocol["anchors"]}.items():
            assert hashlib.sha256(archive.read(p)).hexdigest() == h
    predecessors, qualification = audit_predecessors(), audit_qualification()
    data = tensor_file(ROOT / "data.pt", result["data_sha256"])
    audit_data(data)
    manifest = read(ROOT / "data_manifest.json")
    assert manifest["sha256"] == result["data_sha256"]
    assert (
        manifest["train_rows"],
        manifest["selection_rows"],
        manifest["reporting_rows"],
        manifest["width"],
        manifest["batch"],
        manifest["steps"],
    ) == (65536, 4096, 4096, 384, 256, 600)
    assert manifest["streams"] == {str(s): digest(data["streams"][s]) for s in SEEDS}
    for p, h in {**result["raw_cell_record_hashes"], **result["plot_hashes"]}.items():
        assert sha(p) == h
    rows = result["all_rates"]
    assert len(rows) == 336 and len(result["selected"]) == 168
    expected = set(itertools.product(TASKS, FORMS, SEEDS, RATES))
    assert {(r["task"], r["form"], r["seed"], r["rate"]) for r in rows} == expected
    assert len(list((ROOT / "cells").iterdir())) == 336
    assert len(list((ROOT / "selections").glob("*.json"))) == 168
    assert result["selected"] == [r for r in rows if r["selected"]]
    for task, form, seed in itertools.product(TASKS, FORMS, SEEDS):
        subset = [r for r in rows if (r["task"], r["form"], r["seed"]) == (task, form, seed)]
        assert [r["rate"] for r in subset] == list(RATES)
        chosen = min(subset, key=lambda r: r["selection_mse"])
        selection = read(ROOT / "selections" / f"{task}_{form}_s{seed}.json")
        assert selection == {
            "selected_label": chosen["label"],
            "selected_rate": chosen["rate"],
            "selection_mse_by_rate": {str(r["rate"]): r["selection_mse"] for r in subset},
        }
        assert all(r["selected"] == (r["label"] == chosen["label"]) for r in subset)
    for ti, task in enumerate(TASKS):
        for si, seed in enumerate(SEEDS):
            offset = (ti * 3 + si) % 14
            assert result["form_orders"][f"{task}_{seed}"] == list(FORMS[offset:] + FORMS[:offset])
    endpoints = []
    for task in TASKS:
        x, y = data["x"][65536:].cuda(), data["targets"][task][65536:].cuda()
        for row in (r for r in rows if r["task"] == task):
            cell = ROOT / "cells" / row["label"]
            training = read(cell / "training.json")
            assert read(cell / "reporting.json") == row
            for key in row.keys() - {"selected", "reporting_mse"}:
                assert row[key] == training[key]
            model = PrecisionFFN(row["form"], row["seed"])
            assert {n: digest(v) for n, v in model.state_dict().items()} == training[
                "initial_state_sha256"
            ]
            count = sum(p.numel() for p in model.parameters())
            controls = sum(p.numel() for p in model.curve.parameters())
            assert count == row["parameters"] and controls == row["activation_parameters"]
            assert row["matrix_forward_flops_per_example"] == 2 * (count - controls)
            groups = [
                {
                    "name": g["parameter_name"],
                    "lr": g["lr"],
                    "parameters": sum(p.numel() for p in g["params"]),
                }
                for g in model.optimizer_groups(row["rate"])
            ]
            assert groups == training["optimizer_groups"]
            history = training["history"]
            assert len(history) == 600 and [r["step"] for r in history] == list(range(1, 601))
            assert all(math.isfinite(v) for r in history for v in r.values())
            assert all(
                r["preclip_norm"] >= 0
                and r["loss"] >= 0
                and r["update_ms"] >= r["forward_ms"] + r["backward_ms"] - 1e-9
                for r in history
            )
            for phase in ("update", "forward", "backward"):
                assert (
                    stats.median(r[f"{phase}_ms"] for r in history[50:])
                    == row[f"median_{phase}_ms"]
                )
            assert sum(r["preclip_norm"] > 1 for r in history) / 600 == row["clipped_fraction"]
            assert [r["step"] for r in training["diagnostics"]] == [0, 50, 150, 300, 600]
            assert all(
                r[k]["finite"]
                for r in training["diagnostics"]
                for k in ("preactivation", "activation")
            )
            checkpoint = tensor_file(row["checkpoint"], row["checkpoint_sha256"])
            for key in (
                "task",
                "form",
                "seed",
                "rate",
                "data_sha256",
                "protocol_sha256",
                "optimizer_steps",
            ):
                assert checkpoint[key] == row[key]
            assert checkpoint["step"] == 600 and checkpoint["resume_supported"] is False
            assert checkpoint["stream_sha256"] == digest(data["streams"][row["seed"]])
            assert set(checkpoint["optimizer_steps"]) == {600}
            assert len(checkpoint["optimizer_steps"]) == len(list(model.parameters()))
            assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.cuda()
            if hasattr(model.curve, "shape_parameters"):
                assert model.curve.shape_parameters() == training["diagnostics"][-1]["shape"]
                assert (
                    model.curve.theta.detach().cpu().tolist()
                    == training["diagnostics"][-1]["theta"]
                )
            selection, reporting = (
                rescore(model, x[:4096], y[:4096]),
                rescore(model, x[4096:], y[4096:]),
            )
            assert selection == row["selection_mse"] and reporting == row["reporting_mse"], row[
                "label"
            ]
            endpoints.append(
                {"label": row["label"], "selection_mse": selection, "reporting_mse": reporting}
            )
            del model, checkpoint
        del x, y
        print(json.dumps({"task": task, "checkpoints_audited": len(endpoints)}), flush=True)
    zero = {t: data["targets"][t][69632:].double().square().mean().item() for t in TASKS}
    assert zero == result["zero_mse"]
    summary = reduce_summary(result["selected"], rows, zero)
    close_structure(summary, result["summary"])
    assert all(sha(p) == h for p, h in before["source_hashes"].items())
    write_json(
        OUT / "result.json",
        {
            "status": "PASS",
            "training_runs_audited": 336,
            "selection_scores_exact": 336,
            "reporting_scores_exact": 336,
            "selections_verified": 168,
            "histories_verified": 336,
            "summaries_independently_reduced": True,
            "qualification": qualification,
            "predecessors": predecessors,
            "data_regeneration_exact": True,
            "initializations_exact": 336,
            "source_hashes_preserved": len(protocol["source_hashes"]),
            "h090_result_sha256": sha(ROOT / "result.json"),
            "audit_before_sha256": sha(OUT / "before.json"),
            "research_goal_achieved": False,
            "decisions": summary["decisions"],
            "endpoints": endpoints,
        },
    )


if __name__ == "__main__":
    main()
