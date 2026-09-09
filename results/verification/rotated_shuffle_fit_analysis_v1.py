"""Audit H071 fitting artifacts and independently rescore selected checkpoints on CPU."""

import json
import math
import statistics
from pathlib import Path

import torch

from results.rotated_shuffle_fit_v1.source.study import (
    COUNTS,
    FORMS,
    SEEDS,
    TASKS,
    finite_tree,
    make_data,
    make_model,
    score,
    state_hashes,
    tensor_sha,
)
from src.core.reproducibility import sha256, write_json

torch.set_num_threads(4)

ROOT = Path("results/rotated_shuffle_fit_v1")
OUT = Path("results/verification/rotated_shuffle_fit_analysis_v1.json")
assert not OUT.exists()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


process, protocol, result = [
    read(ROOT / n) for n in ("fitting_process.json", "protocol.json", "result.json")
]
assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
assert sha256(ROOT / "fitting.log") == process["log_sha256"]
assert sha256(ROOT / "protocol.json") == process["protocol_sha256"]
assert sha256(ROOT / "source.zip") == process["source_zip_sha256"]
assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
manifest = read(ROOT / "data_manifest.json")
assert sha256(ROOT / "data.pt") == manifest["data_sha256"] == result["data_sha256"]
data = torch.load(ROOT / "data.pt", map_location="cpu", weights_only=True)
assert tensor_sha(data["x"]) == manifest["x_sha256"] and data["x"].shape == (6144, 384)
assert all(tensor_sha(y) == manifest["targets"][t] for t, y in data["targets"].items())
assert all(tensor_sha(v) == manifest["streams"][str(s)] for s, v in data["streams"].items())
assert all(
    v.shape == (300, 256) and v.min() >= 0 and v.max() < 4096 for v in data["streams"].values()
)
regenerated = make_data()
assert torch.equal(regenerated["x"], data["x"])
for group in ("targets", "train_std", "streams"):
    assert all(torch.equal(regenerated[group][n], t) for n, t in data[group].items())
del regenerated
paths = sorted((ROOT / "cells").iterdir())
assert len(paths) == 252 and len(list((ROOT / "selections").glob("*.json"))) == 126
records, files = {}, {}
for path in paths:
    record, report = read(path / "training.json"), read(path / "heldout.json")
    history = record["history"]
    assert len(history) == 300 and [h["step"] for h in history] == list(range(1, 301))
    assert all(
        math.isfinite(h[k]) and h[k] >= 0
        for h in history
        for k in ("loss", "pre_clip_norm", "update_ms")
    )
    assert record["sampler_sha256"] == manifest["streams"][str(record["seed"])]
    assert record["data_sha256"] == manifest["data_sha256"]
    assert record["protocol_sha256"] == sha256(ROOT / "protocol.json")
    assert record["parameters"] == COUNTS[record["form"]]
    assert (
        sha256(path / "checkpoint.pt") == record["checkpoint_sha256"] == report["checkpoint_sha256"]
    )
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert finite_tree(checkpoint) and checkpoint["step"] == 300
    assert sum(t.numel() for t in checkpoint["model"].values()) == COUNTS[record["form"]]
    assert {n: tensor_sha(t) for n, t in checkpoint["model"].items()} == record[
        "final_state_hashes"
    ]
    assert all(state["step"].item() == 300 for state in checkpoint["optimizer"]["state"].values())
    assert sum(g["parameter_count"] for g in record["groups"]) == COUNTS[record["form"]]
    record["heldout"] = report
    records[path.name] = record
    files[path.name] = {p.name: sha256(p) for p in path.iterdir() if p.is_file()}

for seed in SEEDS:
    initial = {form: state_hashes(make_model(form, seed)) for form in FORMS}
    for record in records.values():
        if record["seed"] == seed:
            assert record["initial_state_hashes"] == initial[record["form"]]
    for form in ("absorbable", "cross"):
        assert all(initial[form][n] == h for n, h in initial["plain"].items())

selected = {}
for path in sorted((ROOT / "selections").glob("*.json")):
    selection = read(path)
    pair = [records[name] for name in selection["values"]]
    assert len(pair) == 2 and {r["base_rate"] for r in pair} == {0.001, 0.003}
    best = min(pair, key=lambda r: (r["selection_mse"], r["base_rate"]))
    assert best["cell"] == selection["selected_cell"]
    assert all(selection["values"][r["cell"]] == r["selection_mse"] for r in pair)
    assert all(
        path.stat().st_mtime_ns <= (ROOT / "cells" / r["cell"] / "heldout.json").stat().st_mtime_ns
        for r in pair
    )
    selected[best["task"], best["form"], best["seed"]] = best
assert len(selected) == 126
for row in result["selected_rows"]:
    r = selected[row["task"], row["form"], row["seed"]]
    assert row["cell"] == r["cell"] and row["heldout_mse"] == r["heldout"]["mse"]

torch.set_num_threads(4)
rescored = {}
for key, r in selected.items():
    checkpoint = torch.load(
        ROOT / "cells" / r["cell"] / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    model = make_model(r["form"], r["seed"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    value = score(model, data["x"][5120:], data["targets"][r["task"]][5120:])
    reference = r["heldout"]["mse"]
    assert math.isclose(value, reference, rel_tol=1e-5, abs_tol=1e-7), (key, value, reference)
    rescored[r["cell"]] = {
        "cpu_mse": value,
        "gpu_mse": reference,
        "relative_error": abs(value / reference - 1),
    }
    if r["form"] in ("absorbable", "cross"):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("rotation.theta"):
                    p.zero_()
        value = score(model, data["x"][5120:], data["targets"][r["task"]][5120:])
        reference = r["heldout"]["reset_angle_mse"]
        assert math.isclose(value, reference, rel_tol=1e-5, abs_tol=1e-7)
        rescored[r["cell"]].update(reset_cpu_mse=value, reset_gpu_mse=reference)


def ratio(form, ref, tasks=TASKS, seeds=SEEDS):
    return gm(
        [
            selected[t, form, s]["heldout"]["mse"] / selected[t, ref, s]["heldout"]["mse"]
            for t in tasks
            for s in seeds
        ]
    )


ratios = {f: {r: ratio(f, r) for r in FORMS} for f in FORMS}
assert ratios == result["ratios"]
independent_gates = {}
for form in ("absorbable", "cross"):
    tests = {
        "two_percent_over_plain": ratio(form, "plain") <= 0.98,
        "every_seed_better_than_plain": all(ratio(form, "plain", seeds=(s,)) < 1 for s in SEEDS),
        "beats_calibrated_narrow": ratio(form, "narrow") < 1,
        "generic_regression_cap": all(ratio(form, "plain", tasks=(t,)) <= 1.05 for t in TASKS[3:]),
        "seventy_percent_reduction": COUNTS[form] <= 0.3 * COUNTS["full_swiglu"],
        "finite_final_state": True,
    }
    if form == "cross":
        tests["one_percent_over_absorbable"] = ratio(form, "absorbable") <= 0.99
    independent_gates[form] = {
        "tests": tests,
        "earns_full_model_resource_qualification": all(tests.values()),
    }
assert independent_gates == result["gates"]
assert result["scientific_verdict"] == (
    "EARNS_RESOURCE_QUALIFICATION"
    if any(g["earns_full_model_resource_qualification"] for g in independent_gates.values())
    else "REJECTED_AT_THIS_FITTING_BUDGET"
)

resources = {}
for form in FORMS:
    rs = [r for r in selected.values() if r["form"] == form]
    resources[form] = {
        "median_selected_update_ms": statistics.median(r["median_update_ms"] for r in rs),
        "min_peak_allocated_mib": min(r["peak_allocated_bytes"] for r in rs) / 2**20,
        "max_peak_allocated_mib": max(r["peak_allocated_bytes"] for r in rs) / 2**20,
        "max_peak_reserved_mib": max(r["peak_reserved_bytes"] for r in rs) / 2**20,
        "mean_clip_fraction": statistics.mean(r["clip_fraction"] for r in rs),
        "max_pre_clip_norm": max(h["pre_clip_norm"] for r in rs for h in r["history"]),
        "selected_rate_counts": {
            str(lr): sum(r["base_rate"] == lr for r in rs) for lr in (0.001, 0.003)
        },
        "geomean_last50_loss_over_prior50": gm(
            [
                statistics.mean(h["loss"] for h in r["history"][250:])
                / statistics.mean(h["loss"] for h in r["history"][200:250])
                for r in rs
            ]
        ),
    }
    if form in ("absorbable", "cross"):
        resources[form]["reset_angle_error_ratio"] = gm(
            [r["heldout"]["reset_angle_mse"] / r["heldout"]["mse"] for r in rs]
        )
        resources[form]["max_abs_angle_radians"] = max(
            p["max_abs_radians"] for r in rs for p in r["final_diagnostics"]["angles"].values()
        )
        resources[form]["mean_projection_sine_rms"] = statistics.mean(
            p["sine_rms"] for r in rs for p in r["final_diagnostics"]["angles"].values()
        )
record = {
    "status": "PASS",
    "scientific_verdict": result["scientific_verdict"],
    "ratios": ratios,
    "task_ratios_to_plain": {t: {f: ratio(f, "plain", tasks=(t,)) for f in FORMS} for t in TASKS},
    "resources": resources,
    "rescored_selected_checkpoints": 126,
    "rescored_angle_ablations": 42,
    "max_cpu_gpu_relative_mse_error": max(r["relative_error"] for r in rescored.values()),
    "cpu_rescores": rescored,
    "cell_artifact_hashes": files,
    "selected_initializations_match": True,
    "selection_precedes_heldout_files": True,
    "all_252_checkpoint_weights_moments_finite": True,
    "source_unchanged": True,
    "completed_updates": 75600,
    "scientific_attempts": 1,
    "regenerated_data_and_streams_exact": True,
    "independently_rederived_gates": independent_gates,
    "research_goal_achieved": False,
}
write_json(OUT, record)
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("cpu_rescores", "cell_artifact_hashes")}
    )
)
