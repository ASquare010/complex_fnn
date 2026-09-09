"""H075 independent CPU audit of the combined 17 original and four recovered cells."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

ROOT = Path("results/ungated_resource_recovery_v1")
OLD = Path("results/ungated_resource_v1")
OUT = Path("results/verification/ungated_resource_recovery_analysis_v1.json")
assert not OUT.exists()
torch.set_num_threads(4)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tensor_record(t):
    v = t.detach().cpu().contiguous()
    return {
        "shape": list(v.shape),
        "dtype": str(v.dtype),
        "sha256": hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def normalize(t):
    if isinstance(t, torch.Tensor):
        return tensor_record(t)
    if isinstance(t, dict):
        return {str(k): normalize(v) for k, v in t.items()}
    if isinstance(t, (tuple, list)):
        return [normalize(v) for v in t]
    return t


def digest(t):
    return hashlib.sha256(
        json.dumps(normalize(t), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def finite(t):
    if isinstance(t, torch.Tensor):
        return bool(torch.isfinite(t).all())
    if isinstance(t, dict):
        return all(finite(v) for v in t.values())
    if isinstance(t, (list, tuple)):
        return all(finite(v) for v in t)
    return not isinstance(t, float) or math.isfinite(t)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


protocol, result = read(ROOT / "protocol.json"), read(ROOT / "result.json")
assert result["status"] == "complete" and len(protocol["sources"]) == 103
assert all(sha(n) == h for n, h in protocol["sources"].items())
assert sha("research/ungated_resource_plan.md") == protocol["plan_sha256"]
assert sha("research/ungated_resource_recovery_plan.md") == protocol["recovery_plan_sha256"]
stream = torch.randint(0, 4096, (20, 16, 129), generator=torch.Generator().manual_seed(60017))
assert torch.equal(stream, torch.load(OLD / "tokens.pt", weights_only=True, map_location="cpu"))
assert (
    tensor_record(stream) == protocol["token_record"]
    and sha(OLD / "tokens.pt") == protocol["token_file_sha256"]
)
forms = {
    "full_swiglu": (1024, 9437184),
    "full_gelu": (1536, 9437184),
    "narrow_swiglu": (304, 2801664),
    "narrow_gelu": (456, 2801664),
    "plain": (2048, 2801664),
    "gelu_same": (2048, 1867776),
    "gelu_matched": (3264, 2801664),
}
modes = ("none", "block", "block_inner")
rows, comparisons, artifacts, process_hashes = {}, {}, {}, {}
shared = None
plain_initial = None
for form, (hidden, count) in forms.items():
    baseline_initial = baseline_final = baseline_sig = baseline_history = baseline_row = None
    for mode in modes:
        cell = f"{form}_{mode}"
        folder = Path(protocol["origins"][cell]) / "workers" / cell
        row = read(folder / "result.json")
        config = read(folder / "config.json")
        initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
        final = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        signature = read(folder / "initial_signature.json")
        history = list(
            map(json.loads, (folder / "history.jsonl").read_text(encoding="utf-8").splitlines())
        )
        assert (
            initial["step"] == 0
            and final["step"] == 20
            and final["synthetic"]
            and finite(initial)
            and finite(final)
        )
        assert not initial["optimizer"]["state"]
        assert len(final["optimizer"]["state"]) == len(final["model"])
        assert all(
            float(v["step"]) == 20 and "exp_avg" in v and "exp_avg_sq" in v
            for v in final["optimizer"]["state"].values()
        )
        assert (
            len(history) == 20
            and [v["step"] for v in history] == list(range(1, 21))
            and finite(history)
        )
        assert (
            sum(p.numel() for n, p in initial["model"].items() if ".ffn." in n)
            == count
            == row["ffn_parameters"]
        )
        assert (
            sum(p.numel() for p in initial["model"].values())
            == count + 6297984
            == row["total_parameters"]
        )
        assert config["model"]["hidden"] == hidden and config["actual_ffn_parameters"] == count
        if form.startswith("gelu_"):
            assert config["model"]["variant"] == "blockshuffle_gelu"
            assert not any(".gate." in n for n in initial["model"])
        assert row["ffn_projection_mac_per_token_per_layer"] == count // 8
        assert row["ffn_reduction_percent"] == 100 * (1 - count / 9437184)
        assert signature["weights"] == digest(initial["model"]) and signature[
            "optimizer"
        ] == digest(initial["optimizer"])
        assert set(signature["gradients"]) == set(initial["model"])
        assert all(
            signature["gradients"][n]["shape"] == list(p.shape) for n, p in initial["model"].items()
        )
        assert row["final"] == {
            "weights": digest(final["model"]),
            "optimizer": digest(final["optimizer"]),
        }
        assert row["cpu_rng"] == tensor_record(final["cpu_rng"]) and row[
            "cuda_rng"
        ] == tensor_record(final["cuda_rng"])
        assert row["token_record"] == protocol["token_record"]
        assert row["step_seconds"] == [v["seconds"] for v in history[10:]]
        assert row["median_step_ms"] == 1000 * statistics.median(row["step_seconds"])
        assert row["training_tokens_per_second"] == 20480 / sum(row["step_seconds"])
        assert row["clipped_step_fraction"] == sum(v["norm_pre_clip"] > 1 for v in history) / 20
        assert row["peak_reserved_bytes"] >= row["peak_allocated_bytes"] > 0
        for n, k in (
            ("initial.pt", "initial_weights_sha256"),
            ("checkpoint.pt", "checkpoint_sha256"),
            ("history.jsonl", "history_sha256"),
            ("initial_signature.json", "initial_signature_sha256"),
        ):
            assert sha(folder / n) == row[k]
        for phase in ("initial", "final"):
            diagnostic = read(folder / (phase + "_diagnostics.json"))
            assert (
                finite(diagnostic)
                and len(diagnostic["layers"]) == 24
                and len(diagnostic["sampled_activation_slopes"]) == 8
            )
            assert all(v["finite"] for group in diagnostic.values() for v in group.values())
        assert finite(read(folder / "final_gradient_stats.json"))
        assert all(
            read(folder / "adapter.json")[k]
            for k in (
                "parameter_objects_preserved",
                "optimizer_references_preserved",
                "state_preserved",
            )
        )
        current_shared = {n: p for n, p in initial["model"].items() if ".ffn." not in n}
        if shared is None:
            shared = current_shared
        assert equal(shared, current_shared)
        assert signature["non_ffn"] == digest(shared)
        if form == "plain" and mode == "none":
            plain_initial = initial["model"]
        if form == "gelu_same":
            assert all(torch.equal(p, plain_initial[n]) for n, p in initial["model"].items())
        clean_history = [{k: v[k] for k in ("step", "loss", "norm_pre_clip")} for v in history]
        if mode == "none":
            baseline_initial, baseline_final, baseline_sig, baseline_history, baseline_row = (
                initial,
                final,
                signature,
                clean_history,
                row,
            )
        else:
            assert equal(initial, baseline_initial)
            comparisons[cell] = {
                "initial_signature": signature == baseline_sig,
                "all_losses_and_norms": clean_history == baseline_history,
                "final_weights_and_moments": equal(final["model"], baseline_final["model"])
                and equal(final["optimizer"], baseline_final["optimizer"]),
                "token_rng": all(
                    row[k] == baseline_row[k] for k in ("token_record", "cpu_rng", "cuda_rng")
                ),
            }
        assert row == result["rows"][cell]
        rows[cell] = row
        artifacts[cell] = {n.name: sha(n) for n in folder.iterdir() if n.is_file()}
for label in ("harness", "cpu_small", "full_1", "full_2", *rows, "finish"):
    origin = OLD if label == "harness" else Path(protocol["origins"].get(label, ROOT.as_posix()))
    path = origin / "processes" / (label + ".json")
    event = read(path)
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert event["protocol_sha256"] == sha(origin / "protocol.json") and event[
        "source_archive_sha256"
    ] == sha(origin / "source.zip")
    assert event["log_sha256"] == sha(path.with_suffix(".log"))
    process_hashes[label] = sha(path)
assert comparisons == result["comparisons"]
exact = all(all(v.values()) for v in comparisons.values())
assert exact == result["all_comparisons_exact"]
chosen = {
    f: sorted(
        modes,
        key=lambda m: (
            rows[f"{f}_{m}"]["peak_allocated_bytes"],
            rows[f"{f}_{m}"]["median_step_ms"],
            modes.index(m),
        ),
    )[0]
    for f in forms
}
gates = {}
for f in ("gelu_same", "gelu_matched"):
    mode = chosen[f]
    row = rows[f"{f}_{mode}"]
    gates[f] = {
        "all_workers_and_fidelity_exact": exact
        and all(r["status"] == "PASS" for r in rows.values()),
        "at_least_70_percent_fewer_ffn_weights": row["ffn_reduction_percent"] >= 70,
        "all_diagnostics_and_state_finite": all(r["all_finite"] for r in rows.values()),
        **{
            f"memory_within_ten_percent_{ref}": row["peak_allocated_bytes"]
            <= 1.1 * min(rows[f"{ref}_{m}"]["peak_allocated_bytes"] for m in modes)
            for ref in ("full_swiglu", "full_gelu")
        },
        "time_within_25_percent_plain_same_mode": row["median_step_ms"]
        <= 1.25 * rows[f"plain_{mode}"]["median_step_ms"],
    }
assert gates == result["gates"] and chosen == result["selected_modes"]
assert {f: all(v.values()) for f, v in gates.items()} == result["earns_language_screen"]
assert (
    result["combined_optimizer_updates"] == 21 * 20
    and result["combined_synthetic_training_targets"] == 21 * 20 * 16 * 128
)
assert result["combined_initial_probe_targets"] == 21 * 16 * 128 and result["corpus_targets"] == 0
for name in ("cpu_small", "full_1", "full_2"):
    probe = read(ROOT / "probes" / (name + ".json"))
    assert probe["status"] == "PASS" and all(
        probe[k] == 0 for k in ("forwards", "backwards", "optimizer_updates", "corpus_targets")
    )
    if name != "cpu_small":
        original = read(OLD / "workers/gelu_same_none/initial_signature.json")
        assert probe["evidence"]["initial_weights"] == original["weights"]
        assert probe["evidence"]["initial_optimizer"] == original["optimizer"]
        assert probe["evidence"]["optimizer_groups"] == original["groups"]
assert result["new_optimizer_updates"] == 80 and result["completed_cell_reruns"] == 0
assert (
    result["explicit_preupdate_failed_cell_retries"] == 1
    and result["combined_worker_launches"] == 22
)
assert len(list((ROOT / "workers").iterdir())) == 4
assert not list((OLD / "workers/gelu_same_block_inner").iterdir())
assert sha("results/verification/ungated_resource_final_v1.json") == protocol["h074_final_sha256"]
assert sha(OLD / "failure.json") == protocol["h074_failure_sha256"]
for cell, files in protocol["preserved_h074_worker_hashes"].items():
    assert all(sha(OLD / "workers" / cell / n) == h for n, h in files.items())
assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
record = {
    "status": "PASS",
    "all_comparisons_exact": exact,
    "checkpoint_pairs_inspected": 21,
    "original_cells_preserved": 17,
    "recovered_cells": 4,
    "construction_probes_passed": 3,
    "new_optimizer_updates": 80,
    "combined_optimizer_updates": 420,
    "explicit_preupdate_failed_cell_retries": 1,
    "completed_cell_reruns": 0,
    "direct_cpu_fidelity_comparisons": 14,
    "all_weights_moments_diagnostics_finite": True,
    "tokens_regenerated_exactly": True,
    "shared_initialization_exact": True,
    "selected_modes": chosen,
    "independently_rederived_gates": gates,
    "earns_language_screen": result["earns_language_screen"],
    "worker_artifact_hashes": artifacts,
    "process_hashes": process_hashes,
    "result_sha256": sha(ROOT / "result.json"),
    "research_goal_achieved": False,
}
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("worker_artifact_hashes", "process_hashes")}
    ),
    flush=True,
)
