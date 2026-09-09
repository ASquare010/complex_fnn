"""Audit the 17 completed H074 workers; incomplete study grants no promotion."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

ROOT = Path("results/ungated_resource_v1")
OUT = Path("results/verification/ungated_resource_partial_analysis_v1.json")
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


protocol = read(ROOT / "protocol.json")
failure = read(ROOT / "failure.json")
assert failure["status"] == "INCOMPLETE_RUNTIME_FAILURE" and len(protocol["sources"]) == 100
assert not (ROOT / "result.json").exists()
assert all(sha(n) == h for n, h in protocol["sources"].items())
assert sha("research/ungated_resource_plan.md") == protocol["plan_sha256"]
stream = torch.randint(0, 4096, (20, 16, 129), generator=torch.Generator().manual_seed(60017))
assert torch.equal(stream, torch.load(ROOT / "tokens.pt", weights_only=True, map_location="cpu"))
assert (
    tensor_record(stream) == protocol["token_record"]
    and sha(ROOT / "tokens.pt") == protocol["token_file_sha256"]
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
        folder = ROOT / "workers" / cell
        if cell not in failure["completed_workers"]:
            assert not (folder / "result.json").exists()
            continue
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
        rows[cell] = row
        artifacts[cell] = {n.name: sha(n) for n in folder.iterdir() if n.is_file()}
for label in ("harness", *rows):
    path = ROOT / "processes" / (label + ".json")
    event = read(path)
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert event["protocol_sha256"] == sha(ROOT / "protocol.json") and event[
        "source_archive_sha256"
    ] == sha(ROOT / "source.zip")
    assert event["log_sha256"] == sha(path.with_suffix(".log"))
    process_hashes[label] = sha(path)

assert len(rows) == 17 and len(comparisons) == 11
failed = ROOT / "processes" / (failure["failed_worker"] + ".json")
event = read(failed)
assert event["status"] == "FAIL" and event["returncode"] == 1 and event["source_unchanged"]
assert sha(failed) == failure["failure_process_sha256"]
assert sha(failed.with_suffix(".log")) == event["log_sha256"] == failure["failure_log_sha256"]
assert not list((ROOT / "workers" / failure["failed_worker"]).iterdir())
assert all(
    not (ROOT / "processes" / (c + ".json")).exists() and not (ROOT / "workers" / c).exists()
    for c in failure["unlaunched_workers"]
)
assert sum(r["updates"] for r in rows.values()) == failure["completed_optimizer_updates"] == 340
assert (
    sum(r["synthetic_training_targets"] for r in rows.values())
    == failure["completed_synthetic_training_targets"]
    == 696320
)
assert (
    sum(r["initial_probe_targets"] for r in rows.values())
    == failure["completed_initial_probe_targets"]
    == 34816
)
record = {
    "status": "PASS",
    "scientific_status": "INCOMPLETE_RUNTIME_FAILURE",
    "completed_comparisons_exact": all(all(v.values()) for v in comparisons.values()),
    "comparisons": comparisons,
    "checkpoint_pairs_inspected": 17,
    "direct_cpu_fidelity_comparisons": 11,
    "all_completed_weights_moments_diagnostics_finite": True,
    "tokens_regenerated_exactly": True,
    "shared_initialization_exact": True,
    "completed_optimizer_updates": 340,
    "failed_worker_optimizer_updates": 0,
    "failed_worker_directory_empty": True,
    "unlaunched_workers": failure["unlaunched_workers"],
    "earns_language_screen": failure["earns_language_screen"],
    "worker_artifact_hashes": artifacts,
    "process_hashes": process_hashes,
    "failure_sha256": sha(ROOT / "failure.json"),
    "research_goal_achieved": False,
}
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k not in ("worker_artifact_hashes", "process_hashes", "comparisons")
        }
    ),
    flush=True,
)
