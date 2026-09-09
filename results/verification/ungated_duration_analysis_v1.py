"""Audit all H078 language endpoints and independently rescore the full validation set."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch

from results.ungated_duration_v1.source import study
from src.core.benchmark import evaluate_forward
from src.core.data import TokenData
from src.core.optimization import initialize_dense_width, parameter_groups

ROOT = Path("results/ungated_duration_v1")
OUT = Path("results/verification/ungated_duration_analysis_v1.json")
assert not OUT.exists()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def record(t):
    t = t.detach().cpu().contiguous()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def normal(t):
    if isinstance(t, torch.Tensor):
        return record(t)
    if isinstance(t, dict):
        return {str(k): normal(v) for k, v in t.items()}
    if isinstance(t, (tuple, list)):
        return [normal(v) for v in t]
    return t


def digest(t):
    return hashlib.sha256(
        json.dumps(normal(t), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def finite(t):
    if isinstance(t, torch.Tensor):
        return bool(torch.isfinite(t).all())
    if isinstance(t, dict):
        return all(finite(v) for v in t.values())
    if isinstance(t, (tuple, list)):
        return all(finite(v) for v in t)
    return not isinstance(t, float) or math.isfinite(t)


p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
assert r["status"] == "complete" and read(ROOT / "coordinator_status.json")["status"] == "PASS"
assert len(p["sources"]) == 112 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/ungated_duration_plan.md") == p["plan_sha256"]
assert all(sha(study.CACHE / n) == h for n, h in p["data_files"].items())
assert sha(study.CACHE / "manifest.json") == p["data_manifest_sha256"]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
data = TokenData(study.CACHE, "cuda", 10017)
train_length, valid_length = len(data.train), len(data.valid)
assert (train_length, valid_length) == (3083650, 322802)
valid_cpu = torch.from_numpy(np.load(study.CACHE / "valid.npy").astype("int64"))
actual = torch.cat([y.detach().cpu().flatten() for _, y in data.validation(16, 128, 158)])
assert torch.equal(actual, valid_cpu[1:322689]) and actual.numel() == 322688
validation_shapes = [list(y.shape) for _, y in data.validation(16, 128, 158)]
assert len(validation_shapes) == 158 and validation_shapes[-1] == [9, 128]
assert all(s == [16, 128] for s in validation_shapes[:-1])
qualified_initials = read(ROOT / "qualification/initial_signatures.json")
qualified_samplers = read(ROOT / "qualification/samplers.json")
qualification_hashes = read(ROOT / "qualification_manifest.json")
assert len(qualification_hashes) == 2 and all(sha(n) == h for n, h in qualification_hashes.items())
assert qualified_samplers["optimizer_updates"] == qualified_samplers["model_forwards"] == 0
assert qualified_samplers["actual_batch_calls"] == 3200
assert qualified_samplers["unscored_sampled_target_elements"] == 6553600
assert qualified_samplers["validation"]["ordered_targets"] == record(actual)
expected_states = {}
for seed in (17,):
    expected_sampler = torch.Generator(device="cuda").manual_seed(seed + 10000)
    q = qualified_samplers["seeds"][str(seed)]
    for step in range(3200):
        starts = torch.randint(train_length - 128, (16,), device="cuda", generator=expected_sampler)
        if step == 0:
            indices = starts[:, None] + torch.arange(128, device="cuda")
            assert record(starts) == q["first_starts"]
            assert record(data.train[indices]) == q["first_x"]
            assert record(data.train[indices + 1]) == q["first_y"]
        if step in (799, 3199):
            assert record(expected_sampler.get_state()) == q[f"state_after_{step + 1}"]
    expected_states[seed] = expected_sampler.get_state()
prior = read("results/ungated_lm_replication_v1/result.json")
assert prior["replicated_primary_pass"]
forms = {
    "full_swiglu": 9437184,
    "full_gelu": 9437184,
    "narrow_swiglu": 2801664,
    "narrow_gelu": 2801664,
    "plain": 2801664,
    "gelu_matched": 2801664,
}
rates = {f: 0.0006 if f in ("narrow_gelu", "plain") else 0.0012 for f in forms}
metrics, hashes, rescores, processes = {}, {}, {}, {}
common_initial = {}
for form, count in forms.items():
    for seed in (17,):
        rate = rates[form]
        expected_state = expected_states[seed]
        cell = f"{form}_seed{seed}"
        folder = ROOT / "runs" / cell
        m, q = read(folder / "metrics.json"), read(folder / "qualification.json")
        config = read(folder / "config.json")
        ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert m["status"] == "SCREENED" and q["status"] == "PASS" and finite(m) and finite(ckpt)
        assert m["training"] == p["configurations"][cell]["training"] == ckpt["training_config"]
        assert m["model"] == p["configurations"][cell]["model"] == ckpt["model_config"]
        assert all(config[k] == m[k] for k in config)
        assert ckpt["step"] == 3200 and len(ckpt["optimizer"]["state"]) == len(ckpt["model"])
        assert all(
            float(v["step"]) == 3200 and "exp_avg" in v and "exp_avg_sq" in v
            for v in ckpt["optimizer"]["state"].values()
        )
        assert (
            torch.equal(ckpt["sampling_rng"], expected_state)
            and record(expected_state) == q["sampling_rng"]
        )
        assert (
            digest(ckpt["model"]) == q["final_weights"]
            and digest(ckpt["optimizer"]) == q["final_optimizer"]
        )
        assert (
            sum(v.numel() for n, v in ckpt["model"].items() if ".ffn." in n)
            == count
            == m["ffn_parameters"]
        )
        assert (
            sum(v.numel() for v in ckpt["model"].values())
            == count + 6297984
            == m["total_parameters"]
        )
        assert m["ffn_reduction_percent"] == 100 * (1 - count / 9437184)
        assert m["ffn_matrix_forward_flops_per_token"] == 2 * count
        assert (
            m["model_matrix_forward_flops_per_token"]
            == 8 * (8 * 384**2 + 4 * 128 * 384) + 2 * count + 2 * 384 * 4096
        )
        assert (
            m["training_matrix_flops_per_token_estimate"]
            == 3 * m["model_matrix_forward_flops_per_token"]
        )
        assert m["training_tokens"] == 6553600 and m["validation_tokens"] == 322688
        assert m["parameter_bytes"] == 4 * m["total_parameters"]
        assert m["optimizer_state_bytes"] == sum(
            t.numel() * t.element_size()
            for v in ckpt["optimizer"]["state"].values()
            for t in v.values()
            if isinstance(t, torch.Tensor)
        )
        assert m["data"]["files"] == p["data_files"] and m["environment"] == p["environment"]
        assert m["provenance"]["source_files"] == p["sources"]
        with zipfile.ZipFile(folder / "source.zip") as z:
            assert z.testzip() is None
            assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
        all_steps = list(
            map(json.loads, (folder / "all_steps.jsonl").read_text(encoding="utf-8").splitlines())
        )
        history = list(
            map(json.loads, (folder / "history.jsonl").read_text(encoding="utf-8").splitlines())
        )
        assert (
            len(all_steps) == 3200
            and [v["step"] for v in all_steps] == list(range(1, 3201))
            and finite(all_steps)
        )
        assert [v["step"] for v in history] == [1, 800, 1600, 2400, 3200] and finite(history)
        assert history[-1]["validation_loss"] == m["validation_loss"]
        for row in history:
            assert all(
                row[k] == all_steps[row["step"] - 1][k]
                for k in ("step", "training_loss", "gradient_norm_pre_clip", "learning_rate")
            )
        for index, row in enumerate(all_steps):
            expected = (
                rate * (index + 1) / 320
                if index < 320
                else rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ((index - 320) / 2879))))
            )
            assert row["learning_rate"] == expected
        assert (
            m["clipped_step_fraction"]
            == sum(v["gradient_norm_pre_clip"] > 1 for v in all_steps) / 3200
        )
        assert m["final_gradient_norm"] == all_steps[-1]["gradient_norm_pre_clip"]
        assert math.isclose(
            m["training_tokens_per_second"] * m["timed_training_seconds"],
            3190 * 16 * 128,
            rel_tol=1e-12,
        )
        for phase in ("initial", "final"):
            diagnostic = read(folder / (phase + "_diagnostics.json"))
            assert (
                len(diagnostic) == 24
                and finite(diagnostic)
                and all(v["finite"] for v in diagnostic.values())
            )
        initial = read(folder / "initial_state_signature.json")
        assert initial == qualified_initials[cell]
        mc, tc = study.configuration(form, seed)
        model = study.short.base.make_model(mc, seed, study.MODES[form])
        initialize_dense_width(model, tc)
        assert digest(model.state_dict()) == initial["weights"]
        assert {n: record(v) for n, v in model.named_parameters()} == initial["parameters"]
        common = {n: record(v) for n, v in model.named_parameters() if ".ffn." not in n}
        if seed not in common_initial:
            common_initial[seed] = common
        assert common == common_initial[seed]
        old = prior["rows"][cell]
        assert m["model"] == old["model"]
        assert m["training"] == {**old["training"], "steps": 3200, "log_every": 800}
        oldfolder = Path("results/ungated_lm_replication_v1/runs") / cell
        assert initial == read(oldfolder / "initial_state_signature.json")
        oldhistory = list(map(json.loads, (oldfolder / "history.jsonl").read_text().splitlines()))
        assert history[0]["training_loss"] == oldhistory[0]["training_loss"]
        assert m["initial_validation_loss"] == old["initial_validation_loss"]
        groups = parameter_groups(model, tc)
        groupsummary = [
            {
                "weight_decay": g["weight_decay"],
                "lr_scale": g["lr_scale"],
                "parameter_count": sum(v.numel() for v in g["params"]),
            }
            for g in groups
        ]
        assert groupsummary == m["optimizer_parameter_groups"]
        if form.startswith("narrow_"):
            for g in groups:
                if g["lr_scale"] != 1:
                    assert math.isclose(g["weight_decay"] * g["lr_scale"], 0.1, rel_tol=1e-15)
        assert q["execution_mode"] == study.MODES[form]
        for filename, key in (
            ("metrics.json", "metrics_sha256"),
            ("checkpoint.pt", "checkpoint_sha256"),
            ("source.zip", "source_archive_sha256"),
            ("all_steps.jsonl", "all_steps_sha256"),
        ):
            assert sha(folder / filename) == q[key]
        model.load_state_dict(ckpt["model"], strict=True)
        model = model.cuda().eval()
        score, targets = evaluate_forward(model, data.validation(16, 128, 158), "cuda", "bf16")
        assert targets == 322688
        rescores[cell] = {
            "stored_nll": m["validation_loss"],
            "rescored_nll": score,
            "absolute_error": abs(score - m["validation_loss"]),
            "targets": targets,
        }
        assert score == m["validation_loss"], rescores[cell]
        del model, ckpt
        torch.cuda.empty_cache()
        metrics[cell] = m
        hashes[cell] = {n.name: sha(n) for n in folder.iterdir() if n.is_file()}
        print(json.dumps({"cell": cell, "rescore_exact": True, "nll": score}), flush=True)
for label in ("harness", *metrics, "finish"):
    path = ROOT / "processes" / (label + ".json")
    event = read(path)
    assert event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
    assert sha(path.with_suffix(".log")) == event["log_sha256"]
    assert (
        sha(ROOT / "protocol.json") == event["protocol_sha256"]
        and sha(ROOT / "source.zip") == event["source_archive_sha256"]
    )
    processes[label] = sha(path)
assert "5 passed" in (ROOT / "processes/harness.log").read_text(encoding="utf-8")
# Independently evaluate the fixed seed-17 endpoint; do not call study.decisions.
refs = {f: metrics[f"{f}_seed17"] for f in forms}
m = refs["gelu_matched"]
gates = {
    "all_cells_complete_finite": len(metrics) == 6 and all(finite(v) for v in metrics.values()),
    "at_least_70_percent_fewer_ffn_weights": all(
        100 * (1 - m["ffn_parameters"] / refs[f]["ffn_parameters"]) >= 70
        for f in ("full_swiglu", "full_gelu")
    ),
    **{
        f"nll_within_one_percent_{f}": m["validation_loss"] <= 1.01 * refs[f]["validation_loss"]
        for f in ("full_swiglu", "full_gelu")
    },
    **{
        f"nll_beats_{f}": m["validation_loss"] < refs[f]["validation_loss"]
        for f in ("narrow_swiglu", "narrow_gelu")
    },
    **{
        f"memory_within_ten_percent_{f}": m["peak_allocated_vram_bytes"]
        <= 1.1 * refs[f]["peak_allocated_vram_bytes"]
        for f in ("full_swiglu", "full_gelu")
    },
}
comparisons = {
    f: {
        "nll_difference": m["validation_loss"] - refs[f]["validation_loss"],
        "relative_nll_percent": 100 * (m["validation_loss"] / refs[f]["validation_loss"] - 1),
    }
    for f in forms
    if f != "gelu_matched"
}
margins = {
    f: m["validation_loss"] <= 0.998 * refs[f]["validation_loss"]
    for f in ("narrow_swiglu", "narrow_gelu")
}
plateau = {}
for cell, m in metrics.items():
    history = list(
        map(json.loads, (ROOT / "runs" / cell / "history.jsonl").read_text().splitlines())
    )
    late = 100 * (history[-1]["validation_loss"] / history[-2]["validation_loss"] - 1)
    above = 100 * (history[-1]["validation_loss"] / min(v["validation_loss"] for v in history) - 1)
    assert late == r["rows"][cell]["late_nll_change_percent"]
    assert above == r["rows"][cell]["final_above_best_percent"]
    plateau[cell] = abs(late) <= 0.2 and above <= 0.2
primary = all(gates.values())
assert gates == r["gates"] and primary == r["primary_pass"]
assert (
    comparisons == r["comparisons"] and margins == r["descriptive_point_two_percent_narrow_margin"]
)
assert (
    plateau == r["operational_late_plateau"]
    and all(plateau.values()) == r["all_forms_late_plateau"]
)
assert all(sha(n) == h for n, h in p["sources"].items())
recorded = {
    "status": "PASS",
    "completed_lm_trials": 6,
    "all_checkpoints_moments_finite": True,
    "all_initial_states_reconstructed_exactly": True,
    "all_step_logs_consistent": True,
    "all_sampler_states_exact": True,
    "validation_order_exact": True,
    "validation_targets_per_pass": 322688,
    "cuda_bf16_rescores": rescores,
    "rescore_validation_targets": 1936128,
    "rescore_optimizer_updates": 0,
    "all_6_rescores_exact": True,
    "frozen_recipes_verified": True,
    "initial_nll_and_first_loss_match_h077": True,
    "independently_rederived_gates": gates,
    "primary_pass": primary,
    "independently_rederived_comparisons": comparisons,
    "operational_late_plateau": plateau,
    "cell_artifact_hashes": hashes,
    "process_hashes": processes,
    "qualification_hashes": qualification_hashes,
    "result_sha256": sha(ROOT / "result.json"),
    "official_test_scored": False,
    "research_goal_achieved": False,
}
OUT.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in recorded.items()
            if k not in ("cuda_bf16_rescores", "cell_artifact_hashes", "process_hashes")
        }
    ),
    flush=True,
)
