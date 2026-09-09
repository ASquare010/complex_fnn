"""Independently audit H081 records and rescore all final checkpoints once."""

import hashlib
import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from results.blast_learning_screen_v1.source import study as s
from src.core.benchmark import evaluate_forward
from src.core.data import TokenData
from src.core.native_recompute_audit import digest, finite_tree, tensor_record
from src.core.optimization import group_summary, initialize_dense_width

ROOT = s.ROOT
OUT = Path("results/verification/blast_learning_screen_analysis_v1.json")
assert not OUT.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
p = s.read(ROOT / "protocol.json")
before = s.read(ROOT / "before.json")
result = s.read(ROOT / "result.json")
assert result["status"] == "COMPLETE" and not result["research_goal_achieved"]
assert s.read(ROOT / "coordinator_status.json")["status"] == "PASS"
assert s.sha(s.PLAN) == p["plan_sha256"] == before["plan_sha256"]
assert all(s.sha(n) == h for n, h in p["sources"].items()) and len(p["sources"]) == 121
assert s.sha("results/verification/blast_operator_recovery_final_v1.json") == before["h080_final_sha256"]
assert s.sha("results/blast_operator_recovery_v1/source.zip") == before["h080_source_sha256"]
assert s.sha("results/blast_operator_recovery_v1/result.json") == before["h080_result_sha256"]
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None and len(z.namelist()) == 122
    assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in p["sources"].items())
checks = [s.read(n) for n in sorted((ROOT / "preflight").glob("*.json"))]
assert len(checks) == 9 and all(r["status"] == "PASS" for r in checks)
assert sum(r["optimizer_updates"] for r in checks) == 24
assert sum(r.get("training_targets", 0) for r in checks) == 24960
gradient_pairs = sum(sum(step["gradient_pairs_exact"] for step in r.get("steps", [])) for r in checks)
for label in ("preflight", *s.CELLS, "finish"):
    e = s.read(ROOT / "processes" / (label + ".json"))
    assert e["status"] == "PASS" and e["returncode"] == 0 and e["source_unchanged"]
    assert s.sha(ROOT / "processes" / (label + ".log")) == e["log_sha256"]
data = TokenData(s.CACHE, "cuda", 10017)
assert len(data.train) == 3083650 and len(data.valid) == 322802
assert all(s.sha(s.CACHE / n) == h for n, h in p["data_files"].items())
assert s.sha(s.CACHE / "manifest.json") == p["data_manifest_sha256"]
# Reproduce only the sampler's draw sequence, independently of the training loop.
generator = torch.Generator(device="cuda").manual_seed(10017)
for _ in range(200):
    torch.randint(3083650 - 128, (16,), device="cuda", generator=generator)
expected_rng = generator.get_state()
validation_batches = list(data.validation(16, 128, 158))
assert len(validation_batches) == 158 and validation_batches[-1][0].shape == (9, 128)
assert sum(y.numel() for _, y in validation_batches) == 322688
assert all(torch.equal(x[:, 1:], y[:, :-1]) for x, y in validation_batches)

rows, artifacts = {}, {}
for cell, spec in p["cells"].items():
    folder = ROOT / "runs" / cell
    q, m = s.read(folder / "qualification.json"), s.read(folder / "metrics.json")
    assert q["status"] == "PASS" and q["all_finite"] and finite_tree(m)
    assert all(s.sha(folder / n) == h for n, h in q["artifact_hashes"].items())
    with zipfile.ZipFile(folder / "checkpoint.pt") as z:
        assert z.testzip() is None
    ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert finite_tree(ckpt) and ckpt["step"] == 200
    assert torch.equal(ckpt["sampling_rng"], expected_rng)
    assert tensor_record(ckpt["sampling_rng"]) == q["sampling_rng"]
    mc, tc = s.configuration(spec["form"], spec["rate"])
    assert asdict(mc) == m["model"] == ckpt["model_config"] == p["configurations"][cell]["model"]
    assert asdict(tc) == m["training"] == ckpt["training_config"] == p["configurations"][cell]["training"]
    model = s.make_model(mc, tc.seed, s.MODES[spec["form"]])
    initialize_dense_width(model, tc)
    initial = digest(model.state_dict())
    assert initial == s.read(folder / "initial_state_signature.json")["weights"]
    groups = s.parameter_groups(model, tc)
    assert group_summary(groups) == m["optimizer_parameter_groups"]
    assert s.calibration(model) == s.read(folder / "factor_calibration.json")
    saved_groups = ckpt["optimizer"]["param_groups"]
    assert len(groups) == len(saved_groups)
    ids = []
    for g, saved in zip(groups, saved_groups):
        assert (g["lr_scale"], g["weight_decay"]) == (saved["lr_scale"], saved["weight_decay"])
        assert saved["lr"] == s.trainer.learning_rate(199, tc) * g["lr_scale"]
        assert saved["betas"] == (0.9, 0.95) and saved["eps"] == 1e-8
        assert len(g["params"]) == len(saved["params"])
        for parameter, index in zip(g["params"], saved["params"]):
            state = ckpt["optimizer"]["state"][index]
            assert float(state["step"]) == 200
            assert state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape
            assert bool((state["exp_avg_sq"] >= 0).all())
            ids.append(index)
    assert len(ids) == len(set(ids)) == len(ckpt["optimizer"]["state"])
    assert sum(t.numel() for n, t in ckpt["model"].items() if ".ffn." in n) == m["ffn_parameters"] == mc.unique_ffn_parameters
    assert sum(t.numel() for t in ckpt["model"].values()) == m["total_parameters"] == mc.total_parameters
    assert m["ffn_reduction_percent"] == 100 * (1 - mc.unique_ffn_parameters / 9437184)
    assert m["total_reduction_percent"] == 100 * (1 - mc.total_parameters / 15735168)
    assert all(m[k] == v for k, v in s.operation_counts(mc).items())
    steps = list(map(json.loads, (folder / "all_steps.jsonl").read_text().splitlines()))
    history = list(map(json.loads, (folder / "history.jsonl").read_text().splitlines()))
    assert len(steps) == 200 and finite_tree(steps) and finite_tree(history)
    assert [r["step"] for r in history] == [1, 50, 100, 150, 200]
    for i, step in enumerate(steps):
        assert step["step"] == i+1 and step["learning_rate"] == s.trainer.learning_rate(i, tc)
    assert m["clipped_step_fraction"] == sum(r["gradient_norm_pre_clip"] > 1 for r in steps) / 200
    assert all(all(r[k] == steps[r["step"]-1][k] for k in ("step", "training_loss", "gradient_norm_pre_clip", "learning_rate")) for r in history)
    assert history[-1]["validation_loss"] == m["validation_loss"]
    for phase in ("initial", "final"):
        diagnostic = s.read(folder / (phase + "_diagnostics.json"))
        assert len(diagnostic) == 32 and finite_tree(diagnostic) and all(r["finite"] for r in diagnostic.values())
    assert digest(ckpt["model"]) == q["final_weights"] and digest(ckpt["optimizer"]) == q["final_optimizer"]
    model.load_state_dict(ckpt["model"], strict=True)
    model.to("cuda").eval()
    score, count = evaluate_forward(model, iter(validation_batches), "cuda", "bf16")
    assert count == 322688 and score == m["validation_loss"], (cell, score, m["validation_loss"])
    assert m["training_tokens"] == 409600 and m["validation_tokens"] == count
    assert math.isclose(m["timed_training_seconds"] * m["training_tokens_per_second"], 190 * 2048, rel_tol=1e-12)
    rows[cell] = {"form": spec["form"], "rate": spec["rate"], "validation_loss": score,
                  "peak_mib": m["peak_allocated_vram_bytes"] / 2**20,
                  "mean_training_step_ms": m["timed_training_seconds"] / 190 * 1000,
                  "inference_latency_ms": m["inference_latency_ms"], "clipped_fraction": m["clipped_step_fraction"],
                  "initial_weights": initial, "initial_validation_loss": m["initial_validation_loss"],
                  "first_training_loss": steps[0]["training_loss"], "final_validation_rescore_exact": True,
                  "ffn_parameters": m["ffn_parameters"], "total_parameters": m["total_parameters"]}
    artifacts[cell] = {n.name: s.sha(n) for n in folder.iterdir() if n.is_file()}
    del model, ckpt, groups, saved_groups
    print(json.dumps({"cell": cell, "status": "RESCORE_EXACT", "validation_loss": score}), flush=True)

# Check the historical reproduction diagnostic without treating it as a selection gate.
reproduction = s.read(ROOT / "control_reproduction.json")
assert reproduction["compared_controls"] == 15 and reproduction["optimizer_updates"] == 0
for cell, observed in reproduction["comparisons"].items():
    prior = s.read(Path("results/ungated_lm_screen_v1/runs") / cell / "metrics.json")
    assert rows[cell]["initial_validation_loss"] == prior["initial_validation_loss"]
    assert observed["new_final_nll"] == rows[cell]["validation_loss"]
    assert observed["prior_final_nll"] == prior["validation_loss"]
    assert observed["nll_difference"] == rows[cell]["validation_loss"] - prior["validation_loss"]
assert sum(v["nll_difference"] == 0 for v in reproduction["comparisons"].values()) == 14

# Recalculate selection and all seven numeric gates without calling the selector.
selected = {f: sorted((c for c, r in rows.items() if r["form"] == f),
                      key=lambda c: (rows[c]["validation_loss"], rows[c]["rate"]))[0] for f in s.FORMS}
assert selected == result["selected_cells"]
gates, comparisons = {}, {}
for f in s.NEW_FORMS:
    c = rows[selected[f]]
    gates[f] = {"all_cells_complete_finite": True,
                "at_least_70_percent_fewer_ffn_weights": 1 - c["ffn_parameters"]/9437184 >= 0.70}
    for ref in ("full_swiglu", "full_gelu"):
        r = rows[selected[ref]]
        gates[f][f"nll_within_one_percent_{ref}"] = c["validation_loss"] <= 1.01*r["validation_loss"]
        gates[f][f"memory_within_ten_percent_{ref}"] = c["peak_mib"] <= 1.1*r["peak_mib"]
    for ref in ("narrow_swiglu", "narrow_gelu"):
        gates[f][f"nll_beats_{ref}"] = c["validation_loss"] < rows[selected[ref]]["validation_loss"]
    comparisons[f] = {ref: 100 * (c["validation_loss"] / rows[selected[ref]]["validation_loss"] - 1)
                      for ref in s.FORMS if ref != f}
assert gates == result["gates"]
assert {f: all(g.values()) for f, g in gates.items()} == result["earns_longer_comparison"]
for rate in s.RATES:
    suffix = f"_lr{round(rate*1e6)}"
    a, b = rows["plain"+suffix], rows["plain_calibrated"+suffix]
    assert all(a[k] == b[k] for k in ("initial_weights", "initial_validation_loss", "first_training_loss"))
assert all(s.sha(n) == h for n, h in p["sources"].items())
output = {"status": "PASS", "rows": rows, "selected_cells": selected, "gates": gates,
          "selected_nll_relative_percent": comparisons, "cell_artifact_hashes": artifacts,
          "checkpoint_rescores_exact": 24, "independent_validation_targets": 7744512,
          "qualification_gradient_pairs_exact": gradient_pairs, "qualification_updates": 24,
          "language_optimizer_updates": 4800, "analysis_optimizer_updates": 0,
          "result_sha256": s.sha(ROOT / "result.json"), "protocol_sha256": s.sha(ROOT / "protocol.json"),
          "source_archive_sha256": s.sha(ROOT / "source.zip"), "research_goal_achieved": False}
s.write_json(OUT, output)
print(json.dumps({"status": "PASS", "selected_cells": selected, "gates": gates}), flush=True)
