"""Independent H084 tensor checks and two reset-mode rescores; zero updates."""

import copy
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.latent_activation_recovery_v1.source.adapter import study as old
from src.core.native_recompute_audit import finite_tree

ROOT = Path("results/latent_affine_mechanism_v1")
PRIOR = Path("results/latent_activation_recovery_v1")
OUT = Path("results/verification/latent_affine_mechanism_analysis_v1.json")
MODES = ("original", "reset_gain", "reset_offset", "reset_both", "fold_gain", "fold_all")


def load(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    value = torch.load(path, map_location="cpu", weights_only=True)
    assert finite_tree(value)
    return value


def thash(t):
    t = t.contiguous()
    return hashlib.sha256(json.dumps([str(t.dtype), list(t.shape)]).encode()+t.numpy().tobytes()).hexdigest()


def geom(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


assert not OUT.exists() and read(ROOT / "process.json")["status"] == "PASS"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
p, r, b = [read(ROOT / n) for n in ("protocol.json", "result.json", "before.json")]
assert r["status"] == "COMPLETE" and not r["earns_training"] and not r["research_goal_achieved"]
assert len(p["sources"]) == 131 and all(sha(n) == h for n, h in p["sources"].items())
assert sha("research/latent_affine_mechanism_plan.md") == p["plan_sha256"] == b["plan_sha256"]
assert r["protocol_sha256"] == sha(ROOT / "protocol.json")
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 132
    assert all(hashlib.sha256(z.read(n)).hexdigest() == sha(n) for n in z.namelist())
assert all(sha(n) == h for n, h in b["anchors"].items())
assert all(sha(n) == h for n, h in b["prior_plans"].items())
for name, rec in b["checkpoint_files"].items():
    path = Path(name)
    assert sha(path) == rec["sha256"] and path.stat().st_size == rec["size"]
    assert path.stat().st_mtime_ns == rec["mtime_ns"]
assert len(b["checkpoint_files"]) == 60
assert len(r["rows"]) == len(b["selected_rows"]) == 12
assert len(list((ROOT / "cells").iterdir())) == 12
w = load(ROOT / "witness.pt")
assert sha(ROOT / "witness.pt") == r["witness_sha256"]
assert w["state"]["up.first.weight"].item() == 0
assert w["state"]["up.second.weight"].item() == 1
assert math.isclose(0.5*w["state"]["up.curve.theta_b"].tanh().item(), 0.25, rel_tol=1e-12, abs_tol=1e-12)
torch.testing.assert_close(w["input_gradient"].squeeze(), F.silu(torch.tensor(0.25, dtype=torch.float64)),
                           rtol=1e-12, atol=1e-12)
assert w["input_gradient"].item() > 0
assert torch.count_nonzero(w["output"]) == torch.count_nonzero(w["without_offset_gradient"]) == 0
data = load(PRIOR / "data.pt")
x = data["x"][69632:].cuda()
probe = torch.cat((torch.zeros(1, 384, dtype=torch.float64),
                   torch.randn(8, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9291))))
directions = torch.randn(8, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9292))
scores = {}
for row, selected in zip(r["rows"], b["selected_rows"]):
    name = row["cell"]
    assert name == selected["cell"] and row["seed"] == selected["seed"] and row["task"] == selected["task"]
    cell = ROOT / "cells" / name
    assert row == read(cell / "result.json")
    assert sha(cell / "tensors.pt") == row["tensor_sha256"]
    assert all(sha(cell / n) == h for n, h in r["artifact_hashes"][name].items())
    value = load(cell / "tensors.pt")
    assert torch.equal(value["probe"], probe) and torch.equal(value["directions"], directions)
    reference = value["comparisons"]["original"]
    for mode in ("fold_gain", "fold_all"):
        folded = value["comparisons"][mode]
        for field in ("output", "input_gradient"):
            torch.testing.assert_close(folded[field], reference[field], rtol=1e-10, atol=1e-10)
            assert (folded[field]-reference[field]).abs().max().item() == row["fp64_max_absolute_errors"][mode][field]
        error = abs(row["modes"][mode]["mse"]/row["modes"]["original"]["mse"]-1)
        assert error == row["modes"][mode]["relative_mse_drift"] and error <= 1e-5
    for mode in MODES[:4]:
        origin = value["origin"][mode]
        assert origin["output"].shape == (1, 384) and origin["jvps"].shape == (8, 384)
        assert row["origin"][mode] == {"output_norm": origin["output"].norm().item(),
                                      "jvp_norm": origin["jvps"].norm().item()}
        if mode in ("reset_offset", "reset_both"):
            assert torch.count_nonzero(origin["output"]) == torch.count_nonzero(origin["jvps"]) == 0
    assert row["modes"]["original"]["mse"] == selected["heldout_mse"]
    assert row["modes"]["reset_both"]["mse"] == read(PRIOR / "cells" / name / "ablation.json")["reset_mse"]
    interaction = (row["modes"]["reset_both"]["mse"]-row["modes"]["reset_gain"]["mse"]
                   -row["modes"]["reset_offset"]["mse"]+row["modes"]["original"]["mse"])
    assert interaction == row["factorial_mse_interaction"]
    for mode in MODES:
        rec, moments = row["modes"][mode], value["residuals"][mode]
        mean, second = moments["residual_mean"], moments["residual_second_moment"]
        assert mean.shape == second.shape == (384,) and (second >= 0).all()
        assert mean.square().mean().item() == rec["mean_error_energy"]
        assert second.mean().item() == rec["double_residual_mse"]
        assert math.isclose(rec["double_residual_mse"], rec["centered_error_energy"]+rec["mean_error_energy"], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(rec["mse"], rec["double_residual_mse"], rel_tol=1e-6, abs_tol=1e-6)
    cp_path = PRIOR / "cells" / name / "checkpoint.pt"
    assert sha(cp_path) == row["source_checkpoint_sha256"]
    cp = load(cp_path)
    original = old.make_model("latent_affine", selected["seed"])
    original.load_state_dict(cp["model"], strict=True)
    original.cuda().eval()
    y = data["targets"][selected["task"]][69632:].cuda()
    scores[name] = {}
    for mode in ("reset_gain", "reset_offset"):
        model = copy.deepcopy(original)
        with torch.no_grad():
            for projection in (model.up, model.gate, model.down):
                (projection.curve.theta_a if mode == "reset_gain" else projection.curve.theta_b).zero_()
            total, predictions = 0.0, []
            for i in range(0, 4096, 256):
                pred = model(x[i:i+256])
                total += F.mse_loss(pred, y[i:i+256], reduction="sum").item()
                predictions.append(pred.cpu())
        predictions = torch.cat(predictions)
        mse = total/y.numel()
        assert mse == row["modes"][mode]["mse"]
        assert thash(predictions) == row["modes"][mode]["prediction_sha256"]
        residual = predictions.double()-y.cpu().double()
        assert torch.equal(residual.mean(0), value["residuals"][mode]["residual_mean"])
        assert torch.equal(residual.square().mean(0), value["residuals"][mode]["residual_second_moment"])
        scores[name][mode] = mse
        del model
    del original, cp, y
    print(json.dumps({"audited_cell": name, "new_reset_rescores": 2}), flush=True)
ratios = {mode: geom([row["modes"][mode]["mse"]/row["modes"]["original"]["mse"] for row in r["rows"]]) for mode in MODES}
assert ratios == r["reset_to_original_ratios"]
tasks = ("smooth", "oscillatory", "multiplicative", "piecewise")
task_ratios = {t: {mode: geom([row["modes"][mode]["mse"]/row["modes"]["original"]["mse"]
                              for row in r["rows"] if row["task"] == t]) for mode in MODES} for t in tasks}
assert task_ratios == r["task_reset_to_original_ratios"]
assert r["maximum_fold_mse_drift"] == max(row["modes"][mode]["relative_mse_drift"] for row in r["rows"] for mode in ("fold_gain", "fold_all"))
assert r["optimizer_updates"] == r["corpus_targets"] == r["training_repetitions"] == 0
assert r["reporting_passes"] == 72 and r["reporting_examples"] == 294912
assert all(sha(n) == h for n, h in p["sources"].items())
assert all(sha(n) == h for n, h in b["anchors"].items())
record = {"status": "PASS", "research_goal_achieved": False, "earns_training": False,
          "checkpoint_cases": 12, "fp64_fold_comparisons": 24, "origin_cases_verified": 48,
          "witness_verified": True, "original_and_reset_both_anchors_exact": 24,
          "new_reset_rescores_exact": 24, "independent_reporting_examples": 98304,
          "optimizer_updates": 0, "corpus_targets": 0, "prediction_hashes_exact": 24,
          "residual_moment_records_verified": 72, "source_checkpoint_files_preserved": 60,
          "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
          "source_archive_sha256": sha(ROOT / "source.zip"), "scores": scores,
          "reset_to_original_ratios": ratios, "task_ratios": task_ratios,
          "maximum_fold_mse_drift": r["maximum_fold_mse_drift"], "artifact_hashes": r["artifact_hashes"]}
write_json(OUT, record)
print(json.dumps({k: v for k, v in record.items() if k not in ("scores", "artifact_hashes")}), flush=True)
