"""H084: fixed checkpoint interventions, local algebra and residual decomposition."""

import gc
import hashlib
import json
import math
import statistics
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.latent_activation_recovery_v1.source.adapter import study as old
from results.latent_activation_v1.source.model import LatentCurve, projected_forward
from results.latent_affine_mechanism_v1.source.algebra import MODES, transform
from src.blockshuffle_ffn import BlockShuffleFFN
from src.core.native_recompute_audit import finite_tree

ROOT = Path("results/latent_affine_mechanism_v1")
PRIOR = Path("results/latent_activation_recovery_v1")
PLAN = Path("research/latent_affine_mechanism_plan.md")


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(json.dumps([str(t.dtype), list(t.shape)]).encode()+t.numpy().tobytes()).hexdigest()


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def check_prior(before):
    for name, h in before["anchors"].items():
        assert sha(name) == h, name
    for name, record in before["checkpoint_files"].items():
        p = Path(name)
        assert sha(p) == record["sha256"] and p.stat().st_size == record["size"]
        assert p.stat().st_mtime_ns == record["mtime_ns"]
    assert all(sha(n) == h for n, h in before["prior_plans"].items())
    assert all(sha(n) == h for n, h in before["prior_sources"].items())


def forward_gradient(model, x):
    x = x.detach().clone().requires_grad_(True)
    output = model(x)
    probe = torch.linspace(-0.7, 1.1, output.numel(), dtype=torch.float64).reshape_as(output)
    gradient = torch.autograd.grad((output*probe).mean(), x)[0]
    return {"output": output.detach(), "input_gradient": gradient.detach()}


def origin(model, directions):
    x = torch.zeros(1, 384, dtype=torch.float64)
    values = [torch.autograd.functional.jvp(model, x, v.reshape_as(x), create_graph=False)
              for v in directions]
    return {"output": values[0][0], "jvps": torch.cat([v[1] for v in values])}


def witness():
    model = BlockShuffleFFN(1, 1, 1).double()
    for projection in (model.up, model.gate, model.down):
        projection.curve = LatentCurve(1, 1, "affine").double()
        projection.forward = MethodType(projected_forward, projection)
        with torch.no_grad():
            projection.first.weight.fill_(1)
            projection.second.weight.fill_(1)
    with torch.no_grad():
        model.up.first.weight.zero_()
        model.up.curve.theta_b.fill_(math.atanh(0.5))
    x = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    y = model(x)
    gradient = torch.autograd.grad(y.sum(), x)[0]
    expected = F.silu(torch.tensor(0.25, dtype=torch.float64))
    assert torch.count_nonzero(y) == 0 and gradient.item() > 0
    torch.testing.assert_close(gradient.squeeze(), expected, rtol=1e-12, atol=1e-12)
    plain = transform(model, "reset_offset")
    plain_y = plain(x)
    plain_gradient = torch.autograd.grad(plain_y.sum(), x)[0]
    assert torch.count_nonzero(plain_gradient) == 0
    return {"state": model.state_dict(), "output": y.detach(), "input_gradient": gradient,
            "expected_gradient": expected, "without_offset_gradient": plain_gradient}


@torch.no_grad()
def reporting_score(model, x, y):
    outputs, total = [], 0.0
    for start in range(0, len(x), 256):
        prediction = model(x[start:start+256])
        total += F.mse_loss(prediction, y[start:start+256], reduction="sum").item()
        outputs.append(prediction.cpu())
    prediction = torch.cat(outputs)
    residual = prediction.double()-y.cpu().double()
    mean = residual.mean(0)
    second = residual.square().mean(0)
    centered = (residual-mean).square().mean().item()
    mean_energy = mean.square().mean().item()
    double_mse = second.mean().item()
    mse = total/y.numel()
    assert math.isclose(double_mse, mean_energy+centered, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(mse, double_mse, rel_tol=1e-6, abs_tol=1e-6)
    return {"mse": mse, "double_residual_mse": double_mse,
            "mean_error_energy": mean_energy, "centered_error_energy": centered,
            "prediction_sha256": tensor_sha(prediction)}, {"residual_mean": mean, "residual_second_moment": second}


def run():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before, protocol = read(ROOT / "before.json"), read(ROOT / "protocol.json")
    check_prior(before)
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    assert sha(PLAN) == protocol["plan_sha256"]
    data = torch.load(PRIOR / "data.pt", map_location="cpu", weights_only=True)
    assert finite_tree(data)
    probe = torch.cat((torch.zeros(1, 384, dtype=torch.float64),
                       torch.randn(8, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9291))))
    directions = torch.randn(8, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9292))
    old.save_tensor(witness(), ROOT / "witness.pt")
    rows, artifact_hashes = [], {}
    x = data["x"][69632:].cuda()
    for selected in before["selected_rows"]:
        name, task, seed = selected["cell"], selected["task"], selected["seed"]
        cell = ROOT / "cells" / name
        cell.mkdir(parents=True, exist_ok=False)
        cp_path = PRIOR / "cells" / name / "checkpoint.pt"
        cp = torch.load(cp_path, map_location="cpu", weights_only=True)
        assert finite_tree(cp)
        original = old.make_model("latent_affine", seed)
        original.load_state_dict(cp["model"], strict=True)
        original.eval()
        m64 = old.make_model("latent_affine", seed)
        m64.load_state_dict(cp["model"], strict=True)
        m64.double().eval()
        payload = {"probe": probe, "directions": directions, "comparisons": {}, "origin": {}, "residuals": {}}
        reference = forward_gradient(m64, probe)
        payload["comparisons"]["original"] = reference
        max_errors = {}
        for mode in ("fold_gain", "fold_all"):
            folded = transform(m64, mode)
            value = forward_gradient(folded, probe)
            for key in reference:
                torch.testing.assert_close(value[key], reference[key], rtol=1e-10, atol=1e-10)
            payload["comparisons"][mode] = value
            max_errors[mode] = {k: (value[k]-reference[k]).abs().max().item() for k in value}
            if mode == "fold_all":
                assert sum(p.numel() for p in folded.parameters()) == 350208
                assert sum(b.numel() for b in folded.buffers()) == 4480
            del folded
        for mode in MODES[:4]:
            altered = transform(m64, mode)
            value = origin(altered, directions)
            if mode in ("reset_offset", "reset_both"):
                assert torch.count_nonzero(value["output"]) == torch.count_nonzero(value["jvps"]) == 0
            payload["origin"][mode] = value
            del altered
        original.cuda()
        y = data["targets"][task][69632:].cuda()
        modes = {}
        for mode in MODES:
            altered = transform(original, mode)
            value, residuals = reporting_score(altered, x, y)
            modes[mode] = value
            payload["residuals"][mode] = residuals
            del altered
        assert modes["original"]["mse"] == selected["heldout_mse"]
        prior_reset = read(PRIOR / "cells" / name / "ablation.json")
        assert modes["reset_both"]["mse"] == prior_reset["reset_mse"]
        for mode in ("fold_gain", "fold_all"):
            relative = abs(modes[mode]["mse"]/modes["original"]["mse"]-1)
            assert relative <= 1e-5, (name, mode, relative)
            modes[mode]["relative_mse_drift"] = relative
        assert finite_tree(payload) and finite_tree(modes)
        old.save_tensor(payload, cell / "tensors.pt")
        interaction = (modes["reset_both"]["mse"]-modes["reset_gain"]["mse"]
                       -modes["reset_offset"]["mse"]+modes["original"]["mse"])
        record = {"cell": name, "task": task, "seed": seed, "rate": selected["rate"],
                  "source_checkpoint_sha256": sha(cp_path), "modes": modes,
                  "fp64_max_absolute_errors": max_errors,
                  "origin": {mode: {"output_norm": value["output"].norm().item(),
                                    "jvp_norm": value["jvps"].norm().item()}
                             for mode, value in payload["origin"].items()},
                  "factorial_mse_interaction": interaction, "optimizer_updates": 0,
                  "tensor_sha256": sha(cell / "tensors.pt")}
        write_json(cell / "result.json", record)
        artifact_hashes[name] = {n.name: sha(n) for n in cell.iterdir()}
        rows.append(record)
        print(json.dumps({"cell": name, "reset_ratios": {
            mode: modes[mode]["mse"]/modes["original"]["mse"] for mode in MODES[1:4]}}), flush=True)
        del cp, original, m64, y, payload, reference
        gc.collect()
        torch.cuda.empty_cache()
    assert len(rows) == 12
    check_prior(before)
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    result = {"status": "COMPLETE", "scientific_verdict": "CHECKPOINT_MECHANISM_CHARACTERIZED",
              "research_goal_achieved": False, "earns_training": False,
              "checkpoint_cases": 12, "reporting_passes": 72, "reporting_examples": 294912,
              "optimizer_updates": 0, "corpus_targets": 0, "training_repetitions": 0,
              "fp64_fold_comparisons": 24, "origin_mode_cases": 48, "origin_jvp_directions_per_case": 8,
              "rows": rows, "artifact_hashes": artifact_hashes,
              "reset_to_original_ratios": {mode: geometric([v["modes"][mode]["mse"]/v["modes"]["original"]["mse"] for v in rows]) for mode in MODES},
              "task_reset_to_original_ratios": {t: {mode: geometric([v["modes"][mode]["mse"]/v["modes"]["original"]["mse"] for v in rows if v["task"] == t]) for mode in MODES} for t in ("smooth", "oscillatory", "multiplicative", "piecewise")},
              "maximum_fold_mse_drift": max(v["modes"][mode]["relative_mse_drift"] for v in rows for mode in ("fold_gain", "fold_all")),
              "protocol_sha256": sha(ROOT / "protocol.json"), "witness_sha256": sha(ROOT / "witness.pt")}
    write_json(ROOT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "artifact_hashes")}), flush=True)


if __name__ == "__main__":
    run()
