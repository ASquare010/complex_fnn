"""H100 fitting loop adapted from the frozen H088 loop, with biases counted correctly."""

import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import sha, write_json
from results.neuron_geometry_v1.source.study import save_tensor, score, tensor_sha
from results.triadic_interaction_v1.source.model import FORMS, InteractionFFN

ROOT = Path("results/triadic_interaction_v1")
STEPS, BATCH, TRAIN, SELECT_END = 300, 256, 65536, 69632
COUNTS = {
    f: 6144
    if f == "feature_aware"
    else 1182080
    if f == "full_swiglu"
    else 1181568
    if f.startswith("full_")
    else 351200
    if f in ("cp2", "narrow_swiglu")
    else 351276
    if f in ("cp3", "bounded_cp3")
    else 351050
    if f == "starrelu"
    else 351048
    for f in FORMS
}
MATRICES = {
    f: 6144 if f == "feature_aware" else 1179648 if f.startswith("full_") else 350208 for f in FORMS
}
EXTRAS = {f: 2 if f == "starrelu" else 0 for f in FORMS}


@torch.no_grad()
def diagnostics(model, probe, step):
    before, after = model.features(probe)
    summary = {
        "step": step,
        "gradient_norms": {
            name: p.grad.norm().item() if p.grad is not None else None
            for name, p in model.named_parameters()
        },
    }
    for name, value in (("preactivation", before), ("activation", after)):
        summary[name] = {
            "mean": value.mean().item(),
            "std": value.std(correction=0).item(),
            "rms": value.square().mean().sqrt().item(),
            "abs_max": value.abs().max().item(),
            "finite": bool(torch.isfinite(value).all()),
        }
    summary["projection_statistics"] = {}
    for name in ("up", "gate", "third"):
        projection = getattr(model, name)
        if projection is not None:
            value = projection(probe)
            summary["projection_statistics"][name] = {
                "mean": value.mean().item(),
                "std": value.std(correction=0).item(),
                "abs_max": value.abs().max().item(),
                "finite": bool(torch.isfinite(value).all()),
            }
    summary["shape"] = model.curve.shape_parameters()
    if hasattr(model.curve, "theta"):
        summary["theta"] = model.curve.theta.detach().cpu().tolist()
        summary["control_saturation_fraction"] = (
            (model.curve.theta.tanh().abs() > 0.98).float().mean().item()
        )
    return summary


def train_cell(task, form, seed, rate, x, y, stream, data_sha, protocol_sha):
    label = f"{task}_{form}_s{seed}_lr{int(rate * 1e6)}"
    folder = ROOT / "cells" / label
    folder.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = InteractionFFN(form, seed, task).cuda()
    initial_hashes = {name: tensor_sha(value) for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(rate), betas=(0.9, 0.95), eps=1e-8, weight_decay=0
    )
    group_record = [
        {
            "name": group["parameter_name"],
            "lr": group["lr"],
            "parameters": sum(p.numel() for p in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    history, samples = [], [diagnostics(model, x[:BATCH], 0)]
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        batch_x, batch_y = x[stream[step - 1]], y[stream[step - 1]]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = F.mse_loss(model(batch_x), batch_y)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        value, norm = loss.item(), gradient_norm.item()
        assert math.isfinite(value) and math.isfinite(norm)
        history.append(
            {
                "step": step,
                "loss": value,
                "preclip_norm": norm,
                "forward_ms": 1000 * (forward_end - started),
                "backward_ms": 1000 * (backward_end - forward_end),
                "update_ms": 1000 * (finished - started),
            }
        )
        if step in (50, 150, 300, 600):
            samples.append(diagnostics(model, x[:BATCH], step))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    optimizer_steps = []
    for state in optimizer.state.values():
        optimizer_steps.append(int(state["step"].item()))
        assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
    assert len(optimizer_steps) == len(list(model.parameters())) and set(optimizer_steps) == {STEPS}
    selection = score(model, x[TRAIN:SELECT_END], y[TRAIN:SELECT_END])
    peak = torch.cuda.max_memory_allocated()
    path = folder / "checkpoint.pt"
    checkpoint = {
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "step": STEPS,
        "form": form,
        "seed": seed,
        "rate": rate,
        "task": task,
        "data_sha256": data_sha,
        "protocol_sha256": protocol_sha,
        "stream_sha256": tensor_sha(stream),
        "optimizer_steps": optimizer_steps,
        "resume_supported": False,
    }
    save_tensor(path, checkpoint)
    result = {
        "label": label,
        "task": task,
        "form": form,
        "seed": seed,
        "rate": rate,
        "parameters": COUNTS[form],
        "activation_parameters": EXTRAS[form],
        "matrix_forward_flops_per_example": 2 * MATRICES[form],
        "matrix_flops_exclude_activation_and_backward": True,
        "selection_mse": selection,
        "peak_bytes": peak,
        "median_update_ms": statistics.median(row["update_ms"] for row in history[50:]),
        "median_forward_ms": statistics.median(row["forward_ms"] for row in history[50:]),
        "median_backward_ms": statistics.median(row["backward_ms"] for row in history[50:]),
        "clipped_fraction": sum(row["preclip_norm"] > 1 for row in history) / STEPS,
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": sha(path),
        "initial_state_sha256": initial_hashes,
        "optimizer_groups": group_record,
        "history": history,
        "diagnostics": samples,
        "optimizer_steps": optimizer_steps,
        "data_sha256": data_sha,
        "protocol_sha256": protocol_sha,
        "all_finite": True,
    }
    write_json(folder / "training.json", result)
    del checkpoint, optimizer, model, loss, batch_x, batch_y
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"completed": label, "selection_mse": selection}), flush=True)
    return result
