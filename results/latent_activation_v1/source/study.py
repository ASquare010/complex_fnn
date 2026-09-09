"""H082 uses the existing fitting loop with isolated factories and frozen tasks."""

import math
import os
import statistics
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from results.blast_operator_recovery_v1.source.storage import equal_payload, read, write_json
from results.blast_operator_recovery_v1.source.storage import sha as sha
from results.latent_activation_v1.source.model import COUNTS, FORMS, HIDDEN, make_model, slope
from results.ungated_fit_v1.source import study as base
from src.core.native_recompute_audit import finite_tree

ROOT = Path("results/latent_activation_v1")
PLAN = Path("research/latent_activation_plan.md")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS, RATES, STEPS = (17, 29, 43), (0.001, 0.003), 600
TRAIN_ROWS, SELECT_END, TOTAL_ROWS, WIDTH = 65536, 69632, 73728, 384
ORIGINAL_TARGET, ORIGINAL_DATA = base.target, base.make_data
ORIGINAL_DIAGNOSTICS, ORIGINAL_SAVE, ORIGINAL_HELDOUT = base.diagnostics, torch.save, base.heldout


def permutation():
    return torch.randperm(WIDTH, generator=torch.Generator().manual_seed(9283))


def target(task, x):
    return ORIGINAL_TARGET(task, x[:, permutation()])


def make_data():
    value = ORIGINAL_DATA()
    value["feature_permutation"] = permutation()
    value["neighbor_cross_group_fraction"] = (
        (value["feature_permutation"] // 48) != (value["feature_permutation"].roll(-1) // 48)
    ).float().mean().item()
    return value


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_tensor(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        ORIGINAL_SAVE(value, f)
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    restored = torch.load(path, map_location="cpu", weights_only=True)
    equal_payload(restored, cpu_tree(value))


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


@torch.no_grad()
def diagnostics(model, x):
    result = ORIGINAL_DIAGNOSTICS(model, x)
    handles = []
    for name in ("up", "gate", "down"):
        module = getattr(model, name)
        if module is None or not hasattr(module, "curve"):
            continue

        def observe(curve, inputs, output, name=name):
            z = inputs[0].detach().float()
            derivative = slope(z, curve.theta_a, curve.theta_b, curve.family)
            parameters = torch.cat((curve.theta_a, curve.theta_b))
            item = {
                "family": curve.family, "theta_a": curve.theta_a.tolist(), "theta_b": curve.theta_b.tolist(),
                "amplitude": (0.5 * curve.theta_a.tanh()).tolist(),
                "input_rms": z.square().mean().sqrt().item(),
                "correction_rms": (output.float()-z).square().mean().sqrt().item(),
                "slope_min": derivative.min().item(), "slope_max": derivative.max().item(),
                "slope_mean": derivative.mean().item(), "slope_samples": z.numel(),
                "control_saturation_fraction": (parameters.tanh().abs() > 0.98).float().mean().item(),
                "finite": bool(torch.isfinite(derivative).all() and torch.isfinite(output).all()),
            }
            if curve.family == "affine":
                item["offset"] = (0.5 * curve.theta_b.tanh()).tolist()
            else:
                item["scale"] = (math.log(4)*curve.theta_b.tanh()).exp().tolist()
            assert item["finite"]
            result[name]["curve"] = item

        handles.append(module.curve.register_forward_hook(observe))
    try:
        if handles:
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert finite_tree(result)
    return result


@torch.no_grad()
def heldout(record, x, y):
    value = ORIGINAL_HELDOUT(record, x, y)
    if record["form"].startswith("latent_"):
        selection = read(ROOT / "selections" / f"{record['task']}_{record['form']}_s{record['seed']}.json")
        if selection["selected_cell"] == record["cell"]:
            path = ROOT / "cells" / record["cell"]
            model = make_model(record["form"], record["seed"]).cuda().eval()
            checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"], strict=True)
            for projection in (model.up, model.gate, model.down):
                projection.curve.theta_a.zero_()
                if projection.curve.family == "affine":
                    projection.curve.theta_b.zero_()
            reset_mse = base.score(model, x, y)
            durable_json(path / "ablation.json", {"reset_mse": reset_mse, "original_mse": value["mse"],
                                                  "ratio": reset_mse/value["mse"], "optimizer_updates": 0})
    return value


def summarize(rows):
    if len(rows) != 96 or len({(r["task"], r["form"], r["seed"]) for r in rows}) != 96:
        raise ValueError("All 96 selected task/form/seed rows are required")
    by_key = {(r["task"], r["form"], r["seed"]): r for r in rows}

    def ratio(form, reference, tasks=TASKS, seeds=SEEDS):
        return base.geometric([by_key[t, form, s]["heldout_mse"]/by_key[t, reference, s]["heldout_mse"]
                               for t in tasks for s in seeds])

    def over_zero(form, tasks=TASKS):
        return base.geometric([by_key[t, form, s]["heldout_mse"]/by_key[t, form, s]["zero_mse"]
                               for t in tasks for s in SEEDS])

    positive = sum(over_zero("full_gelu", (t,)) <= 0.98 for t in TASKS) >= 2
    complete = all(r["finite"] and math.isfinite(r["heldout_mse"]) for r in rows)
    gates = {}
    for form in ("latent_tanh", "latent_sine"):
        tests = {
            "complete_finite_grid": complete,
            "positive_control_assay_passes": positive,
            "two_percent_over_plain": ratio(form, "plain") <= 0.98,
            "every_seed_better_than_plain": all(ratio(form, "plain", seeds=(seed,)) < 1 for seed in SEEDS),
            "one_percent_over_equal_count_affine": ratio(form, "latent_affine") <= 0.99,
            "beats_both_calibrated_narrows": all(ratio(form, ref) < 1 for ref in ("narrow_swiglu", "narrow_gelu")),
            "per_task_regression_cap_vs_plain": all(ratio(form, "plain", tasks=(t,)) <= 1.05 for t in TASKS),
            "per_task_regression_cap_vs_zero": all(over_zero(form, (t,)) <= 1.05 for t in TASKS),
            "seventy_percent_reduction": COUNTS[form] <= 0.3 * COUNTS["full_gelu"],
        }
        gates[form] = {"tests": tests, "earns_full_model_resource_qualification": all(tests.values())}
    return {
        "status": "complete", "assay_passed": positive, "selected_rows": rows, "gates": gates,
        "ratios": {f: {ref: ratio(f, ref) for ref in FORMS} for f in FORMS},
        "ratios_to_zero": {t: {f: over_zero(f, (t,)) for f in FORMS} for t in TASKS},
        "by_seed_ratios_to_plain": {f: {str(s): ratio(f, "plain", seeds=(s,)) for s in SEEDS} for f in FORMS},
        "task_mean_mse": {t: {f: statistics.mean(by_key[t, f, s]["heldout_mse"] for s in SEEDS)
                              for f in FORMS} for t in TASKS},
        "cells": 192, "selection_cells": 96, "optimizer_updates": 115200,
        "training_example_presentations": 29491200, "corpus_targets": 0,
        "full_model_resource_workers": 0, "post_training_reset_ablations": 36,
        "research_goal_achieved": False,
        "scientific_verdict": "EARNS_RESOURCE_QUALIFICATION"
        if any(g["earns_full_model_resource_qualification"] for g in gates.values())
        else "REJECTED_AT_THIS_FITTING_BUDGET" if positive else "INCONCLUSIVE_ASSAY_FAILURE",
    }


@contextmanager
def bound():
    replacements = {
        "ROOT": ROOT, "PLAN": PLAN, "TASKS": TASKS, "FORMS": FORMS, "SEEDS": SEEDS,
        "RATES": RATES, "STEPS": STEPS, "COUNTS": COUNTS, "HIDDEN": HIDDEN,
        "make_model": make_model, "make_data": make_data, "target": target,
        "diagnostics": diagnostics, "heldout": heldout, "summarize": summarize, "write_json": durable_json,
    }
    with patch.multiple(base, **replacements), patch.object(torch, "save", save_tensor):
        yield


if __name__ == "__main__":
    try:
        with bound():
            base.run()
    except BaseException as exc:
        durable_json(ROOT / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                            "scientific_retries": 0})
        raise
