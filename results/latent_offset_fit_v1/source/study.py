"""H085 controlled gain/offset fitting, reusing the qualified numerical training loop."""

import math
import statistics
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from results.blast_operator_recovery_v1.source.storage import read, write_json
from results.latent_activation_v1.source import study as previous
from results.latent_offset_fit_v1.source.model import COUNTS, FORMS, HIDDEN, make_model

ROOT = Path("results/latent_offset_fit_v1")
PLAN = Path("research/latent_offset_fit_plan.md")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS, RATES, STEPS = (17, 29, 43), (0.001, 0.003), 600
base = previous.base
ORIGINAL_GROUPS = base.optimizer_groups


def make_data():
    x = (2*torch.rand(73728, 384, generator=torch.Generator().manual_seed(9813))-1)*math.sqrt(3)
    targets, scales = {}, {}
    for task in TASKS:
        targets[task], scales[task] = base.scale_targets(previous.target(task, x))
    streams = {seed: torch.randint(65536, (600, 256), generator=torch.Generator().manual_seed(20000+seed))
               for seed in SEEDS}
    p = previous.permutation()
    return {"x": x, "targets": targets, "train_std": scales, "streams": streams,
            "feature_permutation": p,
            "neighbor_cross_group_fraction": ((p//48) != (p.roll(-1)//48)).float().mean().item()}


def optimizer_groups(model, form, rate):
    groups = ORIGINAL_GROUPS(model, form, rate)
    if not form.endswith("_first_lr"):
        return groups
    lookup = {id(p): g for g in groups for p in g["params"]}
    rebuilt = {}
    for name, parameter in model.named_parameters():
        group = lookup[id(parameter)]
        scale = group["lr_scale"]*(5/8 if name.endswith("first.weight") else 1)
        key = (group["weight_decay"], scale)
        rebuilt.setdefault(key, {"params": [], "weight_decay": key[0], "lr_scale": scale, "lr": rate*scale})["params"].append(parameter)
    return list(rebuilt.values())


@torch.no_grad()
def diagnostics(model, x):
    value = previous.diagnostics(model, x)
    for name in ("up", "gate", "down"):
        projection = getattr(model, name)
        if projection is not None and hasattr(projection, "curve"):
            value[name]["curve"]["trainable_controls"] = list(dict(projection.curve.named_parameters()))
            value[name]["curve"]["fixed_buffers"] = list(dict(projection.curve.named_buffers()))
    return value


@torch.no_grad()
def heldout(record, x, y):
    value = previous.ORIGINAL_HELDOUT(record, x, y)
    if record["form"].startswith("latent_"):
        choice = read(ROOT / "selections" / f"{record['task']}_{record['form']}_s{record['seed']}.json")
        if choice["selected_cell"] == record["cell"]:
            path = ROOT / "cells" / record["cell"]
            model = make_model(record["form"], record["seed"]).cuda().eval()
            cp = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(cp["model"], strict=True)
            for projection in (model.up, model.gate, model.down):
                for parameter in projection.curve.parameters():
                    parameter.zero_()
                assert all(torch.count_nonzero(v) == 0 for v in projection.curve.buffers())
            reset = base.score(model, x, y)
            write_json(path / "ablation.json", {"reset_mse": reset, "original_mse": value["mse"],
                                               "ratio": reset/value["mse"], "optimizer_updates": 0})
    return value


def decisions(rows, rate_rows):
    if len(rows) != 120 or len({(r["task"], r["form"], r["seed"]) for r in rows}) != 120:
        raise ValueError("All 120 selected rows required")
    if len(rate_rows) != 240 or len({(r["task"], r["form"], r["seed"], r["rate"]) for r in rate_rows}) != 240:
        raise ValueError("All 240 rate rows required")
    selected = {(r["task"], r["form"], r["seed"]): r for r in rows}
    per_rate = {(r["task"], r["form"], r["seed"], r["rate"]): r for r in rate_rows}

    def ratio(form, ref, tasks=TASKS, seeds=SEEDS, rate=None):
        def value(t, f, seed):
            return (selected[t, f, seed] if rate is None else per_rate[t, f, seed, rate])["heldout_mse"]
        return base.geometric([value(t, form, seed)/value(t, ref, seed) for t in tasks for seed in seeds])

    def zero(form, task):
        return base.geometric([selected[task, form, seed]["heldout_mse"]/selected[task, form, seed]["zero_mse"] for seed in SEEDS])

    positive = sum(zero("full_gelu", t) <= 0.98 for t in TASKS) >= 2
    complete = all(r["finite"] and math.isfinite(r["heldout_mse"]) for r in [*rows, *rate_rows])
    gates, robustness = {}, {}
    for form in ("latent_offset", "latent_offset_first_lr"):
        ref = "plain_first_lr" if form.endswith("_first_lr") else "plain"
        robust = {str(rate): {"one_percent": ratio(form, ref, rate=rate) <= 0.99,
                              "every_seed": all(ratio(form, ref, seeds=(seed,), rate=rate) < 1 for seed in SEEDS)}
                  for rate in RATES}
        robustness[form] = {"reference": ref, "per_rate": robust,
                            "passes": all(all(v.values()) for v in robust.values())}
        tests = {
            "complete_finite_grid": complete,
            "positive_control_assay_passes": positive,
            "two_percent_over_plain": ratio(form, "plain") <= 0.98,
            "every_seed_better_than_plain": all(ratio(form, "plain", seeds=(seed,)) < 1 for seed in SEEDS),
            "one_percent_over_equal_count_gain": ratio(form, "latent_gain") <= 0.99,
            "one_percent_over_fixed_lr_plain": ratio(form, "plain_first_lr") <= 0.99,
            "beats_both_calibrated_narrows": all(ratio(form, ref) < 1 for ref in ("narrow_swiglu", "narrow_gelu")),
            "within_one_percent_full_affine": ratio(form, "latent_affine") <= 1.01,
            "per_task_regression_cap_vs_plain": all(ratio(form, "plain", tasks=(t,)) <= 1.05 for t in TASKS),
            "per_task_regression_cap_vs_zero": all(zero(form, t) <= 1.05 for t in TASKS),
            "seventy_percent_reduction": COUNTS[form] <= 0.3*COUNTS["full_gelu"],
            "matched_rate_offset_benefit": robustness[form]["passes"],
        }
        gates[form] = {"tests": tests, "earns_full_model_resource_qualification": all(tests.values())}
    return {"status": "complete", "assay_passed": positive, "selected_rows": rows, "rate_rows": rate_rows,
            "gates": gates, "robustness": robustness,
            "ratios": {f: {ref: ratio(f, ref) for ref in FORMS} for f in FORMS},
            "fixed_rate_ratios": {str(rate): {f: {ref: ratio(f, ref, rate=rate) for ref in FORMS} for f in FORMS} for rate in RATES},
            "fixed_rate_by_seed_ratios": {str(rate): {f: {str(seed): ratio(f, "plain_first_lr" if f.endswith("_first_lr") else "plain", seeds=(seed,), rate=rate) for seed in SEEDS} for f in FORMS} for rate in RATES},
            "ratios_to_zero": {t: {f: zero(f, t) for f in FORMS} for t in TASKS},
            "by_seed_ratios_to_plain": {f: {str(seed): ratio(f, "plain", seeds=(seed,)) for seed in SEEDS} for f in FORMS},
            "task_mean_mse": {t: {f: statistics.mean(selected[t, f, seed]["heldout_mse"] for seed in SEEDS) for f in FORMS} for t in TASKS},
            "cells": 240, "selection_cells": 120, "optimizer_updates": 144000,
            "training_example_presentations": 36864000, "corpus_targets": 0, "full_model_resource_workers": 0,
            "post_training_reset_ablations": 48, "research_goal_achieved": False,
            "scientific_verdict": "EARNS_RESOURCE_QUALIFICATION" if any(g["earns_full_model_resource_qualification"] for g in gates.values())
            else "REJECTED_AT_THIS_FITTING_BUDGET" if positive else "INCONCLUSIVE_ASSAY_FAILURE"}


def summarize(rows):
    rate_rows = []
    for task in TASKS:
        for seed in SEEDS:
            for form in FORMS:
                for rate in RATES:
                    name = f"{task}_{form}_s{seed}_lr{round(rate*1e6)}"
                    path = ROOT / "cells" / name
                    r, h = read(path / "training.json"), read(path / "heldout.json")
                    rate_rows.append({"task": task, "form": form, "seed": seed, "rate": rate, "cell": name,
                                      "heldout_mse": h["mse"], "zero_mse": h["zero_mse"],
                                      "finite": r["all_final_weights_and_moments_finite"]})
    return decisions(rows, rate_rows)


@contextmanager
def bound():
    replacements = {"ROOT": ROOT, "PLAN": PLAN, "TASKS": TASKS, "FORMS": FORMS, "SEEDS": SEEDS,
                    "RATES": RATES, "STEPS": STEPS, "COUNTS": COUNTS, "HIDDEN": HIDDEN,
                    "make_model": make_model, "make_data": make_data, "target": previous.target,
                    "optimizer_groups": optimizer_groups, "diagnostics": diagnostics,
                    "heldout": heldout, "summarize": summarize, "write_json": previous.durable_json}
    with ExitStack() as stack:
        for name, value in replacements.items():
            stack.enter_context(patch.object(base, name, value))
        stack.enter_context(patch.object(torch, "save", previous.save_tensor))
        yield


if __name__ == "__main__":
    try:
        with bound():
            base.run()
    except BaseException as exc:
        write_json(ROOT / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "scientific_retries": 0})
        raise
