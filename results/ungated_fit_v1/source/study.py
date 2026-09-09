"""H073 full-size ungated fitting with absolute-error and positive controls, isolated from the active model factory."""

import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.blockshuffle_ffn import BlockShuffleFFN, BlockShuffleLinear
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import group_summary, parameter_groups
from src.core.reproducibility import environment, sha256, write_json
from src.dense_ffn import DenseFFN

ROOT = Path("results/ungated_fit_v1")
PLAN = Path("research/ungated_fit_plan.md")
TASKS = (
    "linear",
    "plain_teacher",
    "gelu_teacher",
    "smooth",
    "oscillatory",
    "multiplicative",
    "piecewise",
)
FORMS = (
    "plain",
    "gelu_same",
    "gelu_matched",
    "narrow_swiglu",
    "narrow_gelu",
    "full_swiglu",
    "full_gelu",
)
SEEDS, RATES, STEPS, WIDTH = (17, 29, 43), (0.001, 0.003), 300, 384
TRAIN_ROWS, SELECT_END, TOTAL_ROWS = 65536, 69632, 73728
COUNTS = dict(zip(FORMS, (350208, 233472, 350208, 350208, 350208, 1179648, 1179648)))
HIDDEN = dict(zip(FORMS, (2048, 2048, 3264, 304, 456, 1024, 1536)))


def write_new(path, value):
    assert not path.exists(), path
    write_json(path, value)


def tensor_sha(tensor):
    t = tensor.detach().cpu().contiguous()
    header = json.dumps([str(t.dtype), list(t.shape)]).encode()
    return hashlib.sha256(header + t.numpy().tobytes()).hexdigest()


def state_hashes(model):
    return {name: tensor_sha(t) for name, t in model.state_dict().items()}


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


@torch.no_grad()
def make_model(form, seed):
    if form in ("plain", "gelu_same", "gelu_matched"):
        model = BlockShuffleFFN(WIDTH, HIDDEN[form], 8, gated=form == "plain")
    else:
        model = DenseFFN(WIDTH, HIDDEN[form], "gelu" if "gelu" in form else "swiglu")
    for name in ("up", "gate", "down"):
        projection = getattr(model, name)
        if projection is None:
            continue
        if isinstance(projection, BlockShuffleLinear):
            projection.initialize(seed, name)
            gain = math.sqrt(1 / (0.02 * math.sqrt(projection.input_width)))
            projection.first.weight.mul_(gain)
            projection.second.weight.mul_(gain)
        else:
            digest = hashlib.sha256(f"{seed}:{name}.weight".encode()).digest()
            g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            projection.weight.normal_(0, 1 / math.sqrt(projection.in_features), generator=g)
    assert sum(p.numel() for p in model.parameters()) == COUNTS[form]
    return model


def optimizer_groups(model, form, rate):
    variant = "gelu" if "gelu" in form else "swiglu"
    hidden = HIDDEN[form]
    wrapper = nn.Module()
    wrapper.add_module("ffn", model)
    wrapper.blocks = [SimpleNamespace(ffn=model)]
    wrapper.config = ModelConfig(variant=variant, width=WIDTH, hidden=hidden)
    config = TrainConfig(
        learning_rate=rate,
        weight_decay=0,
        ffn_lr_mode="fan_in",
        ffn_width_lr_mode="fan_in" if form.startswith("narrow_") else "none",
    )
    return parameter_groups(wrapper, config)


@torch.no_grad()
def target(task, x):
    if task.endswith("_teacher"):
        form = {"plain_teacher": "plain", "gelu_teacher": "gelu_matched"}[task]
        teacher = make_model(form, 9281).eval()
        return teacher(x)
    g = torch.Generator().manual_seed(9282)
    rotation = torch.linalg.qr(torch.randn(WIDTH, WIDTH, generator=g)).Q
    if task == "linear":
        return x @ rotation
    a, b, c, d = (torch.roll(x, -i, -1) for i in range(4))
    if task == "smooth":
        y = torch.sin(2 * a) + b.square() + torch.exp(0.5 * c) - d
    elif task == "oscillatory":
        y = torch.sin(3 * a) * torch.cos(2 * b) + 0.5 * torch.sin(4 * c + d)
    elif task == "multiplicative":
        y = a * b + 2 * b * c * d + a.square() * d
    elif task == "piecewise":
        y = F.relu(a + b) - 0.7 * (c - d).abs() + torch.where(a > 0, b, c)
    else:
        raise ValueError(task)
    return y @ rotation


def scale_targets(y, train_rows=TRAIN_ROWS):
    std = y[:train_rows].std(dim=0, correction=0)
    assert (std > 0).all()
    return y / std, std


def make_data():
    g = torch.Generator().manual_seed(9812)
    x = (2 * torch.rand(TOTAL_ROWS, WIDTH, generator=g) - 1) * math.sqrt(3)
    targets, scales = {}, {}
    for task in TASKS:
        targets[task], scales[task] = scale_targets(target(task, x))
    streams = {
        seed: torch.randint(
            TRAIN_ROWS, (STEPS, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        for seed in SEEDS
    }
    return {"x": x, "targets": targets, "train_std": scales, "streams": streams}


@torch.no_grad()
def score(model, x, y):
    total = 0.0
    for begin in range(0, len(x), 256):
        prediction = model(x[begin : begin + 256])
        total += F.mse_loss(prediction, y[begin : begin + 256], reduction="sum").item()
    return total / y.numel()


@torch.no_grad()
def diagnostics(model, x):
    result = {}
    for name in ("up", "gate", "down"):
        projection = getattr(model, name)
        if projection is None:
            continue
        result[name] = {
            "weight_norm": math.sqrt(sum(p.square().sum().item() for p in projection.parameters()))
        }
        if name != "down":
            z = projection(x)
            result[name].update(rms=z.square().mean().sqrt().item(), abs_max=z.abs().max().item())
    return result


def train_cell(task, form, seed, rate, x, y, stream, hashes):
    name = f"{task}_{form}_s{seed}_lr{round(rate * 1e6)}"
    cell = ROOT / "cells" / name
    cell.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(form, seed).cuda().train()
    initial_hashes = state_hashes(model)
    groups = optimizer_groups(model, form, rate)
    optimizer = torch.optim.AdamW(groups, lr=rate, betas=(0.9, 0.95), eps=1e-8)
    history = []
    torch.cuda.empty_cache()
    started = time.perf_counter()
    initial_diagnostics = diagnostics(model, x[:256])
    for step in range(STEPS):
        if step == 50:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        idx = stream[step]
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[idx])
        loss = F.mse_loss(prediction, y[idx])
        assert torch.isfinite(loss), (name, step, "nonfinite loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if step == STEPS - 1:
            final_gradients = {n: p.grad.norm().item() for n, p in model.named_parameters()}
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        history.append(
            {
                "step": step + 1,
                "loss": loss.item(),
                "pre_clip_norm": norm.item(),
                "update_ms": elapsed,
            }
        )
    elapsed = time.perf_counter() - started
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    assert finite_tree(model.state_dict()) and finite_tree(optimizer.state_dict())
    model.eval()
    selection_mse = score(model, x[TRAIN_ROWS:], y[TRAIN_ROWS:])
    record = {
        "cell": name,
        "task": task,
        "form": form,
        "seed": seed,
        "base_rate": rate,
        "parameters": COUNTS[form],
        "train_mse": score(model, x[:TRAIN_ROWS], y[:TRAIN_ROWS]),
        "selection_mse": selection_mse,
        "initial_state_hashes": initial_hashes,
        "final_state_hashes": state_hashes(model),
        "groups": group_summary(groups),
        "steps": STEPS,
        "history": history,
        "elapsed_seconds": elapsed,
        "median_update_ms": statistics.median(h["update_ms"] for h in history[50:]),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "clip_fraction": sum(h["pre_clip_norm"] > 1 for h in history) / STEPS,
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": diagnostics(model, x[:256]),
        "final_clipped_gradient_norms": final_gradients,
        "all_final_weights_and_moments_finite": True,
        **hashes,
    }
    checkpoint = {
        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": STEPS,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "cell": name,
        **hashes,
    }
    torch.save(checkpoint, cell / "checkpoint.pt")
    record["checkpoint_sha256"] = sha256(cell / "checkpoint.pt")
    write_new(cell / "training.json", record)
    print(
        json.dumps({"cell": name, "selection_mse": selection_mse, "seconds": elapsed}), flush=True
    )
    del model, optimizer, groups, checkpoint, prediction, loss, norm
    gc.collect()
    torch.cuda.empty_cache()
    return record


@torch.no_grad()
def heldout(record, x, y):
    model = make_model(record["form"], record["seed"]).cuda().eval()
    path = ROOT / "cells" / record["cell"]
    assert sha256(path / "checkpoint.pt") == record["checkpoint_sha256"]
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    result = {"mse": score(model, x, y), "checkpoint_sha256": record["checkpoint_sha256"]}
    result["zero_mse"] = y.double().square().mean().item()
    write_new(path / "heldout.json", result)
    return result


def geometric(values):
    assert all(v > 0 and math.isfinite(v) for v in values)
    return math.exp(statistics.mean(math.log(v) for v in values))


def summarize(rows):
    by_key = {(r["task"], r["form"], r["seed"]): r for r in rows}

    def ratio(form, reference, tasks=TASKS, seeds=SEEDS):
        return geometric(
            [
                by_key[t, form, s]["heldout_mse"] / by_key[t, reference, s]["heldout_mse"]
                for t in tasks
                for s in seeds
            ]
        )

    def over_zero(form, tasks=TASKS, seeds=SEEDS):
        return geometric(
            [
                by_key[t, form, s]["heldout_mse"] / by_key[t, form, s]["zero_mse"]
                for t in tasks
                for s in seeds
            ]
        )

    positive_tests = {
        "full_gelu_linear_half_zero_every_seed": all(
            over_zero("full_gelu", tasks=("linear",), seeds=(s,)) <= 0.5 for s in SEEDS
        ),
        "full_gelu_two_generic_tasks_two_percent_better_than_zero": sum(
            over_zero("full_gelu", tasks=(t,)) <= 0.98 for t in TASKS[3:]
        )
        >= 2,
    }
    assay_passed = all(positive_tests.values())
    gates = {}
    for form in ("gelu_same", "gelu_matched"):
        tests = {
            "positive_control_assay_passes": assay_passed,
            "two_percent_over_plain": ratio(form, "plain") <= 0.98,
            "every_seed_better_than_plain": all(
                ratio(form, "plain", seeds=(s,)) < 1 for s in SEEDS
            ),
            "beats_both_calibrated_narrows": all(
                ratio(form, ref) < 1 for ref in ("narrow_swiglu", "narrow_gelu")
            ),
            "generic_regression_cap_vs_plain": all(
                ratio(form, "plain", tasks=(t,)) <= 1.05 for t in TASKS[3:]
            ),
            "generic_regression_cap_vs_zero": all(
                over_zero(form, tasks=(t,)) <= 1.05 for t in TASKS[3:]
            ),
            "linear_half_zero_every_seed": all(
                over_zero(form, tasks=("linear",), seeds=(s,)) <= 0.5 for s in SEEDS
            ),
            "seventy_percent_reduction": COUNTS[form] <= 0.3 * COUNTS["full_gelu"],
            "finite_final_state": all(r["finite"] for r in rows if r["form"] == form),
        }
        if form == "gelu_matched":
            tests["one_percent_over_same_width"] = ratio(form, "gelu_same") <= 0.99
        gates[form] = {
            "tests": tests,
            "earns_full_model_resource_qualification": all(tests.values()),
        }
    return {
        "status": "complete",
        "positive_control_tests": positive_tests,
        "assay_passed": assay_passed,
        "ratios_to_zero": {t: {f: over_zero(f, tasks=(t,)) for f in FORMS} for t in TASKS},
        "selected_rows": rows,
        "gates": gates,
        "ratios": {f: {r: ratio(f, r) for r in FORMS} for f in FORMS},
        "by_seed_ratios_to_plain": {
            f: {str(s): ratio(f, "plain", seeds=(s,)) for s in SEEDS} for f in FORMS
        },
        "task_mean_mse": {
            t: {f: statistics.mean(by_key[t, f, s]["heldout_mse"] for s in SEEDS) for f in FORMS}
            for t in TASKS
        },
        "optimizer_updates": 294 * STEPS,
        "training_example_presentations": 294 * STEPS * 256,
        "cells": 294,
        "selection_cells": 147,
        "corpus_targets": 0,
        "full_model_resource_workers": 0,
        "research_goal_achieved": False,
        "scientific_verdict": "EARNS_RESOURCE_QUALIFICATION"
        if any(g["earns_full_model_resource_qualification"] for g in gates.values())
        else ("REJECTED_AT_THIS_FITTING_BUDGET" if assay_passed else "INCONCLUSIVE_ASSAY_FAILURE"),
    }


def run():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
    data = make_data()
    assert finite_tree(data)
    data_path = ROOT / "data.pt"
    assert not data_path.exists()
    torch.save(data, data_path)
    manifest = {
        "data_sha256": sha256(data_path),
        "x_sha256": tensor_sha(data["x"]),
        "targets": {t: tensor_sha(y) for t, y in data["targets"].items()},
        "streams": {str(s): tensor_sha(v) for s, v in data["streams"].items()},
        "environment": environment(),
        "protocol_sha256": sha256(ROOT / "protocol.json"),
    }
    write_new(ROOT / "data_manifest.json", manifest)
    rows = []
    for task in TASKS:
        x, y = data["x"][:SELECT_END].cuda(), data["targets"][task][:SELECT_END].cuda()
        for seed in SEEDS:
            stream = data["streams"][seed].cuda()
            for form in FORMS:
                hashes = {
                    "data_sha256": manifest["data_sha256"],
                    "sampler_sha256": manifest["streams"][str(seed)],
                    "protocol_sha256": manifest["protocol_sha256"],
                }
                records = [
                    train_cell(task, form, seed, rate, x, y, stream, hashes) for rate in RATES
                ]
                selected = min(records, key=lambda r: (r["selection_mse"], r["base_rate"]))
                selection_path = ROOT / "selections" / f"{task}_{form}_s{seed}.json"
                write_new(
                    selection_path,
                    {
                        "selected_cell": selected["cell"],
                        "criterion": "final selection MSE, then smaller rate",
                        "values": {r["cell"]: r["selection_mse"] for r in records},
                    },
                )
                tx, ty = data["x"][SELECT_END:].cuda(), data["targets"][task][SELECT_END:].cuda()
                for record in records:
                    report = heldout(record, tx, ty)
                    if record is selected:
                        rows.append(
                            {
                                "task": task,
                                "form": form,
                                "seed": seed,
                                "cell": record["cell"],
                                "rate": record["base_rate"],
                                "selection_mse": record["selection_mse"],
                                "heldout_mse": report["mse"],
                                "zero_mse": report["zero_mse"],
                                "finite": record["all_final_weights_and_moments_finite"],
                            }
                        )
                del tx, ty, records
            del stream
        del x, y
    assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
    assert sha256(PLAN) == protocol["plan_sha256"]
    result = summarize(rows)
    result.update(protocol_sha256=manifest["protocol_sha256"], data_sha256=manifest["data_sha256"])
    write_new(ROOT / "result.json", result)
    print(
        json.dumps({k: result[k] for k in ("status", "scientific_verdict", "gates", "ratios")}),
        flush=True,
    )


if __name__ == "__main__":
    run()
