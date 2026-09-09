"""Shared H088 fitting loop and reporting; no candidate-specific training path."""

import gc
import hashlib
import json
import math
import os
import statistics
import time
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import equal_payload, read, sha, write_json
from results.neuron_geometry_v1.source.model import (
    CANDIDATES,
    CONTROLS,
    COUNTS,
    EXTRAS,
    FORMS,
    GeometryFFN,
    twist,
)

ROOT = Path("results/neuron_geometry_v1")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS, RATES, STEPS, BATCH = (17, 29, 43), (0.001, 0.003), 600, 256
TRAIN, SELECT_END, TOTAL, WIDTH = 65536, 69632, 73728, 384
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def save_tensor(path, payload):
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    equal_payload(torch.load(path, map_location="cpu", weights_only=True), payload)


def make_data():
    x = (
        2 * torch.rand(TOTAL, WIDTH, generator=torch.Generator().manual_seed(9814)) - 1
    ) * math.sqrt(3)
    permutation = torch.randperm(WIDTH, generator=torch.Generator().manual_seed(9283))
    rotation = torch.linalg.qr(
        torch.randn(WIDTH, WIDTH, generator=torch.Generator().manual_seed(9282))
    ).Q
    a, b, c, d = (torch.roll(x[:, permutation], -offset, -1) for offset in range(4))
    raw = {
        "smooth": torch.sin(2 * a) + b.square() + torch.exp(0.5 * c) - d,
        "oscillatory": torch.sin(3 * a) * torch.cos(2 * b) + 0.5 * torch.sin(4 * c + d),
        "multiplicative": a * b + 2 * b * c * d + a.square() * d,
        "piecewise": F.relu(a + b) - 0.7 * (c - d).abs() + torch.where(a > 0, b, c),
    }
    targets, scales = {}, {}
    for task, values in raw.items():
        rotated = values @ rotation
        scales[task] = rotated[:TRAIN].std(dim=0, correction=0)
        targets[task] = rotated / scales[task]
        assert torch.isfinite(targets[task]).all() and (scales[task] > 0).all()
    streams = {
        seed: torch.randint(
            TRAIN, (STEPS, BATCH), generator=torch.Generator().manual_seed(20000 + seed)
        )
        for seed in SEEDS
    }
    return {
        "x": x,
        "targets": targets,
        "scales": scales,
        "streams": streams,
        "permutation": permutation,
        "rotation": rotation,
    }


@torch.inference_mode()
def score(model, x, y):
    total = 0.0
    for start in range(0, len(x), BATCH):
        prediction = model(x[start : start + BATCH])
        total += F.mse_loss(prediction, y[start : start + BATCH], reduction="sum").item()
    return total / y.numel()


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
            "near_zero_fraction": (value.abs() < 1e-6).float().mean().item(),
            "finite": bool(torch.isfinite(value).all()),
        }
    if hasattr(model.curve, "shape_parameters"):
        summary["shape"] = model.curve.shape_parameters()
        theta = model.curve.theta.detach()
        summary["theta"] = theta.cpu().tolist()
        summary["control_saturation_fraction"] = (theta.tanh().abs() > 0.98).float().mean().item()
    return summary


def train_cell(task, form, seed, rate, x, y, stream, data_sha, protocol_sha):
    label = f"{task}_{form}_s{seed}_lr{int(rate * 1e6)}"
    folder = ROOT / "cells" / label
    folder.mkdir()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GeometryFFN(form, seed).cuda()
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
        "matrix_forward_flops_per_example": 2 * (COUNTS[form] - EXTRAS[form]),
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


def geometric(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(selected, all_rows, zero_mse):
    lookup = {(row["task"], row["form"], row["seed"]): row for row in selected}
    rate_lookup = {(r["task"], r["form"], r["seed"], r["rate"]): r for r in all_rows}

    def ratio(form, reference, seeds=SEEDS):
        return geometric(
            lookup[task, form, seed]["reporting_mse"]
            / lookup[task, reference, seed]["reporting_mse"]
            for task in TASKS
            for seed in seeds
        )

    per_task, resources = {}, {}
    for form in FORMS:
        per_task[form] = {}
        for task in TASKS:
            values = [lookup[task, form, seed]["reporting_mse"] for seed in SEEDS]
            per_task[form][task] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "variance": statistics.variance(values),
                "sample_sd": statistics.stdev(values),
                "values": values,
            }
        rows = [lookup[task, form, seed] for task in TASKS for seed in SEEDS]
        resources[form] = {
            "mean_median_update_ms": statistics.mean(row["median_update_ms"] for row in rows),
            "mean_median_forward_ms": statistics.mean(row["median_forward_ms"] for row in rows),
            "mean_median_backward_ms": statistics.mean(row["median_backward_ms"] for row in rows),
            "max_peak_bytes": max(row["peak_bytes"] for row in rows),
            "mean_clipped_fraction": statistics.mean(row["clipped_fraction"] for row in rows),
            "upper_rate_selected": sum(row["rate"] == RATES[-1] for row in rows),
        }
    assay = sum(per_task["full_gelu"][task]["mean"] <= 0.98 * zero_mse[task] for task in TASKS) >= 2
    decisions = {}
    for form in CANDIDATES:
        ratios = {
            ref: ratio(form, ref) for ref in (*CONTROLS, "twist_fixed", "full_gelu", "full_swiglu")
        }
        seed_ratios = {ref: [ratio(form, ref, (seed,)) for seed in SEEDS] for ref in CONTROLS}
        gates = {
            "positive_assay": assay,
            "complete_finite_grid": len(all_rows) == 336 and all(r["all_finite"] for r in all_rows),
            "seventy_percent_reduction": 1 - COUNTS[form] / 1179648 >= 0.70,
            "two_percent_over_each_narrow_control": all(ratios[ref] <= 0.98 for ref in CONTROLS),
            "every_seed_better_than_each_control": all(
                r < 1 for rows in seed_ratios.values() for r in rows
            ),
            "per_task_regression_cap": all(
                per_task[form][task]["mean"]
                <= 1.05 * min(per_task[ref][task]["mean"] for ref in CONTROLS)
                for task in TASKS
            ),
            "learned_twist_beats_fixed": not form.startswith("twist_")
            or ratios["twist_fixed"] <= 0.99,
        }
        matched = {
            str(rate): {
                ref: geometric(
                    rate_lookup[task, form, seed, rate]["reporting_mse"]
                    / rate_lookup[task, ref, seed, rate]["reporting_mse"]
                    for task in TASKS
                    for seed in SEEDS
                )
                for ref in CONTROLS
            }
            for rate in RATES
        }
        decisions[form] = {
            "gates": gates,
            "selected_ratios": ratios,
            "seed_ratios": seed_ratios,
            "matched_rate_ratios": matched,
            "verdict": "EARNS_RESOURCE_STUDY" if all(gates.values()) else "REJECTED_AT_THIS_BUDGET",
        }
    return {
        "positive_assay": assay,
        "per_task": per_task,
        "resources": resources,
        "decisions": decisions,
        "all_ratios_vs_narrow_gelu": {form: ratio(form, "narrow_gelu") for form in FORMS},
    }


def plot_shapes(selected, cells):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "plots"
    folder.mkdir()
    selected_lookup = {(row["task"], row["form"], row["seed"]): row for row in selected}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        row = selected_lookup[task, "twist_learned", 17]
        samples = cells[row["label"]]["diagnostics"]
        for group in range(4):
            ax.plot(
                [sample["step"] for sample in samples],
                [sample["shape"]["amplitude"][group] for sample in samples],
                marker="o",
                label=f"Group {group + 1}",
            )
        ax.set(
            title=task,
            xlabel="Optimizer updates",
            ylabel="Rotation amplitude a",
            ylim=(-2.05, 2.05),
        )
    axes.flat[0].legend()
    fig.suptitle("Learned pair geometry: seed 17, independently selected rates")
    for extension in ("png", "svg"):
        fig.savefig(folder / f"twist_evolution.{extension}", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, form in zip(axes[:2], ("bezier_1p", "bump_2p")):
        row = selected_lookup["smooth", form, 17]
        state = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"]
        model = GeometryFFN(form)
        model.load_state_dict(state)
        groups = model.curve.groups
        grid = torch.linspace(-0.5, 1.5, 300)
        values = model.curve(grid[:, None].repeat(1, groups)).detach()
        ax.plot(grid, F.relu(grid), "k--", label="Initial ReLU")
        for group in range(groups):
            ax.plot(grid, values[:, group], label=f"Group {group + 1}")
        ax.set(title=form, xlabel="Input", ylabel="Activation")
        ax.legend(fontsize=8)
    row = selected_lookup["smooth", "twist_learned", 17]
    theta = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"][
        "curve.theta"
    ][:1]
    u, v = torch.meshgrid(torch.linspace(-2, 3, 19), torch.linspace(-2.5, 2.5, 19), indexing="xy")
    points = torch.stack((u, v), -1)
    displacement = twist(points, theta) - points
    axes[2].quiver(
        u, v, displacement[..., 0], displacement[..., 1], angles="xy", scale_units="xy", scale=1
    )
    axes[2].scatter([1], [0], c="red", s=20)
    axes[2].set(
        title="Learned twist: group 1", xlabel="Feature 1", ylabel="Feature 2", aspect="equal"
    )
    fig.suptitle("Learned activations on smooth task, seed 17; not model predictions")
    for extension in ("png", "svg"):
        fig.savefig(folder / f"learned_geometry.{extension}", dpi=140)
    plt.close(fig)


def main():
    protocol_path = ROOT / "protocol.json"
    protocol = read(protocol_path)
    assert all(sha(name) == expected for name, expected in protocol["source_hashes"].items())
    assert read(ROOT / "qualification_process.json")["status"] == "PASS"
    assert not (ROOT / "data.pt").exists() and not list((ROOT / "cells").iterdir())
    data = make_data()
    save_tensor(ROOT / "data.pt", data)
    data_sha, protocol_sha = sha(ROOT / "data.pt"), sha(protocol_path)
    write_json(
        ROOT / "data_manifest.json",
        {
            "sha256": data_sha,
            "input_seed": 9814,
            "train_rows": TRAIN,
            "selection_rows": SELECT_END - TRAIN,
            "reporting_rows": TOTAL - SELECT_END,
            "width": WIDTH,
            "batch": BATCH,
            "steps": STEPS,
            "streams": {str(seed): tensor_sha(value) for seed, value in data["streams"].items()},
        },
    )
    selected, all_rows, cells, zero_mse, orders = [], [], {}, {}, {}
    for task_index, task in enumerate(TASKS):
        x, y = data["x"].cuda(), data["targets"][task].cuda()
        zero_mse[task] = data["targets"][task][SELECT_END:].double().square().mean().item()
        for seed_index, seed in enumerate(SEEDS):
            stream = data["streams"][seed].cuda()
            offset = (task_index * 3 + seed_index) % len(FORMS)
            order = FORMS[offset:] + FORMS[:offset]
            orders[f"{task}_{seed}"] = list(order)
            for form in order:
                results = [
                    train_cell(task, form, seed, rate, x, y, stream, data_sha, protocol_sha)
                    for rate in RATES
                ]
                chosen = min(results, key=lambda row: row["selection_mse"])
                write_json(
                    ROOT / "selections" / f"{task}_{form}_s{seed}.json",
                    {
                        "selected_label": chosen["label"],
                        "selected_rate": chosen["rate"],
                        "selection_mse_by_rate": {
                            str(row["rate"]): row["selection_mse"] for row in results
                        },
                    },
                )
                for result in results:
                    model = GeometryFFN(form, seed).cuda()
                    checkpoint = torch.load(
                        result["checkpoint"], map_location="cpu", weights_only=True
                    )
                    model.load_state_dict(checkpoint["model"])
                    reporting = score(model, x[SELECT_END:], y[SELECT_END:])
                    row = {
                        name: value
                        for name, value in result.items()
                        if name
                        not in (
                            "history",
                            "diagnostics",
                            "initial_state_sha256",
                            "optimizer_groups",
                        )
                    }
                    row.update(reporting_mse=reporting, selected=result["label"] == chosen["label"])
                    write_json(ROOT / "cells" / result["label"] / "reporting.json", row)
                    cells[row["label"]] = result
                    all_rows.append(row)
                    if row["selected"]:
                        selected.append(row)
                    del model, checkpoint
            del stream
        del x, y
        gc.collect()
        torch.cuda.empty_cache()
    assert len(all_rows) == 336 and len(selected) == 168
    summary = summarize(selected, all_rows, zero_mse)
    plot_shapes(selected, cells)
    assert all(sha(name) == expected for name, expected in protocol["source_hashes"].items())
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE_PENDING_INDEPENDENT_AUDIT",
            "research_goal_achieved": False,
            "runs": 336,
            "updates": 201600,
            "training_example_presentations": 51609600,
            "selected": selected,
            "all_rates": all_rows,
            "summary": summary,
            "zero_mse": zero_mse,
            "form_orders": orders,
            "data_sha256": data_sha,
            "protocol_sha256": protocol_sha,
            "source_archive_sha256": sha(ROOT / "source.zip"),
            "raw_cell_record_hashes": {
                p.as_posix(): sha(p) for p in (ROOT / "cells").rglob("*.json")
            },
            "plot_hashes": {p.as_posix(): sha(p) for p in (ROOT / "plots").iterdir()},
        },
    )
    print(json.dumps(summary["decisions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
