"""H094 direct-input basis screen using the established shared fitting loop."""

import gc
import json
import math
import os
import statistics as stats
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.input_basis_lift_v1.source.model import FORMS as BASIS_FORMS
from results.input_basis_lift_v1.source.model import InputBasisFFN
from results.neuron_geometry_v1.source import study as shared
from results.neuron_geometry_v1.source.model import GeometryFFN
from src.core.reproducibility import environment, provenance

ROOT = Path("results/input_basis_fit_v1")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
SEEDS, RATES = (17, 29, 43), (0.001, 0.003)
FORMS = (
    *BASIS_FORMS,
    "full_gelu",
    "full_swiglu",
    "narrow_gelu",
    "narrow_relu",
    "narrow_swiglu",
    "gelu_offset",
)
CANDIDATES = ("rational", "hermite", "trig")
CONTROLS = (
    "narrow_gelu",
    "narrow_relu",
    "narrow_swiglu",
    "gelu_offset",
    "duplicate",
    "linear",
    "antipodal",
    "raw_gelu",
)
STEPS, BATCH, TRAIN, SELECT_END, TOTAL = 300, 256, 65536, 69632, 73728


class BasisRecipe(InputBasisFFN):
    def __init__(self, form, seed=17):
        super().__init__(form, seed)
        self.curve = nn.Identity()  # Shared diagnostics interface; no parameters or forward use.

    def optimizer_groups(self, base_rate):
        return [
            {"params": [p], "lr": base_rate, "parameter_name": name}
            for name, p in self.named_parameters()
        ]


def make_model(form, seed=17):
    return BasisRecipe(form, seed) if form in BASIS_FORMS else GeometryFFN(form, seed)


def make_data():
    x = (2 * torch.rand(TOTAL, 384, generator=torch.Generator().manual_seed(9844)) - 1) * math.sqrt(
        3
    )
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(9283))
    rotation = torch.linalg.qr(
        torch.randn(384, 384, generator=torch.Generator().manual_seed(9282))
    ).Q
    a, b, c, d = (torch.roll(x[:, perm], -i, -1) for i in range(4))
    raw = {
        "smooth": torch.sin(2 * a) + b.square() + torch.exp(0.5 * c) - d,
        "oscillatory": torch.sin(3 * a) * torch.cos(2 * b) + 0.5 * torch.sin(4 * c + d),
        "multiplicative": a * b + 2 * b * c * d + a.square() * d,
        "piecewise": F.relu(a + b) - 0.7 * (c - d).abs() + torch.where(a > 0, b, c),
    }
    targets, scales = {}, {}
    for task, values in raw.items():
        values = values @ rotation
        scales[task] = values[:TRAIN].std(dim=0, correction=0)
        targets[task] = values / scales[task]
        assert torch.isfinite(targets[task]).all() and (scales[task] > 0).all()
    return {
        "x": x,
        "targets": targets,
        "scales": scales,
        "permutation": perm,
        "rotation": rotation,
        "streams": {
            seed: torch.randint(
                TRAIN, (STEPS, BATCH), generator=torch.Generator().manual_seed(20000 + seed)
            )
            for seed in SEEDS
        },
    }


def geometric(values):
    return math.exp(stats.mean(math.log(v) for v in values))


def summarize(rows, zero):
    selected = [row for row in rows if row["selected"]]
    lookup = {(r["task"], r["form"], r["seed"]): r for r in selected}

    def ratio(form, ref, seeds=SEEDS):
        return geometric(
            lookup[t, form, s]["reporting_mse"] / lookup[t, ref, s]["reporting_mse"]
            for t in TASKS
            for s in seeds
        )

    per_task, resources = {}, {}
    for form in FORMS:
        per_task[form] = {}
        for task in TASKS:
            values = [lookup[task, form, s]["reporting_mse"] for s in SEEDS]
            per_task[form][task] = {
                "mean": stats.mean(values),
                "median": stats.median(values),
                "variance": stats.variance(values),
                "sample_sd": stats.stdev(values),
                "values": values,
            }
        selected_form = [r for r in selected if r["form"] == form]
        resources[form] = {
            k: stats.mean(r[k] for r in selected_form)
            for k in (
                "median_update_ms",
                "median_forward_ms",
                "median_backward_ms",
                "clipped_fraction",
            )
        }
        resources[form]["max_peak_bytes"] = max(r["peak_bytes"] for r in selected_form)
    decisions = {}
    for form in CANDIDATES:
        ratios = {ref: ratio(form, ref) for ref in (*CONTROLS, "full_gelu", "full_swiglu")}
        seed_ratios = {ref: [ratio(form, ref, (s,)) for s in SEEDS] for ref in CONTROLS}
        gates = {
            "complete_finite_grid": len(rows) == 336 and all(r["all_finite"] for r in rows),
            "positive_assay": sum(per_task["full_gelu"][t]["mean"] <= 0.98 * zero[t] for t in TASKS)
            >= 2,
            "seventy_percent_reduction": 1 - 350208 / 1179648 >= 0.70,
            "two_percent_over_each_control": all(ratios[r] <= 0.98 for r in CONTROLS),
            "every_seed_beats_each_control": all(v < 1 for a in seed_ratios.values() for v in a),
            "per_task_regression_cap": all(
                per_task[form][t]["mean"] <= 1.05 * min(per_task[r][t]["mean"] for r in CONTROLS)
                for t in TASKS
            ),
            "full_reference_proxy": all(ratios[r] <= 1.01 for r in ("full_gelu", "full_swiglu")),
            "update_time_cap": resources[form]["median_update_ms"]
            <= 1.25 * resources["narrow_gelu"]["median_update_ms"],
        }
        decisions[form] = {
            "verdict": "EARNS_CONFIRMATION" if all(gates.values()) else "REJECTED_AT_THIS_BUDGET",
            "gates": gates,
            "ratios": ratios,
            "seed_ratios": seed_ratios,
        }
    return {
        "per_task": per_task,
        "resources": resources,
        "decisions": decisions,
        "ratios_vs_narrow_gelu": {f: ratio(f, "narrow_gelu") for f in FORMS},
    }


def configure():
    shared.ROOT, shared.STEPS, shared.GeometryFFN = ROOT, STEPS, make_model
    shared.COUNTS = {f: sum(p.numel() for p in make_model(f).parameters()) for f in FORMS}
    shared.EXTRAS = {f: 4 if f == "gelu_offset" else 0 for f in FORMS}


def run():
    configure()
    assert read("results/input_basis_recovery_v1/result.json")["status"] == "PASS"
    sources = dict(read("results/input_basis_recovery_v1/before.json")["source_hashes"])
    for p in [
        *sorted((ROOT / "source").glob("*.py")),
        Path("research/input_basis_fit_plan.md"),
        Path("results/neuron_geometry_v1/source/model.py"),
        Path("results/neuron_geometry_v1/source/study.py"),
    ]:
        sources[p.as_posix()] = sha(p)
    assert all(sha(p) == h for p, h in sources.items())
    write_json(
        ROOT / "before.json",
        {
            "source_hashes": sources,
            "qualification_sha256": sha("results/input_basis_recovery_v1/result.json"),
        },
    )
    write_json(
        ROOT / "protocol.json",
        {
            **read(ROOT / "before.json"),
            "environment": environment(),
            "provenance": provenance(),
            "forms": list(FORMS),
            "seeds": list(SEEDS),
            "rates": list(RATES),
            "steps": STEPS,
            "batch": BATCH,
            "input_seed": 9844,
        },
    )
    for folder in ("cells", "selections"):
        (ROOT / folder).mkdir()
    data = make_data()
    shared.save_tensor(ROOT / "data.pt", data)
    data_hash, protocol_hash = sha(ROOT / "data.pt"), sha(ROOT / "protocol.json")
    write_json(
        ROOT / "data_manifest.json",
        {
            "sha256": data_hash,
            "input_seed": 9844,
            "train": TRAIN,
            "selection": 4096,
            "reporting": 4096,
            "features": 384,
            "batch": BATCH,
            "updates_per_run": STEPS,
        },
    )
    all_rows, zero = [], {}
    for ti, task in enumerate(TASKS):
        x, y = data["x"].cuda(), data["targets"][task].cuda()
        zero[task] = data["targets"][task][SELECT_END:].double().square().mean().item()
        for si, seed in enumerate(SEEDS):
            stream = data["streams"][seed].cuda()
            offset = (ti * 3 + si) % len(FORMS)
            for form in FORMS[offset:] + FORMS[:offset]:
                runs = [
                    shared.train_cell(
                        task, form, seed, rate, x, y, stream, data_hash, protocol_hash
                    )
                    for rate in RATES
                ]
                chosen = min(runs, key=lambda r: r["selection_mse"])["label"]
                write_json(
                    ROOT / "selections" / f"{task}_{form}_s{seed}.json", {"selected_label": chosen}
                )
                for result in runs:
                    model = make_model(form, seed).cuda()
                    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
                    model.load_state_dict(state["model"])
                    mse = shared.score(model, x[SELECT_END:], y[SELECT_END:])
                    row = {
                        k: v
                        for k, v in result.items()
                        if k
                        not in (
                            "history",
                            "diagnostics",
                            "initial_state_sha256",
                            "optimizer_groups",
                        )
                    }
                    row.update(reporting_mse=mse, selected=result["label"] == chosen)
                    write_json(ROOT / "cells" / row["label"] / "reporting.json", row)
                    all_rows.append(row)
                    del model, state
            del stream
        del x, y
        gc.collect()
        torch.cuda.empty_cache()
    assert len(all_rows) == 336
    assert all(sha(p) == h for p, h in sources.items())
    write_json(
        ROOT / "raw_result.json",
        {
            "rows": all_rows,
            "zero_mse": zero,
            "summary": summarize(all_rows, zero),
            "data_sha256": data_hash,
            "protocol_sha256": protocol_hash,
        },
    )
    print(json.dumps(summarize(all_rows, zero)["decisions"], indent=2), flush=True)


if __name__ == "__main__":
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "fitting_process.json", status)
    started = time.perf_counter()
    try:
        run()
        status["status"] = "PASS"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        status.update(
            elapsed_seconds=time.perf_counter() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(ROOT / "fitting_process.json", status, exclusive=False)
