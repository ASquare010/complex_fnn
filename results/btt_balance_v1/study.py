"""H164 fixed-budget synthetic learning and local training-resource screen."""

import gc
import hashlib
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from results.btt_balance_v1.model import ARMS, Model

ROOT = Path("results/btt_balance_v1")
SEEDS = (431, 443, 457)
TASKS = ("teacher", "product")
RATES = (0.001, 0.003)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    p = read(ROOT / "protocol.json")
    for path, expected in p["hashes"].items():
        assert sha(path) == expected, path
    return p


def clear():
    gc.collect()
    torch.clear_autocast_cache()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0


def data(task, seed):
    generator = torch.Generator().manual_seed(seed + (10000 if task == "product" else 20000))
    x = torch.randn(6144, 64, generator=generator)
    if task == "teacher":
        up = torch.randn(64, 152, generator=generator) / 8
        down = torch.randn(152, 64, generator=generator) / math.sqrt(152)
        bias = torch.randn(152, generator=generator) * 0.25
        y = F.gelu(x @ up + bias) @ down
    else:
        y = x * x.roll(-1, dims=1)
    mean = y[:4096].mean(0)
    scale = (y[:4096] - mean).square().mean().sqrt()
    y = (y - mean) / scale
    return dict(x=x, y=y, mean=mean, scale=scale)


def prepare():
    assert read(ROOT / "check.json")["passed"]
    for task in TASKS:
        for seed in SEEDS:
            torch.save(data(task, seed), ROOT / f"data_{task}_{seed}.pt")
    paths = [
        *ROOT.glob("*.py"),
        *ROOT.glob("data_*.pt"),
        ROOT / "check.json",
        Path("research/btt_balance_plan.md"),
        Path("src/blockshuffle_ffn/__init__.py"),
        Path("src/core/structured_linear.py"),
        Path("research/references/btt_official/nn_cola_nn.py.txt"),
        Path("research/references/btt_official/ops_operators.py.txt"),
    ]
    write(
        ROOT / "protocol.json",
        dict(
            study="H164",
            created_ns=time.time_ns(),
            tasks=TASKS,
            seeds=SEEDS,
            arms=ARMS,
            rates=RATES,
            steps=600,
            hashes={p.as_posix(): sha(p) for p in paths},
        ),
    )


@torch.no_grad()
def score(model, x, y):
    total = 0.0
    for a, b in zip(x.split(256), y.split(256), strict=True):
        total += (model(a.cuda()) - b.cuda()).square().sum().item()
    return total / y.numel()


def fit(task, seed, arm, lr, dataset):
    folder = ROOT / f"{task}_{seed}_{arm}_{lr}"
    folder.mkdir(exist_ok=False)
    clear()
    model = Model(arm, seed).cuda()
    optimizer = torch.optim.AdamW(model.groups(lr), betas=(0.9, 0.999), weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 30000)
    x, y = dataset["x"], dataset["y"]
    best = float("inf")
    trace, endpoints = [], []
    for step in range(1, 601):
        ids = torch.randint(4096, (256,), generator=generator)
        bx, by = x[ids].cuda(), y[ids].cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(bx) - by).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        loss_value, norm_value = loss.item(), norm.item()
        assert math.isfinite(loss_value)
        trace.append(dict(step=step, loss=loss_value, norm=norm_value))
        del bx, by, loss, norm
        if step in (200, 400, 600):
            validation = score(model, x[4096:5120], y[4096:5120])
            endpoints.append(dict(step=step, validation=validation))
            if validation < best:
                best, best_step = validation, step
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, folder / "selected.pt")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / "final.pt")
    model.load_state_dict(state)
    report = score(model, x[5120:], y[5120:])
    result = dict(
        task=task,
        seed=seed,
        arm=arm,
        lr=lr,
        step=best_step,
        validation=best,
        report=report,
        parameters=sum(p.numel() for p in model.parameters()),
        trace=trace,
        endpoints=endpoints,
        selected_path=str(folder / "selected.pt"),
        selected_sha=sha(folder / "selected.pt"),
        final_sha=sha(folder / "final.pt"),
    )
    write(folder / "result.json", result)
    del optimizer, model, state
    clear()
    return result


def profile(row):
    clear()
    torch.cuda.reset_peak_memory_stats()
    model = Model(row["arm"], row["seed"]).cuda()
    model.load_state_dict(torch.load(row["selected_path"], weights_only=True, map_location="cpu"))
    optimizer = torch.optim.AdamW(model.groups(row["lr"]), betas=(0.9, 0.999), weight_decay=0.0)
    generator = torch.Generator().manual_seed(64000 + row["seed"])
    x = torch.randn(8192, 64, generator=generator).cuda()
    y = torch.randn(8192, 64, generator=generator).cuda()
    records = []
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        events[0].record()
        loss = (model(x) - y).square().mean()
        events[1].record()
        loss.backward()
        events[2].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events[3].record()
        events[3].synchronize()
        wall = (time.perf_counter() - start) * 1000
        records.append(
            dict(
                step=step + 1,
                wall_ms=wall,
                cuda_ms=events[0].elapsed_time(events[3]),
                forward_ms=events[0].elapsed_time(events[1]),
                backward_ms=events[1].elapsed_time(events[2]),
                optimizer_ms=events[2].elapsed_time(events[3]),
            )
        )
        del events, loss
    memory = dict(
        allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()
    )
    result = dict(
        task=row["task"],
        seed=row["seed"],
        arm=row["arm"],
        memory=memory,
        records=records,
        median_ms=st.median(r["cuda_ms"] for r in records[5:]),
        median_wall_ms=st.median(r["wall_ms"] for r in records[5:]),
        optimizer_updates=20,
    )
    del model, optimizer, x, y
    clear()
    write(ROOT / f"profile_{row['task']}_{row['seed']}_{row['arm']}.json", result)
    return result


def run():
    verify()
    results, selected, profiles = [], [], []
    for task in TASKS:
        for seed in SEEDS:
            dataset = torch.load(ROOT / f"data_{task}_{seed}.pt", weights_only=True)
            # Rotate execution order to avoid always profiling a particular arm first.
            offset = SEEDS.index(seed)
            order = ARMS[offset:] + ARMS[:offset]
            for arm in order:
                candidates = [fit(task, seed, arm, lr, dataset) for lr in RATES]
                results.extend(candidates)
                chosen = min(candidates, key=lambda r: r["validation"])
                selected.append(chosen)
                profiles.append(profile(chosen))
                print(
                    task,
                    seed,
                    arm,
                    "selected",
                    chosen["lr"],
                    chosen["step"],
                    "report",
                    chosen["report"],
                    flush=True,
                )
    verify()
    write(
        ROOT / "result.json",
        dict(
            study="H164",
            runs=results,
            selected=selected,
            profiles=profiles,
            training_updates=50400,
            profile_updates=840,
            total_updates=51240,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            tf32=False,
        ),
    )


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
