"""Balanced small learning screen; validation selects rates only after every fit."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.fp32_classifier_profile_v1.source.common import cpu_tree
from results.ordinary_long_training_v1.io import write_json
from results.readout_learning_v1.model import Model
from results.readout_learning_v1.data import make_data
from pathlib import Path
import time

ROOT = Path("results/readout_learning_v1")
F = torch.nn.functional


@torch.no_grad()
def score(model, x, y):
    total = 0.0
    for start in range(0, len(x), 512):
        error = model(x[start : start + 512].cuda()) - y[start : start + 512].cuda()
        total += error.double().square().sum().item()
    return total / y.numel()


def one(task, seed, arm, rate, data, index):
    model = Model(arm, seed).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=rate, betas=(0.9, 0.95), weight_decay=0)
    x, y = data["x"][:4096].cuda(), data["y"][:4096].cuda()
    order = data["order"].cuda()
    history = []
    start = time.perf_counter()
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x[order[step]]), y[order[step]])
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if (step + 1) % 25 == 0:
            history.append(dict(step=step + 1, loss=loss.item(), gradient_norm=grad.item()))
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    val = score(model, data["x"][4096:6144], data["y"][4096:6144])
    report = score(model, data["x"][6144:], data["y"][6144:])
    state = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(opt.state_dict()))
    assert all(bool(torch.isfinite(v).all()) for v in state["model"].values())
    path = ROOT / f"case{index:03d}.pt"
    assert not path.exists()
    torch.save(state, path)
    row = dict(
        index=index,
        task=task,
        seed=seed,
        arm=arm,
        rate=rate,
        steps=300,
        parameters=sum(p.numel() for p in model.parameters()),
        buffer_bytes=sum(v.numel() * v.element_size() for v in model.buffers()),
        history=history,
        wall_seconds=wall,
        peak_bytes=torch.cuda.max_memory_allocated(),
        validation_mse=val,
        reporting_mse=report,
        zero_reporting_mse=data["y"][6144:].double().square().mean().item(),
        state_path=path.as_posix(),
        state_sha256=sha(path),
    )
    write_json(ROOT / f"case{index:03d}.json", row)
    return row


def run():
    p = read(ROOT / "protocol.json")
    assert (
        read(ROOT / "checks.json")["passed"]
        and (ROOT / "prepare_exit.txt").read_text().strip() == "0"
    )
    hashes(p["sources"])
    cases = []
    bounds = [boundary()]
    for ti, task in enumerate(p["tasks"]):
        for si, seed in enumerate(p["seeds"]):
            data = make_data(task, seed)
            torch.save(data, ROOT / f"data_{task}_{seed}.pt")
            shift = (ti * 3 + si) % len(p["arms"])
            arms = p["arms"][shift:] + p["arms"][:shift]
            for arm in arms:
                for rate in p["rates"]:
                    row = one(task, seed, arm, rate, data, len(cases))
                    cases.append(row)
                    bounds.append(boundary())
                    print(
                        row["index"],
                        task,
                        seed,
                        arm,
                        rate,
                        "val",
                        round(row["validation_mse"], 5),
                        flush=True,
                    )
    hashes(p["sources"])
    hashes(p["maintained_files"])
    write_json(
        ROOT / "result.json",
        dict(cases=cases, boundaries=bounds, training_updates=59400, backwards=59400),
    )


if __name__ == "__main__":
    run()
