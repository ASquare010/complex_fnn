"""H161 bounded scale probe. Run via the UV-managed Python; no source mutation."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("results/exact_offload_scale_v1")
PLAN = Path("research/exact_offload_scale_plan.md")
CONFIG = dict(
    variant="gelu", width=512, hidden=608, layers=12, heads=8, context=512, vocab_size=4096
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def verify():
    p = read(ROOT / "protocol.json")
    for path, expected in p["hashes"].items():
        assert sha(path) == expected, path
    return p


def prepare():
    paths = [
        Path(__file__),
        Path("results/exact_offload_scale_v1/audit.py"),
        PLAN,
        *Path("src").rglob("*.py"),
        Path("results/streamed_evaluation_v1/source/evaluation.py"),
    ]
    paths += [f for f in Path("data/wikitext2_v1").iterdir() if f.is_file()]
    schedule = []
    for i, seed in enumerate((401, 409, 419)):
        for arm in ("ordinary", "helper") if i % 2 == 0 else ("helper", "ordinary"):
            schedule.append(dict(index=len(schedule), seed=seed, arm=arm))
    write(
        ROOT / "protocol.json",
        dict(
            config=CONFIG,
            schedule=schedule,
            hashes={f.as_posix(): sha(f) for f in paths},
            frozen_at_ns=time.time_ns(),
            updates=180,
            initial_backwards=6,
        ),
    )


def torch_setup():
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    return torch


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(model):
    return hashlib.sha256(
        json.dumps(
            {k: tensor_hash(v) for k, v in model.state_dict().items()}, sort_keys=True
        ).encode()
    ).hexdigest()


def evaluate(model, data, native=False):
    import torch
    from torch.nn import functional as F

    from results.streamed_evaluation_v1.source.evaluation import batch_loss

    was = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in data.validation(16, 512, 10**9):
            if native:
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            else:
                loss = batch_loss(model, x, y, "classifier_chunks")
            total += loss.item() * y.numel()
            count += y.numel()
    model.train(was)
    return dict(nll=total / count, targets=count)


def worker(index):
    from contextlib import nullcontext

    from src.core.config import ModelConfig, TrainConfig
    from src.core.data import TokenData
    from src.core.optimization import parameter_groups
    from src.core.training_memory import buffer_model_loss, offload_checkpoint_inputs
    from src.core.transformer import Transformer

    torch = torch_setup()
    from torch.nn import functional as F

    p = verify()
    row = p["schedule"][index]
    folder = ROOT / f"case{index:02d}"
    assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory.reset_peak_host_memory_stats()
    model = Transformer(ModelConfig(**p["config"]), row["seed"]).cuda().train()
    model.set_recompute_scope("block")
    initial_hash = state_hash(model)
    tc = TrainConfig(steps=30, batch_size=16, precision="fp32")
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=0.0006, betas=(0.9, 0.95), eps=1e-8
    )
    data = TokenData(Path("data/wikitext2_v1"), "cuda", row["seed"] + 10000)
    sampler = data.generator.get_state()
    initial_score = evaluate(model, data)

    def loss_fn(x, y, model=model):
        if row["arm"] == "helper":
            return buffer_model_loss(model, x, y)
        return F.cross_entropy(model(x).flatten(0, 1), y.flatten())

    context = offload_checkpoint_inputs(model, 4) if row["arm"] == "helper" else nullcontext()
    with context:
        x, y = data.batch(16, 512)
        loss = loss_fn(x, y)
        loss.backward()
        initial_loss = loss.item()
        gradients = {name: v.grad.detach().cpu() for name, v in model.named_parameters()}
        assert all(torch.isfinite(v).all() for v in gradients.values())
        torch.save(gradients, folder / "initial_gradients.pt")
        del gradients, loss, x, y
        data.generator.set_state(sampler)
        records = []
        for step in range(30):
            x, y = data.batch(16, 512)
            batch_hash = tensor_hash(x) + tensor_hash(y)
            optimizer.zero_grad(set_to_none=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            events[0].record()
            loss = loss_fn(x, y)
            events[1].record()
            loss.backward()
            events[2].record()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            events[3].record()
            events[3].synchronize()
            wall = (time.perf_counter() - start) * 1000
            record = dict(
                step=step + 1,
                loss=loss.item(),
                norm=norm.item(),
                wall_ms=wall,
                cuda_ms=events[0].elapsed_time(events[3]),
                forward_ms=events[0].elapsed_time(events[1]),
                backward_ms=events[1].elapsed_time(events[2]),
                optimizer_ms=events[2].elapsed_time(events[3]),
                batch_hash=batch_hash,
            )
            records.append(record)
            with (folder / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            del loss, norm, x, y, events
    assert all("forward" not in b.__dict__ for b in model.blocks)
    final_score = evaluate(model, data)
    assert all(torch.isfinite(v).all() for v in model.parameters())
    assert all(
        torch.isfinite(v).all()
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    )
    torch.cuda.synchronize()
    memory = dict(
        allocated=torch.cuda.max_memory_allocated(),
        reserved=torch.cuda.max_memory_reserved(),
        host=torch.cuda.memory.host_memory_stats(),
    )
    model_hash = state_hash(model)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / "final.pt")
    result = dict(
        **row,
        parameters=sum(v.numel() for v in model.parameters()),
        initial_hash=initial_hash,
        final_hash=model_hash,
        initial_loss=initial_loss,
        initial_score=initial_score,
        final_score=final_score,
        records=records,
        memory=memory,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        workspace_bytes=torch.backends.cuda.cublas_workspace_size(),
        artifacts={name: sha(folder / name) for name in ("initial_gradients.pt", "final.pt")},
    )
    del model, optimizer, data, sampler, context, loss_fn
    gc.collect()
    torch.cuda.empty_cache()
    result["boundary"] = [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()]
    assert result["boundary"] == [0, 0]
    verify()
    write(folder / "result.json", result)
    print(row, flush=True)


def run():
    p = verify()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + str(Path.cwd() / ".venv/Lib/site-packages")
    env["PYTHONMALLOC"] = "pymalloc"
    env["PYTHONHASHSEED"] = "161"
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    for row in p["schedule"]:
        folder = ROOT / f"case{row['index']:02d}"
        folder.mkdir(exist_ok=False)
        with (
            (folder / "telemetry.csv").open("x") as out,
            (folder / "telemetry.err").open("x") as err,
        ):
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=out,
                stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                with (folder / "worker.log").open("x") as log:
                    code = subprocess.call(
                        [
                            sys.executable,
                            "-B",
                            "-X",
                            "pycache_prefix=" + str(ROOT / "unused_cache"),
                            "-u",
                            str(Path(__file__)),
                            "worker",
                            str(row["index"]),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        (folder / "exit.txt").write_text(str(code))
        assert code == 0, (row, code)
        print("complete", row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "worker"])
    parser.add_argument("index", type=int, nargs="?")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.index)
    elif args.mode == "prepare":
        prepare()
    else:
        run()
