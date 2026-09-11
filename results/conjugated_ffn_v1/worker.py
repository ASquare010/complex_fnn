"""Controlled synthetic falsification; data stay on CPU, models run sequentially."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary, tensor_hash
from results.conjugated_ffn_v1.model import Model
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import statistics as st
import time
import math

ROOT = Path("results/conjugated_ffn_v1")
F = torch.nn.functional


def dataset(task, seed):
    index = ("linear", "gelu_teacher", "cubic").index(task)
    g = torch.Generator().manual_seed(10000 + 1000 * index + seed)
    x = torch.randn(8192, 384, generator=g)
    q = torch.linalg.qr(torch.randn(384, 384, generator=g)).Q
    if task == "linear":
        y = x @ q
    elif task == "gelu_teacher":
        u = torch.randn(384, 384, generator=g) / math.sqrt(384)
        y = F.gelu(x @ u) @ q
    else:
        y = (x * torch.roll(x, 1, 1) * torch.roll(x, 2, 1)) @ q
    xm = x[:4096].mean(0)
    xs = x[:4096].std(0, correction=0).clamp_min(1e-6)
    ym = y[:4096].mean(0)
    ys = (y[:4096] - ym).square().mean().sqrt()
    x = (x - xm) / xs
    y = (y - ym) / ys
    stream = torch.randint(
        4096, (300, 128), generator=torch.Generator().manual_seed(40000 + 1000 * index + seed)
    )
    return dict(x=x, y=y, stream=stream, stats=dict(xmean=xm, xstd=xs, ymean=ym, yscale=ys))


@torch.no_grad()
def score(model, x, y):
    total = 0.0
    for start in range(0, len(x), 512):
        a, b = x[start : start + 512].cuda(), y[start : start + 512].cuda()
        total += F.mse_loss(model(a), b, reduction="sum").item()
    return total / y.numel()


def one(arm, task, seed, data, index):
    model = Model(arm, seed=seed).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0)
    count = sum(p.numel() for p in model.parameters())
    initial = score(model, data["x"][4096:6144], data["y"][4096:6144])
    records = []
    for step, ids in enumerate(data["stream"], 1):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        x, y = data["x"][ids].cuda(), data["y"][ids].cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - begin)
        assert math.isfinite(loss.item())
        records.append(dict(step=step, loss=loss.item(), grad_norm=norm.item(), wall_ms=elapsed))
    validation = score(model, data["x"][4096:6144], data["y"][4096:6144])
    reporting = score(model, data["x"][6144:], data["y"][6144:])
    model.eval()
    with torch.no_grad():
        output = model(data["x"][6144:6656].cuda()).cpu().double()
    rank = int(torch.linalg.matrix_rank(output - output.mean(0), rtol=1e-5))
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert all(
        bool(torch.isfinite(s[k]).all())
        for s in optimizer.state.values()
        for k in ("exp_avg", "exp_avg_sq")
    )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path = ROOT / f"fit{index:02d}.pt"
    assert not path.exists()
    torch.save(state, path)
    result = dict(
        index=index,
        task=task,
        seed=seed,
        arm=arm,
        parameters=count,
        macs=model.mac_per_token,
        fixed_buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
        initial_validation=initial,
        validation=validation,
        reporting=reporting,
        zero_reporting=data["y"][6144:].square().mean().item(),
        output_rank=rank,
        peak_bytes=torch.cuda.max_memory_allocated(),
        reserved_bytes=torch.cuda.max_memory_reserved(),
        median_update_ms=st.median(v["wall_ms"] for v in records[20:]),
        stream_hash=tensor_hash(data["stream"]),
        history=records,
        all_finite=True,
        state_path=path.as_posix(),
        state_sha256=sha(path),
    )
    write_json(ROOT / f"fit{index:02d}.json", result)
    return result


def run():
    p = read(ROOT / "protocol.json")
    hashes(p["sources"])
    hashes(p["maintained_files"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    boundaries = [boundary()]
    results = []
    for task in p["tasks"]:
        for seed in p["seeds"]:
            data = dataset(task, seed)
            path = ROOT / f"data_{task}_{seed}.pt"
            assert not path.exists()
            torch.save(data, path)
            # Rotate order deterministically to avoid always timing candidate last.
            shift = len(results) // 6 % 6
            order = p["arms"][shift:] + p["arms"][:shift]
            for arm in order:
                row = one(arm, task, seed, data, len(results))
                results.append(row)
                boundaries.append(boundary())
                write_json(
                    ROOT / "progress.json", dict(completed=len(results), boundaries=boundaries)
                )
                print(row["index"], task, seed, arm, round(row["reporting"], 5), flush=True)
            del data
    hashes(p["sources"])
    hashes(p["maintained_files"])
    assert len(results) == 54
    # Preserve original per-fit JSON without large-object re-encoding.
    with (ROOT / "result.json").open("x", encoding="utf-8") as out:
        out.write('{"fits":[')
        for i in range(54):
            if i:
                out.write(",")
            out.write((ROOT / f"fit{i:02d}.json").read_text().strip())
        out.write('],"training_updates":16200}\n')
    write_json(ROOT / "boundaries.json", boundaries)


if __name__ == "__main__":
    run()
