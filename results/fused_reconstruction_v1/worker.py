"""Sequential resource preflight with upstream gradients and full optimizer steps."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary, tensor_hash
from results.fp32_classifier_profile_v1.source.common import cpu_tree
from results.fused_reconstruction_v1.model import Model
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import statistics as st
import time
import os

ROOT = Path("results/fused_reconstruction_v1")
F = torch.nn.functional


def make_data(batch, seed):
    x = torch.randn(batch, 384, generator=torch.Generator().manual_seed(50000 + batch + seed))
    q = torch.linalg.qr(
        torch.randn(384, 384, generator=torch.Generator().manual_seed(60000 + seed))
    ).Q
    return dict(x=x, y=x @ q)


def one(arm, batch, seed, data, index):
    model = Model(arm, seed=seed).cuda()
    x = data["x"].cuda().requires_grad_()
    y = data["y"].cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.95), weight_decay=0)
    history = []
    diagnostics = []
    for step in range(1, 13):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin = time.perf_counter()
        events[0].record()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        events[1].record()
        events[1].synchronize()
        wall = 1000 * (time.perf_counter() - begin)
        history.append(
            dict(
                step=step,
                loss=loss.item(),
                preclip_norm=norm.item(),
                wall_ms=wall,
                event_ms=events[0].elapsed_time(events[1]),
            )
        )
        if step in (1, 12):
            diagnostics.append(
                dict(
                    step=step,
                    input_gradient_norm=x.grad.detach().cpu().norm().item(),
                    postclip_parameter_norms={
                        k: v.grad.detach().cpu().norm().item() for k, v in model.named_parameters()
                    },
                )
            )
    with torch.no_grad():
        final_loss = F.mse_loss(model(x), y).item()
    state = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()))
    assert all(bool(torch.isfinite(v).all()) for v in state["model"].values())
    assert all(
        float(v["step"]) == 12
        and all(bool(torch.isfinite(v[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
        for v in state["optimizer"]["state"].values()
    )
    path = ROOT / f"case{index:02d}.pt"
    assert not path.exists()
    torch.save(state, path)
    timed = history[4:]
    times = {
        k: dict(
            mean=st.mean(v[k] for v in timed),
            median=st.median(v[k] for v in timed),
            sample_variance=st.variance(v[k] for v in timed),
        )
        for k in ("wall_ms", "event_ms")
    }
    stability = {
        k: max(st.median(v[k] for v in timed[:4]), st.median(v[k] for v in timed[4:]))
        / min(st.median(v[k] for v in timed[:4]), st.median(v[k] for v in timed[4:]))
        for k in times
    }
    result = dict(
        index=index,
        arm=arm,
        batch=batch,
        seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        optimizer_bytes=sum(
            v.numel() * v.element_size()
            for s in optimizer.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        ),
        buffer_bytes=sum(v.numel() * v.element_size() for v in model.buffers()),
        macs=model.macs,
        peak_bytes=torch.cuda.max_memory_allocated(),
        reserved_bytes=torch.cuda.max_memory_reserved(),
        input_hash=tensor_hash(x),
        target_hash=tensor_hash(y),
        history=history,
        diagnostics=diagnostics,
        timing=times,
        stability=stability,
        initial_loss=history[0]["loss"],
        final_loss=final_loss,
        finite=True,
        state_path=path.as_posix(),
        state_sha256=sha(path),
    )
    write_json(ROOT / f"case{index:02d}.json", result)
    return result


def run():
    p = read(ROOT / "protocol.json")
    assert read(ROOT / "checks.json")["passed"]
    assert (ROOT / "check_exit.txt").read_text().strip() == "0"
    hashes(p["sources"])
    hashes(p["maintained_files"])
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    assert torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    torch.use_deterministic_algorithms(False)
    boundaries = [boundary()]
    cases = []
    for batch in p["batches"]:
        for seed in p["seeds"]:
            data = make_data(batch, seed)
            torch.save(data, ROOT / f"data_{batch}_{seed}.pt")
            shift = len(cases) // 10 % 10
            order = p["arms"][shift:] + p["arms"][:shift]
            for arm in order:
                row = one(arm, batch, seed, data, len(cases))
                cases.append(row)
                boundaries.append(boundary())
                print(
                    row["index"], batch, seed, arm, round(row["peak_bytes"] / 2**20, 3), flush=True
                )
    hashes(p["sources"])
    hashes(p["maintained_files"])
    write_json(
        ROOT / "result.json",
        dict(cases=cases, boundaries=boundaries, training_updates=720, backwards=720),
    )


if __name__ == "__main__":
    run()
