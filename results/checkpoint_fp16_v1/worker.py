"""Clean backward-memory probes and separate full-input diagnostics; no updates."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary, host_stats
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.fp32_classifier_profile_v1.source.common import tree_hash, tensor_hash
from results.ordinary_long_training_v1.io import write_json
from results.checkpoint_fp16_v1.codec import compress_inputs
from src.core.training_memory import buffer_model_loss, offload_checkpoint_inputs
from src.core.transformer import Transformer
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import parameter_groups
from src.core.data import TokenData
from src.core.reproducibility import environment
from contextlib import nullcontext
from pathlib import Path
import os

ROOT = Path("results/checkpoint_fp16_v1")


def build(f, p):
    s = torch.load(f["checkpoint"], map_location="cpu", weights_only=True)
    assert (
        s["step"] == 800
        and tree_hash(s["model"]) == f["model_hash"]
        and tree_hash(s["optimizer"]) == f["optimizer_hash"]
    )
    cfg = ModelConfig(**s["model_config"])
    m = Transformer(cfg, f["seed"]).cuda()
    m.load_state_dict(s["model"])
    m.set_recompute_scope("block")
    tc = TrainConfig(
        steps=30, batch_size=16, learning_rate=0.0006, seed=f["seed"], precision="fp32"
    )
    opt = torch.optim.AdamW(parameter_groups(m, tc), lr=0.0006, betas=(0.9, 0.95))
    opt.load_state_dict(s["optimizer"])
    assert tree_hash(opt.state_dict()) == f["optimizer_hash"]
    data = TokenData(Path(p["datasets"][f["dataset"]]["path"]), "cuda", f["seed"] + 10000)
    data.generator.set_state(s["sampler_state"])
    assert tensor_hash(data.generator.get_state()) == f["sampler_hash"]
    x, y = data.batch(16, 512)
    return m, opt, data, x, y


def one(f, p, arm, index):
    m, opt, data, x, y = build(f, p)
    before_peak = torch.cuda.max_memory_allocated()
    before_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    ctx = (
        compress_inputs(m, half=arm == "fp16")
        if arm in ("identity", "fp16")
        else offload_checkpoint_inputs(m, 4)
        if arm == "offload4"
        else nullcontext(None)
    )
    with ctx as logs:
        loss = m.loss(x, y) if arm.startswith("native") else buffer_model_loss(m, x, y)
        loss.backward()
    torch.cuda.synchronize()
    backward_peak = torch.cuda.max_memory_allocated()
    grads = {n: v.grad.detach().cpu() for n, v in m.named_parameters()}
    assert all(bool(torch.isfinite(v).all()) for v in grads.values()) and bool(torch.isfinite(loss))
    assert all("forward" not in b.__dict__ for b in m.blocks)
    assert (
        tree_hash(m.state_dict()) == f["model_hash"]
        and tree_hash(opt.state_dict()) == f["optimizer_hash"]
    )
    if logs is not None:
        assert len(logs["records"]) == logs["unpack_count"][0] == 8 and not logs["originals"]
    path = ROOT / f"gradient{index:02d}.pt"
    assert not path.exists()
    torch.save(grads, path)
    row = dict(
        index=index,
        dataset=f["dataset"],
        seed=f["seed"],
        arm=arm,
        loss=loss.item(),
        tokens_hash=tensor_hash(x),
        targets_hash=tensor_hash(y),
        model_hash=f["model_hash"],
        optimizer_hash=f["optimizer_hash"],
        parameter_count=sum(v.numel() for v in m.parameters()),
        gradient_path=path.as_posix(),
        gradient_sha256=sha(path),
        construction_peak_bytes=before_peak,
        backward_peak_bytes=backward_peak,
        peak_bytes=max(before_peak, torch.cuda.max_memory_allocated()),
        reserved_bytes=max(before_reserved, torch.cuda.max_memory_reserved()),
        host=host_stats(),
        hook_records=[] if logs is None else logs["records"],
        unpacks=0 if logs is None else logs["unpack_count"][0],
    )
    write_json(ROOT / f"case{index:02d}.json", row)
    return row


def diagnostic(f, p, index):
    m, opt, data, x, y = build(f, p)
    with compress_inputs(m, half=True, capture=True) as logs:
        loss = buffer_model_loss(m, x, y)
    assert len(logs["records"]) == len(logs["originals"]) == 8 and logs["unpack_count"][0] == 0
    assert bool(torch.isfinite(loss))
    rows = []
    for v in logs["originals"]:
        a = v["input"]
        q = a.half().float()
        assert torch.isfinite(q).all()
        rows.append(
            dict(
                block=v["index"],
                max_abs=a.abs().max().item(),
                relative_l2=((q - a).double().norm() / a.double().norm()).item(),
                underflow_count=int(((a != 0) & (q == 0)).sum()),
                zero_count=int((q == 0).sum()),
                finite=True,
            )
        )
    path = ROOT / f"inputs{index:02d}.pt"
    torch.save(logs["originals"], path)
    return dict(
        dataset=f["dataset"], seed=f["seed"], path=path.as_posix(), sha256=sha(path), blocks=rows
    )


def run():
    p = read(ROOT / "protocol.json")
    assert (
        read(ROOT / "checks.json")["passed"]
        and (ROOT / "prepare_exit.txt").read_text().strip() == "0"
    )
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    assert torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    env = environment()
    rows = []
    diags = []
    bounds = [boundary()]
    for i, f in enumerate(p["fixtures"]):
        for arm in p["arms"]:
            row = one(f, p, arm, len(rows))
            rows.append(row)
            bounds.append(boundary())
            print(
                f["dataset"],
                f["seed"],
                arm,
                "peakMiB",
                round(row["peak_bytes"] / 2**20, 2),
                flush=True,
            )
        diags.append(diagnostic(f, p, i))
        bounds.append(boundary())
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    write_json(
        ROOT / "result.json",
        dict(
            cases=rows,
            diagnostics=diags,
            boundaries=bounds,
            environment=env,
            gpu_backwards=36,
            diagnostic_forwards=6,
            training_updates=0,
        ),
    )


if __name__ == "__main__":
    run()
