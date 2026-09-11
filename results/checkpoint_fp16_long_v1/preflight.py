"""Check all six initialization-scale gradients before long training."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch, boundary
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.fp32_classifier_profile_v1.source.common import tensor_hash, tree_hash
from results.ordinary_long_training_v1.io import write_json
from results.checkpoint_fp16_long_v1.numerics import compare
from results.checkpoint_fp16_v1.codec import compress_inputs
from src.core.transformer import Transformer
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.training_memory import buffer_model_loss
from contextlib import nullcontext
from pathlib import Path

ROOT = Path("results/checkpoint_fp16_long_v1")


def one(f, p, half, index):
    source = torch.load(f["checkpoint"], map_location="cpu", weights_only=True)
    assert source["step"] == 0 and not source["optimizer"]["state"]
    model = Transformer(ModelConfig(**f["model_config"]), f["seed"]).cuda()
    model.load_state_dict(source["model"])
    model.set_recompute_scope("block")
    data = TokenData(Path(p["datasets"][f["dataset"]]["path"]), "cuda", f["seed"] + 10000)
    x, y = data.batch(16, 512)
    with compress_inputs(model, half=True) if half else nullcontext(None) as logs:
        loss = buffer_model_loss(model, x, y) if half else model.loss(x, y)
        loss.backward()
    gradients = {n: v.grad.detach().cpu() for n, v in model.named_parameters()}
    assert all(torch.isfinite(v).all() for v in gradients.values()) and torch.isfinite(loss)
    assert tree_hash(model.state_dict()) == tree_hash(source["model"])
    assert all("forward" not in b.__dict__ for b in model.blocks)
    if half:
        assert len(logs["records"]) == logs["unpack_count"][0] == 8
    path = ROOT / f"preflight_gradient{index:02d}.pt"
    assert not path.exists()
    torch.save(gradients, path)
    return dict(
        dataset=f["dataset"],
        seed=f["seed"],
        half=half,
        path=path.as_posix(),
        sha256=sha(path),
        loss=loss.item(),
        tokens_hash=tensor_hash(x),
        targets_hash=tensor_hash(y),
    )


p = read(ROOT / "protocol.json")
assert not (ROOT / "preflight.json").exists()
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
bounds = [boundary()]
rows = []
checks = []
for f in p["fixtures"]:
    peers = []
    for half in (False, True):
        row = one(f, p, half, len(rows))
        rows.append(row)
        peers.append(row)
        bounds.append(boundary())
    native, candidate = peers
    assert (
        native["tokens_hash"] == candidate["tokens_hash"]
        and native["targets_hash"] == candidate["targets_hash"]
    )
    check = compare(
        torch.load(candidate["path"], weights_only=True),
        torch.load(native["path"], weights_only=True),
    )
    loss_error = abs(candidate["loss"] / native["loss"] - 1)
    checks.append(
        dict(
            dataset=f["dataset"],
            seed=f["seed"],
            gradient=check,
            loss_relative_error=loss_error,
            passed=check["passed"] and loss_error <= 1e-6,
        )
    )
    print(f["dataset"], f["seed"], checks[-1]["passed"], flush=True)
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in bounds)
write_json(
    ROOT / "preflight.json",
    dict(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        artifacts=rows,
        boundaries=bounds,
        gpu_backwards=12,
        optimizer_updates=0,
    ),
)
assert all(c["passed"] for c in checks)
