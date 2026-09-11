"""Fixed H120 recipe and storage accounting shared by probes and replay."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import copy  # noqa: E402
from pathlib import Path  # noqa: E402

from results.checkpoint_input_offload_v1.source.prepare import (  # noqa: E402, F401
    ROOT,
    hashes,
    read,
    sha,
)
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402, F401
    MemoryLedger,
    clear_boundary,
    set_blocks,
    tensor_hash,
    tree_hash,
)
from results.fp32_decoder_resource_v1.source.common import (  # noqa: E402, F401
    attention_context,
    training_loss,
)
from results.optimizer_memory_v1.source.profile import describe, inventory  # noqa: E402, F401
from src.core.config import ModelConfig, TrainConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.optimization import parameter_groups  # noqa: E402
from src.core.reproducibility import environment, write_json  # noqa: E402, F401
from src.core.transformer import Transformer  # noqa: E402


def check_inputs(p):
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])


def host_stats():
    raw = torch.cuda.memory.host_memory_stats()
    return {
        k: raw.get(k, 0)
        for k in (
            "allocated_bytes.current",
            "allocated_bytes.peak",
            "active_bytes.current",
            "active_bytes.peak",
            "allocations.current",
            "allocations.peak",
            "active_requests.current",
            "active_requests.peak",
        )
    }


def boundary():
    gpu = clear_boundary()
    host = host_stats()
    torch.cuda.memory.reset_peak_host_memory_stats()
    return dict(gpu=gpu, host=host)


def construct(f, p):
    source = torch.load(f["checkpoint"], map_location="cpu", weights_only=True)
    assert source["step"] == f["source_step"] == 800
    cfg = ModelConfig(**source["model_config"])
    assert (cfg.width, cfg.hidden, cfg.layers, cfg.heads, cfg.vocab_size) == (384, 456, 8, 6, 4096)
    model = Transformer(cfg, 101).cuda()
    model.load_state_dict(source["model"])
    model.train()
    set_blocks(model)
    tc = TrainConfig(steps=30, batch_size=8, learning_rate=0.0006, seed=101, precision="fp32")
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc),
        lr=0.0006,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=None,
        fused=None,
    )
    state = copy.deepcopy(source["optimizer"])
    for g in state["param_groups"]:
        g.update(foreach=None, fused=None)
    optimizer.load_state_dict(state)
    for g in optimizer.param_groups:
        assert g["lr"] == 0.0006 and g["eps"] == 1e-8 and tuple(g["betas"]) == (0.9, 0.95)
    data = TokenData(Path(p["datasets"][f["dataset"]]["path"]), "cuda", 10101)
    data.generator.set_state(source["sampler_state"])
    provenance = dict(
        model_hash=tree_hash(model.state_dict()),
        moments_hash=tree_hash(optimizer.state_dict()["state"]),
        sampler_before=tensor_hash(data.generator.get_state()),
        state_keys=list(model.state_dict()),
        optimizer_hash=tree_hash(optimizer.state_dict()),
        parameter_ids={k: id(v) for k, v in model.named_parameters()},
    )
    assert provenance["model_hash"] == tree_hash(source["model"])
    assert provenance["moments_hash"] == tree_hash(source["optimizer"]["state"])
    assert all(float(s["step"]) == 800 for s in optimizer.state.values())
    x, y = data.batch(8, 512)
    provenance.update(
        tokens_hash=tensor_hash(x),
        targets_hash=tensor_hash(y),
        sampler_after=tensor_hash(data.generator.get_state()),
    )
    return model, optimizer, data, x, y, provenance


def unchanged(model, optimizer, data, initial):
    assert initial["model_hash"] == tree_hash(model.state_dict())
    assert initial["optimizer_hash"] == tree_hash(optimizer.state_dict())
    assert initial["state_keys"] == list(model.state_dict())
    assert initial["parameter_ids"] == {k: id(v) for k, v in model.named_parameters()}
    assert initial["sampler_after"] == tensor_hash(data.generator.get_state())
    assert all(float(s["step"]) == 800 for s in optimizer.state.values())


def save_gradient(path, model):
    gradient = {name: par.grad.detach().cpu() for name, par in model.named_parameters()}
    assert all(bool(torch.isfinite(g).all()) for g in gradient.values())
    torch.save(gradient, path)
    return dict(path=path.as_posix(), sha256=sha(path), tree_hash=tree_hash(gradient), finite=True)


class Ledger(MemoryLedger):
    def __init__(self, model, optimizer, data):
        super().__init__()
        self.model, self.optimizer, self.data = model, optimizer, data

    def mark(self, phase, extra=()):
        row = super().mark(phase)
        row.update(inventory(self.model, self.optimizer, self.data, extra))
        row["host"] = host_stats()
        torch.cuda.memory.reset_peak_host_memory_stats()
        return row
