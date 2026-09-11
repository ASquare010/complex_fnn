"""Generate the predeclared CPU initial states without touching CUDA."""

from dataclasses import asdict
from pathlib import Path

import torch

from results.fp32_classifier_profile_v1.source.common import cpu_tree, tree_hash
from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import parameter_groups
from src.core.reproducibility import sha256
from src.core.transformer import Transformer


def initial_state(fixture):
    cfg = ModelConfig(**fixture["model_config"])
    tc = TrainConfig(
        steps=800, batch_size=8, learning_rate=0.0006, seed=fixture["seed"], precision="bf16"
    )
    model = Transformer(cfg, fixture["seed"])
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    assert not optimizer.state
    return dict(
        model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),
        model_config=asdict(cfg),
        training_config=asdict(tc),
        step=0,
        seed=fixture["seed"],
    )


def generate_initials(protocol):
    assert not torch.cuda.is_initialized()
    rows = []
    for fixture in protocol["fixtures"]:
        path = Path(fixture["checkpoint"])
        path.parent.mkdir(parents=True, exist_ok=False)
        state = initial_state(fixture)
        torch.save(state, path)
        rows.append(
            dict(
                fixture=fixture["label"],
                path=path.as_posix(),
                sha256=sha256(path),
                model_hash=tree_hash(state["model"]),
                optimizer_hash=tree_hash(state["optimizer"]),
                seed=fixture["seed"],
                step=0,
                optimizer_updates=0,
            )
        )
    assert not torch.cuda.is_initialized()
    return dict(
        initializations=rows, initialized_on_cpu=True, cuda_initialized=False, optimizer_updates=0
    )
