"""H110 fixed configuration, unchunked evaluation and isolated qualification."""

import hashlib
from dataclasses import asdict

import torch
from torch.nn import functional as F

from results.token_memory_v1.source.execution import common_loss, install
from src.core.benchmark import autocast
from src.core.config import ModelConfig, TrainConfig
from src.core.transformer import Transformer

SEEDS = (17, 29, 43)
SHAPES = ((16, 128), (8, 512))
POLICIES = ("block", "loss_chunks")
STEPS = 800


def configurations(batch, context, seed):
    model = ModelConfig(
        variant="gelu_narrow",
        width=384,
        hidden=456,
        layers=8,
        heads=6,
        groups=8,
        vocab_size=4096,
        context=context,
    )
    train = TrainConfig(
        steps=STEPS,
        batch_size=batch,
        learning_rate=0.0006,
        weight_decay=0.1,
        seed=seed,
        precision="bf16",
        device="cuda",
    )
    assert model.total_parameters == 9099648 and model.unique_ffn_parameters == 2801664
    assert train.ffn_width_init_mode == train.ffn_width_lr_mode == "none"
    return model, train


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def state_hashes(model):
    return {k: tensor_hash(v) for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, data, batch, context, batches):
    model.eval()
    total, targets = 0.0, 0
    order = hashlib.sha256()
    for x, y in data.validation(batch, context, batches):
        with autocast("cuda", "bf16"):
            value = common_loss(model, x, y)
        total += value.item() * y.numel()
        targets += y.numel()
        order.update(tensor_hash(x).encode())
    model.train()
    return {"nll": total / targets, "targets": targets, "order_sha256": order.hexdigest()}


def qualify():
    checks = []
    for device in ("cpu", "cuda"):
        for batch, context in ((2, 7), (1, 11)):
            config = ModelConfig(
                variant="gelu_narrow",
                width=24,
                hidden=48,
                layers=2,
                heads=3,
                groups=8,
                vocab_size=32,
                context=context,
            )
            x = torch.arange(batch * context, device=device).reshape(batch, context) % 31
            reference = None
            for policy in POLICIES:
                model = Transformer(config, 17).to(device)
                if device == "cpu":
                    model.double()
                identities = {k: id(v) for k, v in model.named_parameters()}
                keys = tuple(model.state_dict())
                install(model, policy, 5)
                assert identities == {k: id(v) for k, v in model.named_parameters()}
                assert tuple(model.state_dict()) == keys
                with autocast(device, "bf16" if device == "cuda" else "fp32"):
                    loss = (
                        F.cross_entropy(model(x).flatten(0, 1), (x + 1).flatten())
                        if device == "cpu" and policy == "block"
                        else model.loss(x, x + 1)
                    )
                loss.backward()
                grads = {k: v.grad.detach().cpu() for k, v in model.named_parameters()}
                if reference is None:
                    reference = loss.item(), grads
                errors = {
                    k: (
                        (g.double() - reference[1][k].double()).norm()
                        / reference[1][k].double().norm().clamp_min(1e-8)
                    ).item()
                    for k, g in grads.items()
                }
                error = abs(loss.item() / reference[0] - 1)
                assert max(errors.values()) <= (1e-9 if device == "cpu" else 0.02)
                assert error <= (1e-9 if device == "cpu" else 0.001)
                restored = Transformer(config, 17).to(device)
                if device == "cpu":
                    restored.double()
                restored.load_state_dict(model.state_dict())
                model.eval()
                restored.eval()
                with torch.no_grad(), autocast(device, "bf16" if device == "cuda" else "fp32"):
                    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
                del restored
                checks.append(
                    {
                        "device": device,
                        "shape": [batch, context],
                        "policy": policy,
                        "loss_relative_error": error,
                        "max_gradient_relative_error": max(errors.values()),
                        "parameter_identity_preserved": True,
                        "config": asdict(config),
                    }
                )
                del model, loss, grads
    return checks
