"""Recomputation must preserve actual optimizer behavior, not only a formula."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from src.core.config import VARIANTS, ModelConfig, TrainConfig
from src.core.optimization import group_summary, parameter_groups
from src.core.trainer import configure_gate_recomputation
from src.core.transformer import Transformer


def snapshot(model, optimizer):
    return {
        "weights": {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
        "moments": {
            (i, k): v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for i, state in optimizer.state_dict()["state"].items()
            for k, v in state.items()
        },
    }


def equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            equal(left[k], right[k])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    else:
        assert left == right


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_nonzero_shapes_gradients_and_two_updates(variant, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    config = ModelConfig(
        variant=variant, width=24, layers=2, heads=3, hidden=48, groups=3, vocab_size=32, context=8
    )
    training = TrainConfig(
        ffn_lr_mode="fan_in",
        ffn_decay_mode="parameter",
        recompute_gate=variant.startswith("blockshuffle"),
        gate_recompute_method="checkpoint",
    )
    x = torch.arange(16, device=device).reshape(2, 8)
    for seed in (17, 29, 43):
        reference = None
        for scope in ("none", "ffn", "block"):
            model = Transformer(config, seed).to(device)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if "theta" in n:
                        p.copy_(
                            torch.linspace(-0.3, 0.3, p.numel(), device=device).reshape_as(p)
                            + seed / 1000
                        )
            configure_gate_recomputation(model, training)
            model.set_recompute_scope(scope)
            groups = parameter_groups(model, training)
            optimizer = torch.optim.AdamW(groups, lr=0.0006, betas=(0.9, 0.95), eps=1e-8)
            record = {
                "groups": group_summary(groups),
                "initial": snapshot(model, optimizer),
                "steps": [],
            }
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits, loss = model(x), model.loss(x, x + 1)
                loss.backward()
                gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
                record["steps"].append(
                    {
                        "logits": logits.detach().cpu(),
                        "loss": loss.item(),
                        "gradients": gradients,
                        "norm": norm.item(),
                        "state": snapshot(model, optimizer),
                    }
                )
            if reference is None:
                reference = record
            else:
                equal(reference, record)


@pytest.mark.parametrize("scope", ["ffn", "block"])
def test_evaluation_bypass_parameter_identity_and_invalid_scope(scope):
    model = Transformer(ModelConfig(width=24, layers=2, heads=3, vocab_size=32, context=8), 17)
    identities = {n: id(p) for n, p in model.named_parameters()}
    keys = set(model.state_dict())
    model.set_recompute_scope(scope)
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    assert keys == set(model.state_dict())
    with pytest.raises(ValueError, match="scope"):
        model.set_recompute_scope("invalid")
    assert all(b.recompute_scope == scope for b in model.blocks)
    with patch("src.core.transformer.checkpoint", side_effect=AssertionError("Must bypass")):
        model.eval()(torch.ones(1, 2, dtype=torch.long))
        with torch.no_grad():
            model.train()(torch.ones(1, 2, dtype=torch.long))
    state = model.state_dict()
    restored = Transformer(replace(model.config), 17)
    restored.load_state_dict(state, strict=True)
    assert all(b.recompute_scope == "none" for b in restored.blocks)


def test_resource_promotion_requires_equally_checkpointed_full_controls():
    from src.core.native_recompute_audit import resource_gates

    candidate = {"scope": "ffn", "peak_allocated_bytes": 600, "ffn_reduction_percent": 70.3}
    native = {"scope": "none", "peak_allocated_bytes": 1000}
    references = {
        "full_swiglu": {"scope": "ffn", "peak_allocated_bytes": 600},
        "full_gelu": {"scope": "ffn", "peak_allocated_bytes": 500},
    }
    gates = resource_gates(candidate, native, references)
    assert gates["at_least_ten_percent_native_memory_reduction"]
    assert gates["same_scope_memory_within_ten_percent_full_swiglu"]
    assert not gates["same_scope_memory_within_ten_percent_full_gelu"]
    references["full_gelu"] = {"scope": "none", "peak_allocated_bytes": 900}
    with pytest.raises(ValueError, match="same recomputation scope"):
        resource_gates(candidate, native, references)
