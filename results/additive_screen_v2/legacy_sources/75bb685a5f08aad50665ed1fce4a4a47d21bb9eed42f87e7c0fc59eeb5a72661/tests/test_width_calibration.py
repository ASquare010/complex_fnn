"""Dense-width controls must preserve reference behavior and matrix decay."""

from dataclasses import replace

import pytest
import torch

from src.core.config import ModelConfig, TrainConfig
from src.core.optimization import initialize_dense_width, parameter_groups
from src.core.transformer import Transformer


def small_config(variant="swiglu_narrow", hidden=12):
    return ModelConfig(
        variant=variant, hidden=hidden, width=24, heads=3, layers=2, vocab_size=32, context=8
    )


def test_default_is_noop_and_full_reference_is_an_exact_fixed_point():
    for tc, cfg in (
        (TrainConfig(), small_config()),
        (
            TrainConfig(
                ffn_width_init_mode="fan_in", ffn_width_lr_mode="fan_in", ffn_decay_mode="product"
            ),
            small_config("swiglu", 64),
        ),
    ):
        model = Transformer(cfg)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        initialize_dense_width(model, tc)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        assert all(g["lr_scale"] == 1 for g in parameter_groups(model, tc))


def test_calibration_changes_only_ffn_down_initialization():
    model = Transformer(small_config())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    record = initialize_dense_width(model, TrainConfig(ffn_width_init_mode="fan_in"))
    assert record["initial_down_gain"] == pytest.approx((64 / 12) ** 0.5)
    for name, parameter in model.named_parameters():
        gain = (64 / 12) ** 0.5 if name.endswith(".ffn.down.weight") else 1
        torch.testing.assert_close(parameter, before[name] * gain, atol=0, rtol=0)


def test_lr_changes_only_down_and_product_decay_preserves_exact_pure_shrinkage():
    model = Transformer(small_config()).double()
    tc = TrainConfig(ffn_width_lr_mode="fan_in", ffn_decay_mode="product", learning_rate=0.01)
    groups = parameter_groups(model, tc)
    lookup = {id(p): g for g in groups for p in g["params"]}
    assert len(lookup) == sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        group = lookup[id(parameter)]
        ratio = 64 / 12 if name.endswith(".ffn.down.weight") else 1
        assert group["lr_scale"] == pytest.approx(ratio)
        expected_decay = tc.weight_decay / ratio if parameter.ndim >= 2 else 0
        assert group["weight_decay"] == pytest.approx(expected_decay)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for name, p in model.named_parameters():
        expected = (
            before[name] * (1 - tc.learning_rate * tc.weight_decay) if p.ndim >= 2 else before[name]
        )
        torch.testing.assert_close(p, expected, atol=1e-14, rtol=1e-14)


def test_unsupported_architectures_reject_implicit_calibration():
    for variant in ("blockshuffle_swiglu", "paired_antipodal_swiglu", "shared_swiglu"):
        cfg = replace(small_config(), variant=variant, hidden=32)
        with pytest.raises(ValueError, match="unshared dense"):
            initialize_dense_width(Transformer(cfg), TrainConfig(ffn_width_init_mode="fan_in"))
