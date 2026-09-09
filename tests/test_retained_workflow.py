"""End-to-end checks for the retained models and retired execution paths."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.blockshuffle_ffn import BlockShuffleFFN
from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import sha256
from src.core.serving import PackedGateFFN, pack_gate_factors
from src.core.structured_linear import GroupedLinear
from src.core.trainer import train
from src.core.transformer import Transformer


@pytest.mark.parametrize("shape", [(8, 12, 2), (12, 8, 4)])
def test_shared_grouped_projection_against_dense_and_gradients(shape):
    source, target, groups = shape
    layer = GroupedLinear(source, target, groups).double()
    x = torch.randn(2, 3, source, dtype=torch.float64, requires_grad=True)
    expected = x @ torch.block_diag(*layer.weight.unbind()).T
    torch.testing.assert_close(layer(x), expected, rtol=1e-12, atol=1e-12)
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (x, layer.weight), retain_graph=True
    )
    actual_grads = torch.autograd.grad(layer(x).square().sum(), (x, layer.weight))
    for a, b in zip(expected_grads, actual_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "variant",
    [
        "bezier_shared",
        "overcomplete_headwise_swiglu",
        "blockshuffle_swiglu_affine_activation",
        "swiglu_narrow_rational",
        "blockshuffle_swiglu_rational",
    ],
)
def test_retired_model_is_rejected_instead_of_falling_back_to_dense(variant):
    with pytest.raises(ValueError, match="retired variant"):
        Transformer(ModelConfig(variant=variant))


@pytest.mark.parametrize("backend", ["inductor_rational", "inductor_rational_correction"])
def test_rejected_training_backends_cannot_be_selected(backend):
    with pytest.raises(ValueError, match="failed quality"):
        TrainConfig(activation_backend=backend).validate()


def test_kept_recipe_files_are_loadable_and_have_exact_counts():
    for path in Path("configs").glob("*.json"):
        raw = json.loads(path.read_text())
        cfg, tc = ModelConfig(**raw["model"]), TrainConfig(**raw["training"])
        cfg.validate()
        tc.validate()
        model = Transformer(cfg)
        assert sum(p.numel() for p in model.parameters()) == cfg.total_parameters
        assert (
            sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
            == cfg.unique_ffn_parameters
        )


def test_plain_packing_rejects_custom_subclass_without_partial_conversion():
    class CustomFFN(BlockShuffleFFN):
        pass

    model = Transformer(
        ModelConfig(
            variant="blockshuffle_swiglu",
            width=24,
            heads=3,
            layers=2,
            hidden=48,
            context=8,
            vocab_size=32,
        )
    )
    model.blocks[1].ffn = CustomFFN(24, 48, 8)
    original = [b.ffn for b in model.blocks]
    with pytest.raises(ValueError, match="learned activations"):
        PackedGateFFN(original[1], torch.float32)
    with pytest.raises(ValueError, match="learned activations"):
        pack_gate_factors(model, torch.float32)
    assert [b.ffn for b in model.blocks] == original


@pytest.mark.parametrize("variant", ["swiglu", "blockshuffle_swiglu"])
def test_retained_trainer_writes_reloadable_checkpoint(tmp_path, variant):
    cache = tmp_path / "data"
    cache.mkdir()
    for split in ("train", "valid"):
        np.save(cache / f"{split}.npy", (np.arange(129) % 32).astype(np.uint16))
    manifest = {"vocab_size": 32, "files": {p.name: sha256(p) for p in cache.glob("*.npy")}}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    cfg = ModelConfig(
        variant=variant, vocab_size=32, width=24, layers=2, heads=3, context=8, hidden=48
    )
    tc = TrainConfig(
        steps=2,
        batch_size=2,
        eval_batches=1,
        log_every=1,
        device="cpu",
        precision="fp32",
        recompute_gate=variant.startswith("blockshuffle"),
        gate_recompute_method="native" if variant == "blockshuffle_swiglu" else "checkpoint",
        ffn_lr_mode="fan_in" if variant.startswith("blockshuffle") else "uniform",
    )
    output = tmp_path / "run"
    result = train(cfg, tc, cache, output)
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**checkpoint["model_config"]), tc.seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    x = torch.arange(16).reshape(2, 8)
    assert model.loss(x, x + 1).item() == result["validation_loss"]
    assert checkpoint["step"] == 2 and result["training_tokens"] == 32
    assert len((output / "history.jsonl").read_text().splitlines()) == 2
    assert (output / "source.zip").is_file()
    assert not (output / "failure.json").exists()
    with pytest.raises(FileExistsError):
        train(cfg, tc, cache, output)
