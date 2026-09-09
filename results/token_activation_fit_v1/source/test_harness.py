"""Harness checks without optimization or target-error inspection."""

import torch

from results.token_activation_fit_v1.source.study import (
    COUNTS,
    FORMS,
    SEEDS,
    make_model,
    optimizer_groups,
    scale_targets,
    tensor_sha,
)


def test_counts_and_shared_initial_functions():
    torch.set_num_threads(4)
    x = torch.randn(8, 48, generator=torch.Generator().manual_seed(700))
    for seed in SEEDS:
        plain = make_model("plain", seed)
        for form in FORMS:
            model = make_model(form, seed)
            assert sum(p.numel() for p in model.parameters()) == COUNTS[form]
            if form in ("static", "dynamic"):
                assert torch.equal(model(x), plain(x))
                for n, p in plain.state_dict().items():
                    assert torch.equal(p, model.state_dict()[n])


def test_calibrated_optimizer_covers_parameters_once():
    for form in FORMS:
        model = make_model(form, 17)
        groups = optimizer_groups(model, form, 0.001)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        assert set(ids) == {id(p) for p in model.parameters()}
        if form == "narrow":
            down = next(g for g in groups if any(p is model.down.weight for p in g["params"]))
            assert down["lr_scale"] == 128 / 38
        if form in ("plain", "static", "dynamic"):
            assert {g["lr_scale"] for g in groups} >= {4.0, 256 / 12}


def test_scale_uses_training_rows_only():
    y = torch.tensor([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]])
    scaled, std = scale_targets(y, 2)
    altered = y.clone()
    altered[-1] *= 500
    scaled2, std2 = scale_targets(altered, 2)
    assert torch.equal(std, torch.tensor([1.0, 2.0]))
    assert torch.equal(std, std2) and torch.equal(scaled[:2], scaled2[:2])
    assert not torch.equal(scaled[:2].mean(0), torch.zeros(2))


def test_sampler_reproducibility_and_shape_sensitive_hash():
    for seed in SEEDS:
        a = torch.randint(4096, (300, 256), generator=torch.Generator().manual_seed(20000 + seed))
        b = torch.randint(4096, (300, 256), generator=torch.Generator().manual_seed(20000 + seed))
        assert torch.equal(a, b) and tensor_sha(a) == tensor_sha(b)
        assert tensor_sha(a) != tensor_sha(a.flatten())
