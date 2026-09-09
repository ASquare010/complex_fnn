"""H073 harness qualification before fitting or observed target errors."""

import copy

import torch

from results.ungated_fit_v1.source.study import (
    COUNTS,
    FORMS,
    HIDDEN,
    SEEDS,
    TASKS,
    TRAIN_ROWS,
    make_model,
    optimizer_groups,
    scale_targets,
    summarize,
    tensor_sha,
)


def test_counts_and_shared_factor_initialization():
    torch.set_num_threads(4)
    for seed in SEEDS:
        models = {form: make_model(form, seed) for form in FORMS}
        for form, model in models.items():
            assert sum(p.numel() for p in model.parameters()) == COUNTS[form]
        for name, p in models["gelu_same"].state_dict().items():
            assert torch.equal(p, models["plain"].state_dict()[name])
        assert models["gelu_same"].gate is None and models["gelu_matched"].gate is None


def test_optimizer_coverage_and_calibration():
    for form in FORMS:
        model = make_model(form, 17)
        groups = optimizer_groups(model, form, 0.001)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        assert set(ids) == {id(p) for p in model.parameters()}
        if form.startswith("narrow_"):
            down = next(g for g in groups if any(p is model.down.weight for p in g["params"]))
            expected = (1536 if "gelu" in form else 1024) / HIDDEN[form]
            assert down["lr_scale"] == expected
        if form in ("plain", "gelu_same", "gelu_matched"):
            assert {g["lr_scale"] for g in groups} == {4.0, HIDDEN[form] / 96}
        assert all(g["weight_decay"] == 0 for g in groups)


def test_scaling_uses_train_rows_only():
    y = torch.tensor([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]])
    scaled, std = scale_targets(y, 2)
    y[-1] *= 500
    scaled2, std2 = scale_targets(y, 2)
    assert torch.equal(std, torch.tensor([1.0, 2.0]))
    assert torch.equal(std, std2) and torch.equal(scaled[:2], scaled2[:2])
    assert not torch.equal(scaled[:2].mean(0), torch.zeros(2))


def test_saved_stream_contract():
    assert TRAIN_ROWS == 65536
    for seed in SEEDS:
        a = torch.randint(
            TRAIN_ROWS, (300, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        b = torch.randint(
            TRAIN_ROWS, (300, 256), generator=torch.Generator().manual_seed(20000 + seed)
        )
        assert torch.equal(a, b) and tensor_sha(a) == tensor_sha(b)
        assert tensor_sha(a) != tensor_sha(a.flatten())
        assert a.min() >= 0 and a.max() < TRAIN_ROWS


def test_assay_and_absolute_error_block_promotion():
    rows = []
    for task in TASKS:
        for form in FORMS:
            for seed in SEEDS:
                mse = {
                    "plain": 1.0,
                    "narrow_swiglu": 0.95,
                    "narrow_gelu": 0.95,
                    "full_swiglu": 0.9,
                    "full_gelu": 0.9,
                    "gelu_same": 0.9,
                    "gelu_matched": 0.87,
                }[form]
                if task == "linear" and form in ("full_gelu", "gelu_same", "gelu_matched"):
                    mse = 0.39 if form == "gelu_matched" else 0.4
                rows.append(
                    {
                        "task": task,
                        "form": form,
                        "seed": seed,
                        "heldout_mse": mse,
                        "zero_mse": 1.0,
                        "finite": True,
                    }
                )
    valid = summarize(rows)
    assert valid["assay_passed"] and all(
        g["earns_full_model_resource_qualification"] for g in valid["gates"].values()
    )
    bad = copy.deepcopy(rows)
    next(r for r in bad if (r["task"], r["form"], r["seed"]) == ("linear", "full_gelu", 29))[
        "heldout_mse"
    ] = 0.6
    result = summarize(bad)
    assert result["scientific_verdict"] == "INCONCLUSIVE_ASSAY_FAILURE"
    assert not any(g["earns_full_model_resource_qualification"] for g in result["gates"].values())
    bad = copy.deepcopy(rows)
    for row in bad:
        if row["task"] == "oscillatory" and row["form"] in ("plain", "gelu_same"):
            row["heldout_mse"] = 1.2 if row["form"] == "plain" else 1.06
    result = summarize(bad)
    assert result["gates"]["gelu_same"]["tests"]["generic_regression_cap_vs_plain"]
    assert not result["gates"]["gelu_same"]["tests"]["generic_regression_cap_vs_zero"]
    assert not result["gates"]["gelu_same"]["earns_full_model_resource_qualification"]
