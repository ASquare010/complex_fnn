"""Fairness guards for configuration and result comparisons."""

import copy
import json
from argparse import Namespace

from src.core.cli import build_configs
from src.core.report import cohort, dominates


def test_file_config_with_explicit_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": {"variant": "swiglu", "width": 96, "heads": 3},
                "training": {"steps": 50, "seed": 29, "learning_rate": 0.001},
            }
        )
    )
    model, train = build_configs(Namespace(config=path, steps=60, seed=None), 4096)
    assert model.variant == "swiglu" and model.width == 96
    assert train.steps == 60 and train.seed == 29 and train.learning_rate == 0.001


def test_cohorts_do_not_mix_budget_or_data():
    m = {
        "model": {"variant": "gelu", "hidden": 0, "groups": 8, "width": 192},
        "training": {"steps": 200, "seed": 17},
        "precision": "bf16",
        "data": {"files": {"train": "hash"}},
        "protocol": "micro_v1",
        "environment": {"gpu": "gpu", "torch": "version", "cuda": "cuda", "python": "python"},
    }
    other = copy.deepcopy(m)
    other["model"]["variant"] = "bezier_grouped"
    other["training"]["seed"] = 29
    assert cohort(m) == cohort(other)
    for key in ("ffn_width_init_mode", "ffn_width_lr_mode"):
        calibrated = copy.deepcopy(m)
        calibrated["training"][key] = "none"
        assert cohort(m) == cohort(calibrated)
        calibrated["training"][key] = "fan_in"
        assert cohort(m) != cohort(calibrated)
    explicit = copy.deepcopy(m)
    explicit["training"]["ffn_lr_mode"] = "uniform"
    assert cohort(m) == cohort(explicit)
    explicit["training"]["ffn_lr_mode"] = "fan_in"
    assert cohort(m) != cohort(explicit)
    other["training"]["steps"] = 201
    assert cohort(m) != cohort(other)
    other = copy.deepcopy(m)
    other["data"]["files"]["train"] = "different"
    assert cohort(m) != cohort(other)


def test_pareto_requires_same_seed_and_all_objectives():
    a = {
        "cohort": "x",
        "seed": 17,
        "validation_loss": 3.0,
        "total_parameters": 100,
        "peak_allocated_vram_bytes": 50,
        "inference_tokens_per_second": 10.0,
    }
    b = {**a, "validation_loss": 4.0}
    assert dominates(a, b)
    assert not dominates(a, {**b, "seed": 29})
    assert not dominates(a, {**b, "inference_tokens_per_second": 11.0})
    assert not dominates(a, a)
