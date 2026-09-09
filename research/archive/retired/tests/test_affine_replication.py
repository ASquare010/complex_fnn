"""Small-sample replication statistics and frozen seed overrides."""

import json
from dataclasses import asdict, replace

import pytest

from src.core.affine_longer import configuration, output_path
from src.core.affine_replication import paired_statistics
from src.core.config import ModelConfig, TrainConfig


def test_three_wins_do_not_imply_sign_test_significance():
    result = paired_statistics([-0.1, -0.2, -0.3])
    assert result["mean"] == pytest.approx(-0.2)
    assert result["sample_sd"] == pytest.approx(0.1)
    assert result["one_sided_sign_p"] == 0.125
    assert result["student_t_95_interval"][1] > 0
    tied = paired_statistics([-0.1, 0, -0.2])
    assert tied["nonzero_pairs"] == 2 and tied["one_sided_sign_p"] == 0.25


def test_seed_override_preserves_all_other_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    mc = ModelConfig(variant="blockshuffle_swiglu_affine_activation")
    tc = TrainConfig(steps=800, learning_rate=0.0012, seed=17, log_every=200)
    path = tmp_path / "configs/wikitext2_blockshuffle_affine_800.json"
    path.write_text(json.dumps({"model": asdict(mc), "training": asdict(tc)}))
    assert configuration("blockshuffle_affine") == (mc, tc)
    assert configuration("blockshuffle_affine", 29) == (mc, replace(tc, seed=29))
    assert (
        output_path("blockshuffle_affine", 43).name
        == "wikitext2_blockshuffle_affine_lr1200_s43_800"
    )
