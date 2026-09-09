"""Keep duration evidence separate from convergence and optimizer changes."""

from dataclasses import replace

from src.core.long_duration import RECIPES, configuration, decisions, trajectory_screen
from src.core.optimizer_bracket import configuration as short_configuration
from src.core.trainer import learning_rate


def test_long_duration_changes_only_steps_and_logging():
    for r in RECIPES:
        mc, tc = configuration(r)
        old_mc, old_tc = short_configuration(r, 0.0012)
        assert mc == old_mc and tc == replace(old_tc, steps=3200, log_every=800)
        assert tc.steps * tc.batch_size * mc.context == 6553600
        assert learning_rate(0, tc) == learning_rate(0, old_tc) / 4
        assert abs(learning_rate(3199, tc) - 0.00012) < 1e-15


def test_declining_loss_and_rebound_cannot_be_called_a_plateau():
    h = [
        {"step": 800, "validation_loss": 5.0},
        {"step": 1600, "validation_loss": 4.8},
        {"step": 2400, "validation_loss": 4.6},
        {"step": 3200, "validation_loss": 4.4},
    ]
    assert not trajectory_screen(h)["late_plateau_screen_passes"]
    h[-1]["validation_loss"] = 4.601
    assert trajectory_screen(h)["late_plateau_screen_passes"]
    h[1]["validation_loss"] = 4.0
    assert not trajectory_screen(h)["late_plateau_screen_passes"]
    assert not decisions({})["full_promotion_passes"]
