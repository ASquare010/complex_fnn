"""Protect disjoint target ranges and partial-batch weighting."""

import math

import torch

from src.core.validation_tail import per_window_losses, window_batches


def test_complete_windows_cover_disjoint_targets_with_a_partial_batch():
    tokens = torch.arange(26)
    prefix = list(window_batches(tokens, 4, 3, 0, 2))
    tail = list(window_batches(tokens, 4, 3, 2, 6))
    a = torch.cat([y.flatten() for _, x, y in prefix]).tolist()
    b = torch.cat([y.flatten() for _, x, y in tail]).tolist()
    assert a == list(range(1, 9)) and b == list(range(9, 25))
    assert not set(a) & set(b)
    assert tail[-1][2].shape == (1, 4)
    assert tail[0][1][0, 0].item() == a[-1]


def test_window_losses_and_aggregate_match_analytic_binary_loss():
    tokens = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])

    def forward(x):
        return torch.stack((torch.zeros_like(x).float(), x.float() / 2), -1)

    mean, count, rows = per_window_losses(
        forward, window_batches(tokens, 4, 3, 0, 4), "cpu", "fp32"
    )
    expected = [
        math.log1p(math.exp(float(x) / 2)) - float(y) * float(x) / 2
        for x, y in zip(tokens[:16], tokens[1:17], strict=True)
    ]
    assert count == 16 and len(rows) == 4
    assert abs(mean - sum(expected) / 16) < 1e-7
    assert rows[0]["first_target"] == 1 and rows[-1]["last_target"] == 16
    for i, row in enumerate(rows):
        assert abs(row["nll"] - sum(expected[i * 4 : (i + 1) * 4]) / 4) < 1e-7
