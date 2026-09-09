"""Validation must use the supplied execution path, including compiled callables."""

import pytest
import torch
from torch.nn import functional as F

from src.core.benchmark import evaluate_forward


def test_callable_forward_is_used_and_targets_weight_the_average():
    class ForwardOnly:
        calls = 0

        def __call__(self, x):
            self.calls += 1
            return torch.stack((x.float(), -x.float()), -1)

        def loss(self, *args):
            raise AssertionError("Original loss would bypass the supplied forward")

    fn = ForwardOnly()
    batches = [
        (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[0, 1, 0]])),
        (torch.tensor([[4.0]]), torch.tensor([[1]])),
    ]
    loss, count = evaluate_forward(fn, batches, "cpu", "fp32")
    logits = torch.cat(
        [torch.stack((x.float(), -x.float()), -1).reshape(-1, 2) for x, y in batches]
    )
    labels = torch.cat([y.flatten() for x, y in batches])
    assert fn.calls == 2 and count == 4
    assert loss == pytest.approx(F.cross_entropy(logits, labels).item())


def test_empty_forward_evaluation_is_rejected():
    with pytest.raises(ValueError, match="No evaluation targets"):
        evaluate_forward(lambda x: x, [], "cpu", "fp32")
