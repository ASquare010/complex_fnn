"""Meaningful checks for the shared regression loop and uneven held-out batches."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.core.function_fitting import fit_regression, regression_score

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression harness")


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def features(self, x):
        return x, self(x)

    def optimizer_groups(self, rate):
        return [{"params": list(self.parameters()), "lr": rate}]


def test_fit_and_uneven_score_against_whole_data_objective():
    torch.manual_seed(19)
    x = torch.randn(23, 3)
    y = torch.stack((x[:, 0] + x[:, 1], x[:, 2] - x[:, 1]), dim=1)
    stream = torch.arange(23).repeat(8, 1)
    model = LinearProbe().cuda()
    before = regression_score(model, x, y, batch_size=5)
    record = fit_regression(model, x, y, stream, rate=0.05, warmup=1)
    after = regression_score(model, x, y, batch_size=5)
    with torch.no_grad():
        direct = F.mse_loss(model(x.cuda()), y.cuda()).item()
    assert after < before
    assert abs(after - direct) < 1e-6
    assert len(record["history"]) == len(stream)
    assert record["all_finite"] and record["peak_allocated_bytes"] > 0
    assert record["optimizer_bytes"] > record["parameter_bytes"]
    assert all(row["preclip_norm"] > 0 for row in record["history"])


def test_resident_cuda_data_is_rejected_before_fitting():
    with pytest.raises(ValueError, match="CPU"):
        fit_regression(
            LinearProbe().cuda(),
            torch.ones(8, 3, device="cuda"),
            torch.ones(8, 2),
            torch.arange(8).reshape(1, -1),
            0.01,
        )
