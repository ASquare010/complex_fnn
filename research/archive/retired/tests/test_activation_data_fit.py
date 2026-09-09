"""Independent least-squares fit and correction-energy checks."""

import torch

from src.core.activation_data_fit import fit_affine


def test_affine_fit_recovers_group_slopes_and_exposes_curvature():
    z = torch.linspace(-3, 3, 101, dtype=torch.float64).repeat(2, 1)
    correction = torch.stack((0.3 * z[0] + 0.1, 0.2 * z[1] ** 2))
    result = fit_affine(z, correction)
    torch.testing.assert_close(torch.tensor(result["gain"]), torch.tensor([0.3, 0.0]))
    assert abs(result["bias"][0] - 0.1) < 1e-12
    assert abs(result["bias"][1] - correction[1].mean().item()) < 1e-12
    assert result["squared_error_by_group"][0] < 1e-25
    assert result["squared_error_by_group"][1] > 1
