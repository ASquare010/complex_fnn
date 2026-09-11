"""Greedy regularized orthogonal selection by exact Schur-complement gain."""

import torch

RIDGE = 1e-4
METHODS = {
    "random": ("random", False),
    "weight_norm": ("weight_norm", False),
    "fluctuation": ("fluctuation", False),
    "variance_pivot": ("variance_pivot", False),
    "value_greedy": ("value_greedy", False),
    "sobolev_greedy": ("sobolev_greedy", True),
    "value_select_sobolev_fit": ("value_greedy", True),
    "sobolev_select_value_fit": ("sobolev_greedy", False),
}
WIDTHS = {
    "gelu": {"primary": 448, "half": 768, "three_quarter": 1152},
    "swiglu": {"primary": 298, "half": 512, "three_quarter": 768},
}


def system(state, sobolev: bool):
    beta = state["beta"] if sobolev else 0
    g = state["value_gram"] + beta * state["derivative_gram"]
    g = g + RIDGE * torch.eye(len(g), dtype=g.dtype)
    c = state["value_cross"] + beta * state["derivative_cross"]
    return g, c


def greedy(g: torch.Tensor, c: torch.Tensor, keep: int, variance: bool = False):
    """Returns an ordered subset and every exact one-step objective decrease."""
    g, c = g.clone(), c.clone()
    chosen = torch.zeros(len(g), dtype=torch.bool)
    order, gains = [], []
    minimum_pivot = float("inf")
    for _ in range(keep):
        diagonal = g.diag().clone()
        assert bool((diagonal[~chosen] > 0).all()), "Loss of positive definiteness"
        gain = c.square().sum(1) / diagonal.clamp_min(1e-30)
        score = diagonal.clone() if variance else gain.clone()
        score[chosen] = -torch.inf
        selected = int(score.argmax())
        pivot = diagonal[selected].item()
        minimum_pivot = min(minimum_pivot, pivot)
        order.append(selected)
        gains.append(gain[selected].item())
        q = g[:, selected].clone() / pivot**0.5
        target = c[selected].clone() / pivot**0.5
        g.addr_(q, q, alpha=-1)
        c.addr_(q, target, alpha=-1)
        chosen[selected] = True
    assert len(set(order)) == keep
    return torch.tensor(order), {"gains": gains, "minimum_pivot": minimum_pivot}


def all_orders(state, model, seed: int, keep: int):
    value_g, value_c = system(state, False)
    sobolev_g, sobolev_c = system(state, True)
    norm = model.down.weight.double().norm(dim=0)
    orders = {
        "random": torch.randperm(len(norm), generator=torch.Generator().manual_seed(28000 + seed))[
            :keep
        ],
        "weight_norm": torch.argsort(norm, descending=True, stable=True)[:keep],
        "fluctuation": torch.argsort(norm * state["scale_h"], descending=True, stable=True)[:keep],
    }
    traces = {}
    for name, g, c, variance in (
        ("variance_pivot", value_g, value_c, True),
        ("value_greedy", value_g, value_c, False),
        ("sobolev_greedy", sobolev_g, sobolev_c, False),
    ):
        orders[name], traces[name] = greedy(g, c, keep, variance)
    return orders, traces


def solve(state, selected, sobolev: bool):
    g, c = system(state, sobolev)
    small = g[selected][:, selected]
    target = c[selected]
    cholesky = torch.linalg.cholesky(small)
    coefficient = torch.cholesky_solve(target, cholesky)
    residual = (small @ coefficient - target).abs().max().item()
    assert residual < 1e-8
    return coefficient, {
        "stationarity_max": residual,
        "explained_regularized_energy": (coefficient * target).sum().item(),
    }
