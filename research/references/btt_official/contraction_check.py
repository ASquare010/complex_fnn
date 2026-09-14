"""CPU float64 BTT contraction checks, derived directly from the index equation."""

import hashlib
import json
from pathlib import Path

import numpy as np


def forward(x, right, left):
    middle = np.einsum("ngi,girb->ngrb", x, right)
    return np.einsum("ngrb,brga->nba", middle, left)


rng = np.random.default_rng(164)
checks = []
for m0, m1, n0, n1, rank in ((2, 3, 4, 2, 1), (3, 2, 2, 4, 2), (3, 4, 2, 4, 1)):
    x = rng.normal(size=(5, m1, m0))
    right = rng.normal(size=(m1, m0, rank, n0))
    left = rng.normal(size=(n0, rank, m1, n1))
    upstream = rng.normal(size=(5, n0, n1))
    y = forward(x, right, left)
    matrix = np.einsum("girb,brga->giba", right, left).reshape(m1 * m0, n0 * n1)
    error = np.max(np.abs(y.reshape(5, -1) - x.reshape(5, -1) @ matrix))
    assert error < 1e-12
    dx = np.einsum("nba,girb,brga->ngi", upstream, right, left)
    dr = np.einsum("nba,ngi,brga->girb", upstream, x, left)
    dl = np.einsum("nba,ngi,girb->brga", upstream, x, right)
    derivatives = []
    for k, gradient in enumerate((dx, dr, dl)):
        values = [x, right, left]
        direction = rng.normal(size=values[k].shape)
        delta = 1e-5
        plus, minus = values.copy(), values.copy()
        plus[k] = values[k] + delta * direction
        minus[k] = values[k] - delta * direction
        numeric = np.sum((forward(*plus) - forward(*minus)) * upstream) / (2 * delta)
        analytic = np.sum(direction * gradient)
        relative = abs(numeric - analytic) / max(1, abs(numeric), abs(analytic))
        assert relative < 1e-7
        derivatives.append(float(relative))
    checks.append(
        dict(shape=[m0, m1, n0, n1, rank], dense_error=float(error), vjp_errors=derivatives)
    )

witnesses = []
for m0, m1, n0, n1 in ((16, 32, 19, 32), (19, 32, 16, 32)):
    # Selector cores yield a partial permutation; uniqueness proves its rank.
    edges = [(g * m0 + i, i * n1 + g) for g in range(m1) for i in range(min(m0, n0))]
    assert len(edges) == 512
    assert len({i for i, j in edges}) == len({j for i, j in edges}) == 512
    witnesses.append(dict(input=m0 * m1, output=n0 * n1, rank=512, nonzero_edges=len(edges)))
result = dict(
    passed=True,
    checks=checks,
    balanced_rank_witnesses=witnesses,
    training_updates=0,
    gpu_work=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
Path("research/references/btt_official/contraction_check.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
