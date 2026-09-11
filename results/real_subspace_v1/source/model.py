"""Exact folding of input/output projectors; all estimator statistics are train-only."""

import torch
from torch import nn
from torch.nn import functional as F

METHODS = (
    "pca",
    "weight_input",
    "linear_aware",
    "energy",
    "white_energy",
    "white_residual",
    "white_permuted",
)
RANKS = (32, 64, 128)


def eig(matrix):
    return torch.linalg.eigh((matrix + matrix.T) / 2)


def bounded_energy(y):
    energy = y.square().sum(1)
    energy = energy / energy.mean().clamp_min(torch.finfo(y.dtype).tiny)
    w = energy / (1 + energy)
    return w - w.mean()


def estimate(x, y, teacher, seed):
    """All tensors CPU double; save the matrices so a different reader can audit."""
    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean(0)
    x, y = x - mx, y - my
    n, d = x.shape
    covariance = x.T @ x / n
    values, vectors = eig(covariance)
    floored = values.clamp_min(values[-1] * 1e-6)
    h = (vectors * floored.sqrt()) @ vectors.T
    t = (vectors * floored.rsqrt()) @ vectors.T
    cross = x.T @ y / n
    ridge = covariance.trace() / d * 1e-6
    affine = torch.linalg.solve(covariance + ridge * torch.eye(d, dtype=x.dtype), cross)
    residual = y - x @ affine
    energy, residual_energy = bounded_energy(y), bounded_energy(residual)
    permuted = energy[torch.randperm(n, generator=torch.Generator().manual_seed(15000 + seed))]
    l = teacher.up.weight.detach().cpu().double()
    if teacher.gate is not None:
        l = torch.cat((l, teacher.gate.weight.detach().cpu().double()))
    weighted = x.T @ (x * energy[:, None]) / n
    matrices = {
        "pca": covariance,
        "weight_input": l.T @ l,
        "linear_aware": h @ l.T @ l @ h,
        "energy": weighted,
        "white_energy": t @ weighted @ t,
        "white_residual": t @ (x.T @ (x * residual_energy[:, None]) / n) @ t,
        "white_permuted": t @ (x.T @ (x * permuted[:, None]) / n) @ t,
    }
    bases, spectra = {}, {}
    for method, matrix in matrices.items():
        spectra[method], bases[method] = eig(matrix)
    output_spectrum, output_basis = eig(y.T @ y / n)
    return {
        "mx": mx,
        "my": my,
        "h": h,
        "t": t,
        "covariance": covariance,
        "eigenvalues": values,
        "floored": int((values != floored).sum()),
        "affine": affine,
        "ridge": ridge,
        "cross": cross,
        "matrices": matrices,
        "bases": bases,
        "spectra": spectra,
        "output_basis": output_basis,
        "output_spectrum": output_spectrum,
    }


def maps(record, method, rank):
    b = record["bases"][method][:, -rank:]
    c = record["output_basis"][:, -rank:]
    if method in ("linear_aware", "white_energy", "white_residual", "white_permuted"):
        return record["t"] @ b, b.T @ record["h"], c
    return b, b.T, c


class ProjectedFFN(nn.Module):
    """No dense projectors retained; parameter count includes all folded biases."""

    def __init__(self, teacher, a, e, c, mx, my):
        super().__init__()
        d, r = a.shape
        h = teacher.up.out_features
        dtype, device = a.dtype, a.device
        self.activation = teacher.activation
        self.input = nn.Linear(d, r, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(r, h, bias=True, dtype=dtype, device=device)
        self.gate = (
            nn.Linear(r, h, bias=True, dtype=dtype, device=device)
            if teacher.gate is not None
            else None
        )
        self.down = nn.Linear(h, r, bias=False, dtype=dtype, device=device)
        self.output = nn.Linear(r, d, bias=True, dtype=dtype, device=device)
        with torch.no_grad():
            offset = mx - (mx @ a) @ e
            self.input.weight.copy_(a.T)
            self.up.weight.copy_(teacher.up.weight @ e.T)
            self.up.bias.copy_(teacher.up.weight @ offset)
            if self.gate is not None:
                self.gate.weight.copy_(teacher.gate.weight @ e.T)
                self.gate.bias.copy_(teacher.gate.weight @ offset)
            self.down.weight.copy_(c.T @ teacher.down.weight)
            self.output.weight.copy_(c)
            self.output.bias.copy_(my - (my @ c) @ c.T)

    def forward(self, x):
        z = self.input(x)
        u = self.up(z)
        features = F.silu(u) * self.gate(z) if self.gate is not None else F.gelu(u)
        return self.output(self.down(features))
