"""Training-only label-energy moments; no access to a teacher basis or task name."""

import torch


def weights_from_labels(y: torch.Tensor, mode: str) -> torch.Tensor:
    if y.ndim != 2 or y.shape[0] < 2 or not torch.isfinite(y).all():
        raise ValueError("Expected finite [samples, outputs] labels")
    energy = y.double().square().sum(-1)
    mean = energy.mean()
    if mean <= 0:
        raise ValueError("Labels must have positive mean energy")
    energy = energy / mean
    if mode == "raw":
        weights = energy
    elif mode == "bounded":
        weights = energy / (1 + energy)
    else:
        raise ValueError(mode)
    return weights - weights.mean()


def centered_moment(
    x: torch.Tensor, weights: torch.Tensor, chunk_size: int = 2048, device: str = "cuda"
) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != weights.numel() or x.shape[0] < 2:
        raise ValueError("Expected paired input rows and weights")
    if chunk_size <= 0 or not torch.isfinite(x).all() or not torch.isfinite(weights).all():
        raise ValueError("Invalid chunk size or nonfinite inputs")
    result = torch.zeros(x.shape[1], x.shape[1], device=device, dtype=torch.float64)
    centered = weights.double() - weights.double().mean()
    for start in range(0, len(x), chunk_size):
        part = x[start : start + chunk_size].to(device=device, dtype=torch.float64)
        w = centered[start : start + chunk_size].to(device)
        result.add_(part.T @ (w[:, None] * part))
    result /= len(x)
    return ((result + result.T) / 2).cpu()


def leading_subspace(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not 0 < rank < matrix.shape[0]:
        raise ValueError("Expected square matrix and a valid rank")
    if not torch.isfinite(matrix).all():
        raise ValueError("Nonfinite matrix")
    eigenvalues, vectors = torch.linalg.eigh(matrix.double().cpu())
    return vectors[:, -rank:], eigenvalues
