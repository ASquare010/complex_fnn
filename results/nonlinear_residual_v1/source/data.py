"""Population-affine-free Gaussian tasks with fixed low-dimensional latent structure."""

import torch

from results.nonlinear_residual_v1.source.model import TASKS, target_features

TRAIN, SELECT_END, TOTAL = 65536, 69632, 73728


def make_data():
    x = torch.randn(TOTAL, 384, generator=torch.Generator().manual_seed(9860))
    indices = torch.randperm(384, generator=torch.Generator().manual_seed(9861))[:16]
    rotation = torch.linalg.qr(
        torch.randn(384, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9862))
    ).Q[:16]
    q = x[:, indices].double()
    targets, scales = {}, {}
    for task in TASKS:
        latent = target_features(q, task)
        raw = latent @ rotation
        scale = raw[:TRAIN].std(dim=0, correction=0)
        assert (scale > 0).all()
        targets[task] = (raw / scale).float()
        scales[task] = scale
        assert torch.isfinite(targets[task]).all()
    return {
        "x": x,
        "indices": indices,
        "rotation": rotation,
        "targets": targets,
        "scales": scales,
        "streams": {
            seed: torch.randint(
                TRAIN, (300, 256), generator=torch.Generator().manual_seed(20000 + seed)
            )
            for seed in (17, 29, 43)
        },
    }
