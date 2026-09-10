"""H100 fresh hidden directions; no candidate is given the input basis."""

import torch

from results.triadic_interaction_v1.source.model import TASKS, input_basis, target_features

TRAIN, SELECT_END, TOTAL = 65536, 69632, 73728


def make_data():
    x = torch.randn(TOTAL, 384, generator=torch.Generator().manual_seed(9900))
    basis = input_basis()
    rotation = torch.linalg.qr(
        torch.randn(384, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(9902))
    ).Q[:16]
    q = x.double() @ basis
    targets, scales = {}, {}
    for task in TASKS:
        raw = target_features(q, task) @ rotation
        scale = raw[:TRAIN].std(0, correction=0)
        assert (scale > 0).all()
        scales[task], targets[task] = scale, (raw / scale).float()
        assert torch.isfinite(targets[task]).all()
    return {
        "x": x,
        "basis": basis,
        "rotation": rotation,
        "targets": targets,
        "scales": scales,
        "streams": {
            s: torch.randint(TRAIN, (300, 256), generator=torch.Generator().manual_seed(30000 + s))
            for s in (17, 29, 43)
        },
    }
