"""Gradient comparison shared by worker and independent CPU verifier."""

import hashlib

import torch


def digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def compare(actual, reference):
    assert actual.keys() == reference.keys()
    difference = reference_norm = 0.0
    layers = {}
    for name, base in reference.items():
        value = actual[name]
        delta = value.double() - base.double()
        ds = delta.square().sum().item()
        rs = base.double().square().sum().item()
        difference += ds
        reference_norm += rs
        layers[name] = dict(
            exact=digest(value) == digest(base),
            relative_l2=(ds / max(rs, 1e-60)) ** 0.5,
            max_abs=delta.abs().max().item(),
        )
    return dict(
        exact=all(x["exact"] for x in layers.values()),
        exact_tensors=sum(x["exact"] for x in layers.values()),
        relative_l2=(difference / max(reference_norm, 1e-60)) ** 0.5,
        layers=layers,
    )
