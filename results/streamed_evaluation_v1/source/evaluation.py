"""Evaluation policies with a fixed attention context and valid-target weighting."""

import hashlib

import torch
from torch.nn import functional as F

from src.core.benchmark import autocast
from src.core.token_memory import chunked_linear_cross_entropy

POLICIES = ("native", "classifier_chunks", "sequence_chunks")
CHUNK = 512


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@torch.no_grad()
def batch_loss(model, x: torch.Tensor, y: torch.Tensor, policy: str, chunk: int = CHUNK):
    if policy == "native":
        logits = model(x)
        logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        return F.cross_entropy(logits.flatten(0, 1), y.flatten())
    if policy == "classifier_chunks":
        hidden = model.embedding(x)
        for block in model.blocks:
            hidden = block(hidden)
        return chunked_linear_cross_entropy(model.norm(hidden), model.embedding.weight, y, chunk)
    if policy != "sequence_chunks":
        raise ValueError(f"Unknown evaluation policy: {policy}")
    per_batch = max(1, chunk // x.shape[1])
    total, count = None, 0
    for part, labels in zip(x.split(per_batch), y.split(per_batch), strict=True):
        valid = int((labels != -100).sum().item())
        if not valid:
            continue
        term = batch_loss(model, part, labels, "native", chunk) * valid
        total = term if total is None else total + term
        count += valid
    return total / count if count else model.embedding.weight.new_tensor(float("nan"))


@torch.no_grad()
def evaluate(model, data, batch: int, context: int, batches: int, policy: str):
    was_training = model.training
    model.eval()
    total, targets = 0.0, 0
    order = hashlib.sha256()
    try:
        for x, y in data.validation(batch, context, batches):
            with autocast("cuda", "bf16"):
                value = batch_loss(model, x, y, policy)
            count = int((y != -100).sum().item())
            total += value.item() * count
            targets += count
            order.update(tensor_hash(x).encode())
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("No validation targets")
    return {"nll": total / targets, "targets": targets, "order_sha256": order.hexdigest()}
