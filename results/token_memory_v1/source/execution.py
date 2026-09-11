"""Isolated full-model adapter for H101; parameter identities/names stay intact."""

from types import MethodType

import torch
from torch.nn import functional as F

from src.core.token_memory import chunked_ffn, chunked_linear_cross_entropy

POLICIES = ("native", "ffn", "block", "ffn_chunks", "loss_chunks", "both_chunks")


def install(model, policy: str, chunk_size: int = 512) -> None:
    if policy not in POLICIES:
        raise ValueError(policy)
    scope = policy if policy in ("ffn", "block") else "none"
    if policy.endswith("chunks"):
        scope = "block"
    model.set_recompute_scope(scope)
    if policy in ("ffn_chunks", "both_chunks"):
        for block in model.blocks:
            original = block.ffn.forward

            def forward(self, x, original=original):
                if self.training and torch.is_grad_enabled():
                    return chunked_ffn(original, x, chunk_size)
                return original(x)

            block.ffn.forward = MethodType(forward, block.ffn)
    if policy in ("loss_chunks", "both_chunks"):

        def loss(self, tokens, targets):
            if tokens.ndim != 2 or tokens.shape[1] > self.config.context:
                raise ValueError("Expected [batch, length] within configured context")
            x = self.embedding(tokens)
            for block in self.blocks:
                x = block(x)
            return chunked_linear_cross_entropy(
                self.norm(x), self.embedding.weight, targets, chunk_size
            )

        model.loss = MethodType(loss, model)


def common_loss(model, tokens, targets):
    """Unchunked scoring for every policy, bypassing the experimental loss adapter."""
    return F.cross_entropy(model(tokens).float().flatten(0, 1), targets.flatten())
