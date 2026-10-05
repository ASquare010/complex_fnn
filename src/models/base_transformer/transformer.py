"""Dense full-attention Transformer baseline."""

import torch
from torch import nn
from torch.nn import functional as F

from models.components import Attention, DenseFFN, RMSNorm, counts, initialize
from settings import ModelConfig


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = DenseFFN(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    """Token IDs [batch, tokens] -> logits [batch, tokens, vocabulary]."""

    model_name = "base_transformer"

    def __init__(self, config: ModelConfig, seed: int = 17):
        super().__init__()
        config.validate()
        if config.name != self.model_name:
            raise ValueError(f"{self.model_name} received settings for {config.name}")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width)
        initialize(self, seed)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("Expected [batch, length] within the configured context")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    def counts(self):
        return counts(self)
