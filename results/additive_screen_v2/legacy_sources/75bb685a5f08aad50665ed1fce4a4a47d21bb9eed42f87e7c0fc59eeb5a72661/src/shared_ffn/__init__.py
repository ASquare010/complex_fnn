"""Placement of shared dense FFNs; computation remains the reference FFN."""

from torch import nn

from src.dense_ffn import DenseFFN

SHARING_SPAN = 4


def shared_ffn_for_layer(
    bank: dict[int, nn.Module], index: int, width: int, hidden: int, activation: str
) -> nn.Module:
    """Return one FFN object per consecutive group of four existing layers."""
    key = index // SHARING_SPAN
    if key not in bank:
        bank[key] = DenseFFN(width, hidden, activation)
    return bank[key]
