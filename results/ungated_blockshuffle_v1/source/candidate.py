"""H072 control constructor: use the existing ungated BlockShuffle implementation."""

from src.blockshuffle_ffn import BlockShuffleFFN


def make_model(hidden, seed):
    if hidden not in (2048, 3264):
        raise ValueError("H072 freezes hidden width 2048 or 3264")
    model = BlockShuffleFFN(384, hidden, 8, gated=False)
    model.up.initialize(seed, "blocks.0.ffn.up")
    model.down.initialize(seed, "blocks.0.ffn.down", 0.25)
    return model
