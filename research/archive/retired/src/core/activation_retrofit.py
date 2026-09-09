"""Attach a zero-correction activation to an existing trained compressed model."""

from dataclasses import replace

import torch

from src.core.transformer import Transformer


def add_learnable_activations(source: Transformer, family: str = "rational") -> Transformer:
    """Copy learned weights exactly; return a new model with zero activation thetas.

    The source and its optimizer are untouched. Build a fresh optimizer for the
    returned model: optimizer state is not transferred. Do not apply width
    initialization again, because the copied weights already contain that gain.
    """
    if source.config.variant not in ("swiglu_narrow", "blockshuffle_swiglu"):
        raise ValueError("Retrofit requires an unmodified compressed SwiGLU base")
    if family not in ("shifted", "rational", "affine"):
        raise ValueError("Choose shifted, rational or affine activation")
    suffix = "affine_activation" if family == "affine" else family
    config = replace(source.config, variant=f"{source.config.variant}_{suffix}")
    reference = source.embedding.weight
    result = Transformer(config).to(device=reference.device, dtype=reference.dtype)
    missing, unexpected = result.load_state_dict(source.state_dict(), strict=False)
    expected = {name for name, _ in result.named_parameters() if ".curve." in name}
    if set(missing) != expected or unexpected:
        raise ValueError("Base checkpoint does not match the expected architecture")
    source_parameters = dict(source.named_parameters())
    for name, parameter in result.named_parameters():
        if name in source_parameters:
            if not torch.equal(parameter, source_parameters[name]):
                raise ValueError(f"Copied parameter differs: {name}")
        elif torch.count_nonzero(parameter):
            raise ValueError("New activation correction must start at zero")
    for old, new in zip(source.blocks, result.blocks):
        if hasattr(old.ffn, "recompute_gate"):
            new.ffn.recompute_gate = old.ffn.recompute_gate
            new.ffn.gate_recompute_method = "checkpoint"
    result.train(source.training)
    return result


def remove_zero_learnable_activations(source: Transformer) -> Transformer:
    """Copy a model with every curve theta reset to zero into its plain FFN base.

    Refuse to discard a nonzero learned correction implicitly. The source and
    its checkpoint remain untouched; no optimizer state or initialization gain
    is applied to the returned deployment copy.
    """
    from src.learnable_activation_ffn import LearnableActivation

    if not all(
        isinstance(getattr(b.ffn, "curve", None), LearnableActivation) for b in source.blocks
    ):
        raise ValueError("Expected learnable residual activations in every FFN")
    if any(torch.count_nonzero(p) for b in source.blocks for p in b.ffn.curve.parameters()):
        raise ValueError("Reset every activation theta to zero before removal")
    base = (
        "blockshuffle_swiglu"
        if source.config.variant.startswith("blockshuffle_swiglu_")
        else "swiglu_narrow"
    )
    config = replace(source.config, variant=base)
    reference = source.embedding.weight
    result = Transformer(config).to(device=reference.device, dtype=reference.dtype)
    common = {n: t for n, t in source.state_dict().items() if ".curve." not in n}
    result.load_state_dict(common, strict=True)
    assert all(torch.equal(t, common[n]) for n, t in result.state_dict().items())
    for old, new in zip(source.blocks, result.blocks):
        if hasattr(old.ffn, "recompute_gate"):
            new.ffn.recompute_gate = old.ffn.recompute_gate
            new.ffn.gate_recompute_method = old.ffn.gate_recompute_method
    result.train(source.training)
    return result
