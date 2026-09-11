"""Scope the proven native CPU-save adapter without changing module ownership."""

from types import SimpleNamespace

from results.checkpoint_input_offload_v1.source.offload import install


def install_subset(model, count):
    assert len(model.blocks) == 8 and count in (0, 4, 8)
    selected = list(range(8 - count, 8))
    view = SimpleNamespace(
        blocks=[model.blocks[i] for i in selected], named_parameters=model.named_parameters
    )
    restore = install(view, True)
    assert [i for i, b in enumerate(model.blocks) if "forward" in b.__dict__] == selected
    return restore, selected
