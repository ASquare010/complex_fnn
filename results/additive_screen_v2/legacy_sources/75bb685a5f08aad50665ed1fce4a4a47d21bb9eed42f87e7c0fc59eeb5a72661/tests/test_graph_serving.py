"""CUDA graph replay must consume fresh inputs and reject a changed shape."""

import pytest
import torch

from src.core.config import ModelConfig
from src.core.graph_serving import GraphForward
from src.core.transformer import Transformer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test")
@pytest.mark.parametrize("variant", ["swiglu", "blockshuffle_swiglu"])
def test_graph_consumes_fresh_inputs_and_matches_eager(variant):
    torch.set_num_threads(4)
    model = (
        Transformer(
            ModelConfig(
                variant=variant, width=24, heads=3, layers=2, vocab_size=32, context=8, groups=4
            )
        )
        .cuda()
        .eval()
    )
    x = torch.zeros((2, 8), dtype=torch.long, device="cuda")
    graph = GraphForward(model, x)
    with torch.no_grad():
        for offset in (1, 7):
            fresh = x + offset
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = model(fresh)
            torch.testing.assert_close(graph(fresh), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="captured shape"):
        graph(x[:, :4])
    model.train()
    with pytest.raises(RuntimeError, match="inference only"):
        graph(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test")
def test_shared_warmup_stream_does_not_accumulate_workspaces():
    import gc

    torch.set_num_threads(4)
    stream = torch.cuda.Stream()
    x = torch.zeros((2, 8), dtype=torch.long, device="cuda")
    retained = []
    for _ in range(3):
        model = (
            Transformer(
                ModelConfig(variant="swiglu", width=24, heads=3, layers=2, vocab_size=32, context=8)
            )
            .cuda()
            .eval()
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        graph = GraphForward(model, x, warmup_stream=stream)
        graph(x)
        torch.cuda.synchronize()
        del graph, model
        gc.collect()
        torch.cuda.empty_cache()
        retained.append(torch.cuda.memory_allocated())
    assert retained[1:] == [retained[0], retained[0]]
