"""Optional CUDA checks of explicit permutations, rounding and inference guards."""

import pytest
import torch

pytest.importorskip("triton", reason="Install the optional compile extra for Triton checks")

from src.blockshuffle_ffn import BlockShuffleFFN
from src.blockshuffle_ffn.triton_inference import TritonInferenceFFN, matrix_work


@pytest.fixture(autouse=True)
def local_compiler_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "width,hidden,groups,tokens", [(32, 48, 4, 5), (192, 1024, 8, 33), (384, 2048, 8, 65)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_matches_native_on_fresh_non_tile_aligned_inputs(
    width, hidden, groups, tokens, dtype
):
    torch.manual_seed(42)
    base = BlockShuffleFFN(width, hidden, groups).cuda().eval()
    fused = TritonInferenceFFN(base).eval()
    assert [id(p) for p in fused.parameters()] == [id(p) for p in base.parameters()]
    for scale in (0.3, 1.0, 2.0):
        x = torch.randn(tokens, width, device="cuda", dtype=dtype) * scale
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = base(x).float()
            actual = fused(x).float()
        error = actual - expected
        assert torch.isfinite(actual).all()
        assert error.square().mean().sqrt() <= 0.005 * expected.square().mean().sqrt()
        assert error.abs().max() <= 0.05
        assert actual.shape == expected.shape


def test_inference_guard_and_padded_matrix_accounting():
    base = BlockShuffleFFN(32, 48, 4)
    fused = TritonInferenceFFN(base)
    with pytest.raises(RuntimeError, match="inference-only"):
        fused(torch.ones(2, 32))
    fused.eval()
    with pytest.raises(RuntimeError, match="inference-only"):
        fused(torch.ones(2, 32))
    assert matrix_work(384, 2048, 8)["logical_matrix_flops_per_token_per_layer"] == 700416
    assert matrix_work(384, 2048, 8)["padded_scalar_matrix_flops_per_token_per_layer"] == 983040
