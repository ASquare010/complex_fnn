"""Small mathematical witnesses before touching teacher calibration data."""

import torch
from torch.nn import functional as F

from results.real_subspace_v1.source.model import ProjectedFFN, estimate, maps
from src.core.config import ModelConfig
from src.core.ffn_capture import capture_ffn_pairs
from src.core.transformer import Transformer
from src.dense_ffn import DenseFFN


def run():
    torch.manual_seed(105)
    checks = []
    for activation in ("gelu", "swiglu"):
        teacher = DenseFFN(12, 19, activation).double()
        x = torch.randn(71, 12, dtype=torch.float64) + 0.7
        with torch.no_grad():
            y = teacher(x)
        state = estimate(x, y, teacher, 17)
        torch.testing.assert_close(
            state["h"] @ state["t"], torch.eye(12).double(), atol=1e-10, rtol=1e-10
        )
        for method in ("pca", "linear_aware", "white_energy", "white_residual"):
            for rank in (4, 12):
                a, e, c = maps(state, method, rank)
                model = ProjectedFFN(teacher, a, e, c, state["mx"], state["my"])
                probe = torch.randn(9, 12, dtype=torch.float64, requires_grad=True)
                expected = (
                    state["my"]
                    + (teacher(state["mx"] + (probe - state["mx"]) @ a @ e) - state["my"]) @ c @ c.T
                )
                actual = model(probe)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                actual_gradient = torch.autograd.grad(actual.square().sum(), probe)[0]
                expected_gradient = torch.autograd.grad(expected.square().sum(), probe)[0]
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10
                )
                if rank == 12:
                    torch.testing.assert_close(actual, teacher(probe), atol=1e-10, rtol=1e-10)
                count = (
                    2 * 12 * rank
                    + (3 if activation == "swiglu" else 2) * 19 * rank
                    + (2 if activation == "swiglu" else 1) * 19
                    + 12
                )
                assert sum(p.numel() for p in model.parameters()) == count
                checks.append(
                    {
                        "activation": activation,
                        "method": method,
                        "rank": rank,
                        "max_error": (actual - expected).abs().max().item(),
                        "parameters": count,
                    }
                )
        # The top response eigenspace beats random projectors on a finite whitened
        # linear objective; singular-value tail gives the independently known optimum.
        l = teacher.up.weight.detach()
        if teacher.gate is not None:
            l = torch.cat((l, teacher.gate.weight.detach()))
        z = (x - state["mx"]) @ state["t"]
        b = state["bases"]["linear_aware"][:, -4:]
        response = state["h"] @ l.T
        loss = ((z - z @ b @ b.T) @ response).square().sum() / len(x)
        optimum = torch.linalg.svdvals(response)[4:].square().sum()
        torch.testing.assert_close(loss, optimum, atol=1e-10, rtol=1e-10)
        checks.append(
            {"activation": activation, "linear_optimum_error": abs((loss - optimum).item())}
        )
    model = Transformer(
        ModelConfig(width=24, hidden=36, layers=2, heads=3, vocab_size=41, context=8)
    ).cuda()
    windows = torch.randint(41, (3, 8))
    model.eval()
    expected = []
    hook = model.blocks[1].ffn.register_forward_hook(
        lambda m, args, out: expected.append((args[0].clone(), out.clone()))
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(windows.cuda())
    hook.remove()
    model.train()
    actual = capture_ffn_pairs(model, windows, [1], batch_size=3)
    assert model.training and not model.blocks[1].ffn._forward_hooks
    for key, value in zip(("x", "y"), expected[0], strict=True):
        torch.testing.assert_close(
            actual[1][key], value.flatten(0, 1).float().cpu(), atol=0, rtol=0
        )
    checks.append({"capture_matches_full_forward": True, "restores_mode_and_hooks": True})
    return checks
