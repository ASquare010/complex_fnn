"""Six FP64 GPU backwards and twelve directional-difference loss forwards."""

# Runtime preloading in common must precede other PyTorch imports.
# ruff: noqa: I001

from results.checkpoint_input_offload_v1.source.common import (
    ROOT,
    boundary,
    check_inputs,
    environment,
    read,
    torch,
    write_json,
)

from contextlib import nullcontext
import traceback

from torch.utils.checkpoint import checkpoint


def value(parameters, mode):
    hidden, *weights = parameters
    for weight in weights:

        def block(x, w=weight):
            return torch.tanh(x @ w)

        if mode == "plain":
            hidden = block(hidden)
        else:
            with (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if mode == "offload"
                else nullcontext()
            ):
                hidden = checkpoint(block, hidden, use_reentrant=False, preserve_rng_state=True)
    return hidden.square().mean() + 0.17 * hidden.mean()


def shape_check(batch, width):
    generator = torch.Generator().manual_seed(121 + width)
    base = [
        torch.randn(shape, generator=generator, dtype=torch.float64).cuda() * 0.2
        for shape in ((batch, width), (width, width + 1), (width + 1, width))
    ]
    rows, reference = [], None
    for mode in ("plain", "native", "offload"):
        parameters = [v.detach().clone().requires_grad_() for v in base]
        loss = value(parameters, mode)
        loss.backward()
        gradients = [v.grad.detach().clone() for v in parameters]
        if reference is None:
            reference = (loss.detach().clone(), gradients)
        else:
            torch.testing.assert_close(loss, reference[0], atol=1e-12, rtol=1e-12)
            for a, b in zip(gradients, reference[1], strict=True):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        rows.append(dict(mode=mode, loss=loss.item(), passed=True))
    differences = []
    for _ in range(3):
        direction = [
            torch.randn(v.shape, generator=generator, dtype=torch.float64).cuda() for v in base
        ]
        scale = sum(d.square().sum() for d in direction).sqrt()
        direction = [d / scale for d in direction]
        analytical = sum((g * d).sum() for g, d in zip(reference[1], direction, strict=True)).item()
        with torch.no_grad():
            plus = value(
                [v + 1e-5 * d for v, d in zip(base, direction, strict=True)], "plain"
            ).item()
            minus = value(
                [v - 1e-5 * d for v, d in zip(base, direction, strict=True)], "plain"
            ).item()
        numerical = (plus - minus) / 2e-5
        error = abs(numerical - analytical)
        assert error <= 1e-7 * max(1, abs(analytical))
        differences.append(
            dict(analytical=analytical, numerical=numerical, absolute_error=error, passed=True)
        )
    return dict(shape=[batch, width], comparisons=rows, directions=differences, passed=True)


def run():
    assert not (ROOT / "qualification.json").exists()
    check_inputs(read(ROOT / "protocol.json"))
    write_json(ROOT / "environment.json", environment())
    boundaries, rows = [boundary()], []
    for shape in ((2, 4), (3, 7)):
        rows.append(shape_check(*shape))
        boundaries.append(boundary())
        write_json(ROOT / "qualification_progress.json", dict(rows=rows, boundaries=boundaries))
    write_json(
        ROOT / "qualification.json",
        dict(
            passed=True,
            rows=rows,
            boundaries=boundaries,
            backwards=6,
            finite_difference_forwards=12,
            optimizer_updates=0,
        ),
    )
    print("FP64 offload/checkpoint gradients and directional derivatives: PASS", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "qualification_failure.json", dict(traceback=traceback.format_exc()))
        raise
