"""Qualify the analytical VJP, then reuse H123's unchanged probe loop."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    boundary,
    check_inputs,
    construct,
    hashes,
    read,
    tensor_hash,
    write_json,
)
from results.gradient_staging_v1 import study as previous
from results.compact_rmsnorm_v1.norm import CompactRMS, install
from pathlib import Path
from unittest.mock import patch

ROOT = Path("results/compact_rmsnorm_v1")


def reference(x, w):
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)) * w


def qualify():
    generator = torch.Generator().manual_seed(124)
    records = []
    for shape, scale in (((2, 5), 0.0), ((2, 3, 8), 0.1), ((3, 16), 10.0)):
        base = torch.randn(shape, generator=generator, dtype=torch.float64).cuda() * scale
        weight = (1 + torch.randn(shape[-1], generator=generator, dtype=torch.float64) * 0.1).cuda()
        g = torch.randn(shape, generator=generator, dtype=torch.float64).cuda()
        expected = None
        for fn in (reference, CompactRMS.apply):
            x, w = base.clone().requires_grad_(), weight.clone().requires_grad_()
            before = [tensor_hash(v) for v in (x, w, g)]
            y = fn(x, w)
            loss = (y * g).sum()
            loss.backward()
            actual = (y.detach().clone(), x.grad.detach().clone(), w.grad.detach().clone())
            if expected is None:
                expected = actual
            else:
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
            assert before == [tensor_hash(v) for v in (x, w, g)]
        errors = []
        for _ in range(2):
            dx = torch.randn(shape, generator=generator, dtype=torch.float64).cuda()
            dw = torch.randn(shape[-1], generator=generator, dtype=torch.float64).cuda()
            length = (dx.square().sum() + dw.square().sum()).sqrt()
            dx, dw = dx / length, dw / length
            analytical = (expected[1] * dx).sum().item() + (expected[2] * dw).sum().item()
            with torch.no_grad():
                plus = (reference(base + 1e-6 * dx, weight + 1e-6 * dw) * g).sum().item()
                minus = (reference(base - 1e-6 * dx, weight - 1e-6 * dw) * g).sum().item()
            error = abs((plus - minus) / 2e-6 - analytical)
            assert error <= 1e-7 * max(1, abs(analytical))
            errors.append(dict(absolute_error=error, analytical=analytical))
        records.append(dict(shape=shape, scale=scale, passed=True, directions=errors))
    return dict(passed=True, records=records, backwards=6, finite_difference_forwards=12)


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    prior, refs = [
        read("results/checkpoint_input_offload_v1/" + n) for n in ("protocol.json", "result.json")
    ]
    check_inputs(prior)
    boundaries = [boundary()]
    qualification = qualify()
    boundaries.append(boundary())
    write_json(ROOT / "qualification.json", qualification)
    cases = []
    arms = [
        ("ordinary", "resident"),
        ("compact", "resident"),
        ("ordinary", "staged"),
        ("compact", "staged"),
    ]
    for i, f in enumerate(p["fixtures"]):
        for kind, mode in arms if i == 0 else list(reversed(arms)):

            def constructor(fixture, protocol, normalization=kind):
                result = construct(fixture, protocol)
                if normalization == "compact":
                    assert install(result[0]) == 17
                return result

            with (
                patch.object(previous, "ROOT", ROOT / kind),
                patch.object(previous, "construct", constructor),
            ):
                measurement = previous.run_case(f, mode, p, prior, refs)
            cases.append(dict(normalization=kind, measurement=measurement))
            boundaries.append(boundary())
    write_json(
        ROOT / "result.json",
        dict(
            cases=cases,
            qualification=qualification,
            boundaries=boundaries,
            backwards=86,
            training_updates=0,
            diagnostic_targets=327680,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
