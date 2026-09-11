"""Qualify native buffer aliasing, then reuse the fixed H123 model probe."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    boundary,
    check_inputs,
    hashes,
    read,
    tensor_hash,
    write_json,
    environment,
)
from results.gradient_staging_v1 import study as previous
from results.native_buffer_loss_v1.operator import NativeBufferLoss, model_loss
from torch.nn import functional as F
from pathlib import Path
from unittest.mock import patch

ROOT = Path("results/native_buffer_loss_v1")


def native(h, w, y):
    return F.cross_entropy(F.linear(h, w), y)


def qualify():
    records = []
    generator = torch.Generator().manual_seed(127)
    shapes = ((3, 7, 5), (9, 1024, 32), (11, 4096, 48), (5, 4097, 16))
    for dtype in (torch.float32, torch.float64):
        for index, (n, v, d) in enumerate(shapes):
            h = torch.randn(d, n, dtype=dtype, generator=generator).t().cuda()
            w = torch.randn(v, d, dtype=dtype, generator=generator).cuda() * (
                10 if index == 3 else 0.1
            )
            y = (torch.arange(n, device="cuda") * 3) % v
            y[::3] = -100
            h.requires_grad_()
            w.requires_grad_()
            upstream = torch.tensor(-0.75, device="cuda", dtype=dtype)
            before = [tensor_hash(x) for x in (h, w, y, upstream)]
            results = []
            for fn in (native, NativeBufferLoss.apply):
                loss = fn(h, w, y)
                gradients = torch.autograd.grad(loss, (h, w), grad_outputs=upstream)
                results.append((loss.detach(), gradients))
            reference, actual = results
            equal = [torch.equal(actual[0], reference[0])] + [
                torch.equal(a, b) for a, b in zip(actual[1], reference[1], strict=True)
            ]
            finite = all(bool(torch.isfinite(x).all()) for x in (actual[0], *actual[1]))
            unchanged = before == [tensor_hash(x) for x in (h, w, y, upstream)]
            directions = []
            if dtype == torch.float64 and index < 2:
                for _ in range(3):
                    dh = torch.randn(h.shape, generator=generator, dtype=dtype).cuda()
                    dw = torch.randn(w.shape, generator=generator, dtype=dtype).cuda()
                    norm = (dh.square().sum() + dw.square().sum()).sqrt()
                    dh, dw = dh / norm, dw / norm
                    analytical = sum(
                        (a * b).sum().item() for a, b in zip(actual[1], (dh, dw), strict=True)
                    )
                    with torch.no_grad():
                        plus = NativeBufferLoss.apply(h + 1e-6 * dh, w + 1e-6 * dw, y).item()
                        minus = NativeBufferLoss.apply(h - 1e-6 * dh, w - 1e-6 * dw, y).item()
                    discrepancy = abs((plus - minus) / 2e-6 * upstream.item() - analytical)
                    directions.append(
                        dict(
                            error=discrepancy, passed=discrepancy <= 1e-7 * max(1, abs(analytical))
                        )
                    )
            records.append(
                dict(
                    shape=[n, v, d],
                    dtype=str(dtype),
                    hidden_stride=list(h.stride()),
                    bitwise_equal=equal,
                    finite=finite,
                    inputs_unchanged=unchanged,
                    max_abs=[
                        (a - b).abs().max().item()
                        for a, b in zip(actual[1], reference[1], strict=True)
                    ],
                    directions=directions,
                    passed=all(equal)
                    and finite
                    and unchanged
                    and all(x["passed"] for x in directions),
                )
            )
            write_json(ROOT / "qualification_progress.json", records)
    return dict(
        passed=all(x["passed"] for x in records),
        records=records,
        backwards=16,
        finite_difference_forwards=12,
    )


def run():
    p = read(ROOT / "protocol.json")
    for key in ("sources", "input_hashes", "maintained_files"):
        hashes(p[key])
    prior, refs = [
        read("results/checkpoint_input_offload_v1/" + n) for n in ("protocol.json", "result.json")
    ]
    check_inputs(prior)
    write_json(ROOT / "environment.json", environment())
    boundaries = [boundary()]
    qualification = qualify()
    write_json(ROOT / "qualification.json", qualification)
    boundaries.append(boundary())
    print("Qualification:", qualification["passed"], flush=True)
    cases = []
    if qualification["passed"]:
        for index, fixture in enumerate(p["fixtures"]):
            for arm in ("native", "reuse") if index == 0 else ("reuse", "native"):
                fn = previous.training_loss if arm == "native" else model_loss
                with (
                    patch.object(previous, "ROOT", ROOT / arm),
                    patch.object(previous, "training_loss", fn),
                ):
                    measurement = previous.run_case(fixture, "resident", p, prior, refs)
                cases.append(dict(arm=arm, fixture=fixture, measurement=measurement))
                boundaries.append(boundary())
                write_json(ROOT / "progress.json", dict(completed_cases=len(cases)))
    write_json(
        ROOT / "result.json",
        dict(
            qualification=qualification,
            cases=cases,
            boundaries=boundaries,
            backwards=16 + 10 * len(cases),
            training_updates=0,
            diagnostic_targets=40960 * len(cases),
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
