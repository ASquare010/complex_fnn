"""Reuse the qualified probe loop; measure actual released BLAS allocation."""

# ruff: noqa: I001
from results.blas_workspace_v1.common import configure, warm, torch
from results.checkpoint_input_offload_v1.source import common as c
from results.gradient_staging_v1 import study as loop
from results.native_buffer_layout_v1.operator import model_loss
from pathlib import Path
from unittest.mock import patch
import gc
import sys

ROOT = Path("results/blas_workspace_v1")


def one(f, p, prior, refs, mode, arm):
    scores = []

    def construct(f, prior):
        values = c.construct(f, prior)
        scores.append(warm(values[0], values[2]))
        return values

    with (
        patch.object(loop, "ROOT", ROOT / mode / arm),
        patch.object(loop, "construct", construct),
        patch.object(loop, "training_loss", model_loss if arm == "reuse" else c.training_loss),
    ):
        measurement = loop.run_case(f, "resident", p, prior, refs)
    assert len(scores) == 1 and scores[0]["targets"] == 4096
    return dict(arm=arm, fixture=f, measurement=measurement, evaluation=scores[0])


def run():
    mode = sys.argv[1]
    p = c.read(ROOT / "protocol.json")
    for key in ("sources", "input_hashes", "maintained_files"):
        c.hashes(p[key])
    prior, refs = [
        c.read("results/checkpoint_input_offload_v1/" + n) for n in ("protocol.json", "result.json")
    ]
    c.check_inputs(prior)
    settings = configure(mode)
    boundaries = [c.boundary()]
    rows = []
    for i, f in enumerate(p["fixtures"]):
        for arm in ("native", "reuse") if i == 0 else ("reuse", "native"):
            row = one(f, p, prior, refs, mode, arm)
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()
            row["workspace_release"] = dict(before=before, after=after, released=before - after)
            assert after == 0
            rows.append(row)
            boundaries.append(c.boundary())
            c.write_json(ROOT / (mode + "_progress.json"), dict(completed=len(rows)))
    c.write_json(
        ROOT / (mode + "_result.json"),
        dict(
            cases=rows,
            settings=settings,
            boundaries=boundaries,
            backwards=40,
            training_updates=0,
            diagnostic_targets=163840,
            evaluation_targets=16384,
        ),
    )


if __name__ == "__main__":
    run()
