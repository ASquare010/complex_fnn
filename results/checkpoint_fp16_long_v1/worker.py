"""FP16 checkpoint adapter around the unchanged fresh-training loop."""

# ruff: noqa: I001
from results.batch_scale_long_v1 import worker as base
from results.checkpoint_fp16_v1.codec import compress_inputs
from src.core.training_memory import buffer_model_loss
from results.ordinary_long_training_v1.io import write_json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import json

ROOT = Path("results/checkpoint_fp16_long_v1")
loop = base.loop


def one(row, p, folder):
    stack = ExitStack()
    logs = None
    selected = []
    original = loop.training_loss

    def loss(model, tokens, targets, policy, chunk=512):
        nonlocal logs
        if row["arm"] == "ordinary":
            return original(model, tokens, targets, policy, chunk)
        assert policy == "fp32_default_native"
        if logs is None:
            logs = stack.enter_context(compress_inputs(model, half=True))
            selected.extend(i for i, b in enumerate(model.blocks) if "forward" in b.__dict__)
        return buffer_model_loss(model, tokens, targets)

    try:
        with (
            patch.object(loop, "ROOT", folder),
            patch.object(loop, "MemoryLedger", base.HostLedger),
            patch.object(loop, "training_loss", loss),
            patch.object(loop, "write_json", write_json),
        ):
            result = loop.run_case(row["fixture"], "fp32_default_native", p)
            assert selected == (list(range(8)) if row["arm"] == "fp16" else [])
            if logs is not None:
                assert (
                    len(logs["records"]) == logs["unpack_count"][0] == 6408
                    and not logs["originals"]
                )
                assert all(
                    v["shape"] == [16, 512, 384] and v["stored_bytes"] == 6291456
                    for v in logs["records"]
                )
            return dict(
                **result,
                wrapped_blocks=selected,
                compressed_inputs=0 if logs is None else len(logs["records"]),
                unpacked_inputs=0 if logs is None else logs["unpack_count"][0],
            )
    finally:
        stack.close()


if __name__ == "__main__":
    assert json.loads((ROOT / "preflight.json").read_text())["passed"]
    assert (ROOT / "preflight_exit.txt").read_text().strip() == "0"
    base.ROOT = ROOT
    base.one = one
    base.run()
