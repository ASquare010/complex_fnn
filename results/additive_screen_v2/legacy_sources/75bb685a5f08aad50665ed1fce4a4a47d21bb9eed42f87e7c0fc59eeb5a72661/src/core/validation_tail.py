"""Evaluate disjoint, previously unscored validation windows without training."""

import argparse
import gc
import json
import math
import traceback
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from src.core.benchmark import autocast, evaluate_forward
from src.core.compile_serving import load_model
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.scale_report import RECIPES


def window_batches(tokens, context, batch_size, start_window, stop_window):
    maximum = (tokens.numel() - 1) // context if context > 0 else 0
    if context <= 0 or batch_size <= 0 or not 0 <= start_window < stop_window <= maximum:
        raise ValueError("Invalid complete-window range")
    for start in range(start_window, stop_window, batch_size):
        offsets = (
            torch.arange(start, min(start + batch_size, stop_window), device=tokens.device)
            * context
        )
        ids = offsets[:, None] + torch.arange(context + 1, device=tokens.device)
        chunk = tokens[ids]
        yield start, chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def per_window_losses(forward, batches, device, precision):
    rows = []
    for start, x, y in batches:
        with autocast(device, precision):
            logits = forward(x)
            losses = (
                F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
                .reshape_as(y)
                .mean(-1)
            )
        for j, value in enumerate(losses.tolist()):
            if not math.isfinite(value):
                raise ValueError("Nonfinite held-out loss")
            window = start + j
            rows.append(
                {
                    "window": window,
                    "first_target": window * y.shape[1] + 1,
                    "last_target": (window + 1) * y.shape[1],
                    "targets": y.shape[1],
                    "nll": value,
                }
            )
    if not rows:
        raise ValueError("No complete windows")
    count = sum(r["targets"] for r in rows)
    mean = sum(r["nll"] * r["targets"] for r in rows) / count
    return mean, count, rows


@torch.no_grad()
def audit(output: Path):
    paths = [
        Path(f"results/runs/scale384_{recipe}_s{seed}_800")
        for seed in (17, 29, 43)
        for recipe in RECIPES
    ]
    if not all((p / "metrics.json").exists() for p in paths):
        raise ValueError("All twelve frozen checkpoints must finish first")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = TokenData(Path("data/tinystories_v1"), "cuda", 10017)
    assert data.valid.numel() == 200803
    maximum = (data.valid.numel() - 1) // 128
    assert maximum == 1568
    protocol = {
        "environment": environment(),
        "provenance": provenance(),
        "plan_sha256": sha256(Path("research/validation_tail_plan.md")),
        "data_hashes": data.manifest["files"],
        "context": 128,
        "batch_size": 16,
        "prefix_windows": [0, 256],
        "tail_windows": [256, maximum],
        "tail_target_positions_inclusive": [32769, 200704],
        "tail_target_count": 167936,
        "runs": [p.name for p in paths],
        "precision": "native CUDA BF16",
        "timing_claim": False,
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write("research/validation_tail_plan.md", "research/validation_tail_plan.md")
    records = {}
    try:
        for path in paths:
            print(f"Evaluating {path.name}", flush=True)
            metrics = json.loads((path / "metrics.json").read_text())
            assert metrics["data"]["files"] == data.manifest["files"]
            model = load_model(path)
            prefix, count = evaluate_forward(model, data.validation(16, 128, 16), "cuda", "bf16")
            if count != 32768 or abs(prefix - metrics["validation_loss"]) > 0.0002:
                raise ValueError(f"Prefix reproduction failed for {path.name}")
            tail, tail_count, rows = per_window_losses(
                model, window_batches(data.valid, 128, 16, 256, maximum), "cuda", "bf16"
            )
            assert tail_count == 167936 and len(rows) == 1312
            assert rows[0]["first_target"] == 32769 and rows[-1]["last_target"] == 200704
            assert all(b["first_target"] == a["last_target"] + 1 for a, b in zip(rows, rows[1:]))
            record = {
                "run": path.name,
                "checkpoint_sha256": sha256(path / "checkpoint.pt"),
                "metrics_sha256": sha256(path / "metrics.json"),
                "prefix_nll": prefix,
                "prefix_reference_nll": metrics["validation_loss"],
                "prefix_target_count": count,
                "tail_nll": tail,
                "tail_target_count": tail_count,
                "aggregate_nll": (prefix * count + tail * tail_count) / (count + tail_count),
                "aggregate_target_count": count + tail_count,
                "windows": rows,
            }
            write_json(output / f"{path.name}.json", record)
            records[path.name] = {k: v for k, v in record.items() if k != "windows"}
            write_json(output / "progress.json", {"status": "running", "completed": records})
            print(json.dumps(records[path.name]), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result = {
            "protocol": protocol,
            "records": records,
            "status": "complete",
            "interpretation": "Previously unscored tail within the existing validation corpus; not a second corpus, convergence or new training result. This tail is no longer unscored for future model selection.",
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": records})
        return result
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "completed": list(records),
                "protocol": protocol,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output)
