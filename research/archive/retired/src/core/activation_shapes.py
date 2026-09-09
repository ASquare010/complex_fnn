"""Selected-checkpoint activation shapes and full-validation reset interventions."""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from src.core.activation_screen import CACHE
from src.core.benchmark import evaluate
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer


def worker(run, output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    path = Path("results/runs") / run
    metrics = json.loads((path / "metrics.json").read_text())
    source_hash = sha256(path / "checkpoint.pt")
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**metrics["model"]), metrics["training"]["seed"])
    model.load_state_dict(checkpoint["model"], strict=True)
    grid = torch.linspace(-6, 6, 257, dtype=torch.float64)
    records = []
    for index, block in enumerate(model.blocks):
        original = block.ffn.curve
        curve = type(original)(original.groups, original.groups).double()
        curve.load_state_dict(original.state_dict(), strict=True)
        z = grid[:, None].expand(-1, original.groups).clone().requires_grad_(True)
        value = curve(z)
        correction = value - F.silu(z)
        derivative = torch.autograd.grad(value.sum(), z)[0]
        assert torch.isfinite(value).all() and torch.isfinite(derivative).all()
        records.append(
            {
                "layer": index,
                "shape": curve.shape_parameters(),
                "theta": {n: p.detach().tolist() for n, p in curve.named_parameters()},
                "correction": correction.detach().T.tolist(),
                "derivative": derivative.T.tolist(),
                "grid_correction_rms": correction.square().mean().sqrt().item(),
                "grid_correction_abs_max": correction.abs().max().item(),
                "grid_derivative_min": derivative.min().item(),
                "grid_derivative_max": derivative.max().item(),
            }
        )
    del checkpoint
    model.cuda()
    data = TokenData(CACHE, "cuda", 10017)
    original_nll, targets = evaluate(model, data.validation(16, 128, 158), "cuda", "bf16")
    assert targets == 322688 and abs(original_nll - metrics["validation_loss"]) < 1e-7
    common = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".curve." not in n
    }
    with torch.no_grad():
        for block in model.blocks:
            for p in block.ffn.curve.parameters():
                p.zero_()
    assert all(
        torch.equal(p.detach().cpu(), common[n]) for n, p in model.named_parameters() if n in common
    )
    reset_nll, reset_targets = evaluate(model, data.validation(16, 128, 158), "cuda", "bf16")
    assert targets == reset_targets and sha256(path / "checkpoint.pt") == source_hash
    result = {
        "run": run,
        "checkpoint_sha256": source_hash,
        "source_checkpoint_unchanged": True,
        "common_trained_parameters_unchanged": True,
        "validation_targets": targets,
        "original_nll": original_nll,
        "reset_nll": reset_nll,
        "reset_relative_nll_percent": 100 * (reset_nll / original_nll - 1),
        "grid": grid.tolist(),
        "layers": records,
        "environment": environment(),
        "provenance": provenance(),
        "interpretation": "Zeroing curves after training measures checkpoint dependence; it is not retraining or a causal attribution of the training gain.",
    }
    write_json(output / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("grid", "layers", "provenance")}),
        flush=True,
    )


def audit(
    output,
    selection=Path("results/activation_screen_v1/result.json"),
    plan=Path("research/learnable_activation_plan.md"),
):
    selected = json.loads(selection.read_text())
    assert selected["status"] == "complete"
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "selected": selected["selected"],
        "provenance": provenance(),
        "plan_sha256": sha256(plan),
        "grid_range": [-6, 6],
        "grid_points": 257,
        "validation_targets": 322688,
        "intervention": "Zero every activation theta; keep all other trained parameters unchanged.",
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
    completed = []
    for recipe, run in selected["selected"].items():
        with (output / f"{recipe}.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "src.core.activation_shapes",
                    "worker",
                    "--run",
                    run,
                    "--output",
                    str(output / recipe),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        result = json.loads((output / recipe / "result.json").read_text())
        completed.append(
            {
                "recipe": recipe,
                "run": run,
                "result_sha256": sha256(output / recipe / "result.json"),
                "reset_relative_nll_percent": result["reset_relative_nll_percent"],
            }
        )
        write_json(output / "progress.json", {"status": "running", "completed": completed})
        print(json.dumps(completed[-1]), flush=True)
    write_json(output / "result.json", {"status": "complete", "cases": completed})
    write_json(output / "progress.json", {"status": "complete", "completed": completed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--run")
    parser.add_argument(
        "--selection", type=Path, default=Path("results/activation_screen_v1/result.json")
    )
    parser.add_argument("--plan", type=Path, default=Path("research/learnable_activation_plan.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        if not args.run:
            parser.error("worker requires run")
        worker(args.run, args.output)
    else:
        audit(args.output, args.selection, args.plan)
