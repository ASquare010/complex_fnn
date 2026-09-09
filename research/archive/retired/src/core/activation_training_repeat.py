"""One bounded full-training repeat of the selected rational activation backend."""

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import torch

from src.core.activation_screen import CACHE, controls, promotion
from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train

CONFIG = Path("configs/wikitext2_blockshuffle_rational_compiled.json")
PLAN = Path("research/activation_training_repeat_plan.md")
RUN = Path("results/runs/wikitext2_blockshuffle_rational_compiled_lr1200_s17_200")
NATIVE = Path("results/runs/wikitext2_blockshuffle_rational_lr1200_s17_200")


def configuration(config_path=CONFIG):
    raw = json.loads(config_path.read_text())
    return ModelConfig(**raw["model"]), TrainConfig(**raw["training"])


def source_check():
    with zipfile.ZipFile(NATIVE / "source.zip") as archive:
        old = ast.parse(archive.read("src/core/trainer.py").decode())
    new = ast.parse(Path("src/core/trainer.py").read_text())

    def loop(tree):
        train_fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"
        )
        return next(
            n
            for n in ast.walk(train_fn)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step"
        )

    assert ast.dump(loop(old)) == ast.dump(loop(new))
    return {"training_step_loop_ast_identical_to_native": True}


def verify(run_path=RUN, config_path=CONFIG):
    mc, tc = configuration(config_path)
    m = json.loads((run_path / "metrics.json").read_text())
    native = json.loads((NATIVE / "metrics.json").read_text())
    assert mc == ModelConfig(**m["model"]) and tc == TrainConfig(**m["training"])
    assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
    assert m["data"]["files"] == native["data"]["files"]
    assert m["optimizer_parameter_groups"] == native["optimizer_parameter_groups"]
    assert m["ffn_parameters"] == 2801984 and m["total_parameters"] == 9099968
    assert m["activation_compilation"]["all_parameters_unchanged_after_warmup"]
    assert not m["activation_compilation"]["sampling_rng_consumed"]
    assert abs(m["initial_validation_loss"] - native["initial_validation_loss"]) < 1e-7
    history = [json.loads(line) for line in (run_path / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in history] == [1, 50, 100, 150, 200]
    assert all(
        math.isfinite(r[k])
        for r in history
        for k in ("training_loss", "validation_loss", "gradient_norm_pre_clip")
    )
    for stage in ("initial", "final"):
        diag = json.loads((run_path / f"{stage}_diagnostics.json").read_text())
        assert all(r["finite"] and r.get("slope_finite", True) for r in diag.values())
    with zipfile.ZipFile(run_path / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in m["provenance"]["source_files"].items()
        )
    repeat = torch.load(run_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    original = torch.load(NATIVE / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert repeat["step"] == original["step"] == 200
    assert torch.equal(repeat["sampling_rng"], original["sampling_rng"])
    assert sum(p.numel() for p in repeat["model"].values()) == m["total_parameters"]
    gates = promotion(m, "blockshuffle", controls())
    gates["within_point_two_percent_native_rational_nll"] = (
        abs(m["validation_loss"] / native["validation_loss"] - 1) <= 0.002
    )
    return {
        "status": "complete",
        "run": run_path.name,
        "validation_nll": m["validation_loss"],
        "native_rational_nll": native["validation_loss"],
        "relative_native_nll_percent": 100 * (m["validation_loss"] / native["validation_loss"] - 1),
        "peak_allocated_vram_bytes": m["peak_allocated_vram_bytes"],
        "gates": gates,
        "repeat_passes": all(gates.values()),
        "sampling_rng_matches_native": True,
        "initial_validation_nll_matches_native": True,
        "warmup_parameters_unchanged": True,
        "config_matches": True,
        "all_source_archive_hashes_match": True,
        "metrics_sha256": sha256(run_path / "metrics.json"),
        "checkpoint_sha256": sha256(run_path / "checkpoint.pt"),
    }


def audit(output):
    assert json.loads(Path("results/activation_execution_v1/result.json").read_text())[
        "execution_passes"
    ]
    assert not RUN.exists()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "plan_sha256": sha256(PLAN),
        "config_sha256": sha256(CONFIG),
        "provenance": provenance(),
        "environment": environment(),
        "worker_timeout_seconds": 900,
        **source_check(),
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    try:
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "src.core.activation_training_repeat",
                    "worker",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=True,
            )
        result = verify()
        write_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(
            output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        mc, tc = configuration()
        train(mc, tc, CACHE, RUN)
    elif args.command == "verify":
        print(json.dumps(verify()), flush=True)
    else:
        if args.output is None:
            parser.error("audit requires output")
        audit(args.output)
