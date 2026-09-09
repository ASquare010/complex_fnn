"""Full-validation reset and plain-model deployment copies for all affine seeds."""

import argparse
import json
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.activation_retrofit import remove_zero_learnable_activations
from src.core.affine_longer import output_path
from src.core.benchmark import autocast, evaluate
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.transformer import Transformer

PLAN = Path("research/affine_activation_removal_plan.md")
SEEDS = (17, 29, 43)
CACHE = Path("data/wikitext2_v1")


def worker(seed, output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    path = output_path("blockshuffle_affine", seed)
    m = json.loads((path / "metrics.json").read_text())
    checkpoint_hash = sha256(path / "checkpoint.pt")
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 800
    model = Transformer(ModelConfig(**m["model"]), seed).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    data = TokenData(CACHE, "cuda", 10000 + seed)
    assert data.manifest["files"] == m["data"]["files"]

    def validation(target):
        nll, count = evaluate(target, data.validation(16, 128, 158), "cuda", "bf16")
        assert count == 322688
        return nll

    original_nll = validation(model)
    assert abs(original_nll - m["validation_loss"]) < 1e-7
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
    reset_nll = validation(model)
    plain = remove_zero_learnable_activations(model)
    model.eval()
    plain.eval()
    x, _ = next(data.validation(16, 128, 1))
    with torch.no_grad(), autocast("cuda", "bf16"):
        a = model(x)
        b = plain(x)
    assert torch.equal(a, b) and torch.isfinite(a).all()
    assert all(torch.equal(p.detach().cpu(), common[n]) for n, p in plain.named_parameters())
    del a, b, model, common
    plain_nll = validation(plain)
    assert plain_nll == reset_nll
    weights = {n: t.detach().cpu() for n, t in plain.state_dict().items()}
    assert sum(t.numel() for t in weights.values()) == 9099648
    torch.save(
        {
            "model": weights,
            "model_config": asdict(plain.config),
            "source_step": 800,
            "source_checkpoint_sha256": checkpoint_hash,
            "optimizer_steps_after_source": 0,
            "intervention": "Zero all activation theta parameters after training, then remove the zero correction modules. No projection or other common weight changed.",
        },
        output / "stripped_checkpoint.pt",
    )
    references = {
        k: json.loads((output_path(k, seed) / "metrics.json").read_text())
        for k in ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")
    }
    gates = {
        "reset_nll_cost_within_point_one_percent": reset_nll <= 1.001 * original_nll,
        "at_least_70_percent_fewer_ffn_weights": plain.config.unique_ffn_parameters
        <= 0.3 * 9437184,
        "beats_calibrated_narrow": reset_nll < references["calibrated_narrow"]["validation_loss"],
        "at_least_point_two_percent_better_than_unmodified_blockshuffle": reset_nll
        <= 0.998 * references["blockshuffle"]["validation_loss"],
    }
    for key in ("full_swiglu", "full_gelu"):
        gates[f"within_one_percent_{key}"] = reset_nll <= 1.01 * references[key]["validation_loss"]
    assert sha256(path / "checkpoint.pt") == checkpoint_hash
    result = {
        "seed": seed,
        "source_run": path.name,
        "original_nll": original_nll,
        "reset_nll": reset_nll,
        "plain_nll": plain_nll,
        "relative_reset_cost_percent": 100 * (reset_nll / original_nll - 1),
        "gates": gates,
        "passes": all(gates.values()),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_unchanged": True,
        "all_common_weights_exact": True,
        "reset_plain_sampled_logits_exact": True,
        "reset_plain_full_nll_exact": True,
        "validation_targets": 322688,
        "stripped_model_config": asdict(plain.config),
        "ffn_parameters": plain.config.unique_ffn_parameters,
        "total_parameters": plain.config.total_parameters,
        "stripped_checkpoint_sha256": sha256(output / "stripped_checkpoint.pt"),
        "plan_sha256": sha256(PLAN),
        "data_hashes": data.manifest["files"],
        "environment": environment(),
        "provenance": provenance(),
    }
    write_json(output / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("provenance", "data_hashes")}),
        flush=True,
    )


def audit(output):
    assert json.loads(Path("results/affine_replication_v1/result.json").read_text())[
        "replication_passes"
    ]
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "plan_sha256": sha256(PLAN),
        "provenance": provenance(),
        "seeds": SEEDS,
        "worker_timeout_seconds": 900,
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    completed = []
    try:
        for seed in SEEDS:
            with (output / f"s{seed}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "src.core.affine_removal",
                        "worker",
                        "--seed",
                        str(seed),
                        "--output",
                        str(output / f"s{seed}"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    check=True,
                )
            r = json.loads((output / f"s{seed}/result.json").read_text())
            completed.append(
                {
                    "seed": seed,
                    "original_nll": r["original_nll"],
                    "reset_nll": r["reset_nll"],
                    "relative_reset_cost_percent": r["relative_reset_cost_percent"],
                    "passes": r["passes"],
                    "result_sha256": sha256(output / f"s{seed}/result.json"),
                }
            )
            write_json(output / "progress.json", {"status": "running", "completed": completed})
            print(json.dumps(completed[-1]), flush=True)
        result = {
            "status": "complete",
            "cases": completed,
            "earns_fused_inference_audit": all(r["passes"] for r in completed),
            "plan_sha256": sha256(PLAN),
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": "complete", "completed": completed})
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "completed": completed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure.json", failure)
        write_json(output / "progress.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        if args.seed is None:
            parser.error("worker requires seed")
        worker(args.seed, args.output)
    else:
        audit(args.output)
