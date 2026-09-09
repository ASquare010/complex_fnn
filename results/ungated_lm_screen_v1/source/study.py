"""H076 isolated language screen using the unchanged shared trainer and GELU operator."""

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch

from results.ungated_resource_v1.source import study as qualified
from src.blockshuffle_ffn import BlockShuffleFFN
from src.core import benchmark, trainer
from src.core import transformer as decoder
from src.core.config import ModelConfig, TrainConfig
from src.core.native_recompute_audit import digest, finite_tree, read, sha, tensor_record, write_new
from src.core.optimization import initialize_dense_width
from src.core.reproducibility import environment, provenance

ROOT = Path("results/ungated_lm_screen_v1")
PLAN = Path("research/ungated_lm_screen_plan.md")
CACHE = Path("data/wikitext2_v1")
FORMS = qualified.FORMS
RATES = (0.0003, 0.0006, 0.0012)
MODES = {f: "block_inner" if f == "narrow_gelu" else "block" for f in FORMS}
CELLS = {f"{f}_lr{round(rate * 1e6)}": {"form": f, "rate": rate} for f in FORMS for rate in RATES}
BASE_FACTORY = decoder.make_ffn
BASE_CLIP = torch.nn.utils.clip_grad_norm_


def configuration(form, rate):
    variant, hidden = FORMS[form]
    cls = qualified.UngatedConfig if variant == "blockshuffle_gelu" else ModelConfig
    mc = cls(
        variant=variant,
        width=384,
        layers=8,
        heads=6,
        context=128,
        vocab_size=4096,
        hidden=hidden,
        groups=8,
    )
    narrow = form.startswith("narrow_")
    tc = TrainConfig(
        steps=200,
        batch_size=16,
        learning_rate=rate,
        seed=17,
        eval_batches=158,
        log_every=50,
        precision="bf16",
        device="cuda",
        ffn_lr_mode="fan_in" if variant.startswith("blockshuffle") else "uniform",
        ffn_decay_mode="product" if narrow else "parameter",
        recompute_gate=False,
        ffn_width_init_mode="fan_in" if narrow else "none",
        ffn_width_lr_mode="fan_in" if narrow else "none",
    )
    mc.validate()
    tc.validate()
    return mc, tc


def make_model(config, seed, mode):
    def factory(mc):
        if isinstance(mc, qualified.UngatedConfig):
            return BlockShuffleFFN(mc.width, mc.ffn_width, mc.groups, gated=False)
        return BASE_FACTORY(mc)

    with patch.object(decoder, "make_ffn", factory):
        model = decoder.Transformer(config, seed)
    model.set_recompute_scope("none" if mode == "none" else "block")
    if mode == "block_inner":
        for block in model.blocks:
            block.ffn.forward = MethodType(qualified.inner_forward, block.ffn)
    return model


def operation_counts(mc):
    if not isinstance(mc, qualified.UngatedConfig):
        return benchmark.forward_flops(mc)
    ffn = 2 * mc.unique_ffn_parameters
    total = (
        mc.layers * (8 * mc.width**2 + 4 * mc.context * mc.width)
        + ffn
        + 2 * mc.width * mc.vocab_size
    )
    return {
        "ffn_matrix_forward_flops_per_token": ffn,
        "model_matrix_forward_flops_per_token": total,
        "training_matrix_flops_per_token_estimate": 3 * total,
        "flop_limitations": "Dense attention upper estimate; excludes norms, softmax, nonlinearities, loss and optimizer",
    }


class Observer:
    """Record native scalar results without changing a loss, gradient or update."""

    def __init__(self, output, config, mode):
        self.output, self.config, self.mode = Path(output), config, mode
        self.pending = None
        self.steps = []

    def constructor(self, mc, seed):
        model = make_model(mc, seed, self.mode)
        observer = self

        def loss(this, x, y):
            value = decoder.Transformer.loss(this, x, y)
            if torch.is_grad_enabled():
                assert observer.pending is None
                observer.pending = value.detach()
            return value

        model.loss = MethodType(loss, model)
        return model

    def initialize(self, model, config):
        result = initialize_dense_width(model, config)
        write_new(
            self.output / "initial_state_signature.json",
            {
                "weights": digest(model.state_dict()),
                "parameters": {n: tensor_record(p) for n, p in model.named_parameters()},
                "width_initialization": result,
                "execution_mode": self.mode,
            },
        )
        return result

    def clip(self, parameters, max_norm, *args, **kwargs):
        norm = BASE_CLIP(parameters, max_norm, *args, **kwargs)
        assert self.pending is not None
        row = {
            "step": len(self.steps) + 1,
            "training_loss": self.pending.item(),
            "gradient_norm_pre_clip": norm.item(),
            "learning_rate": trainer.learning_rate(len(self.steps), self.config),
        }
        self.pending = None
        assert finite_tree(row)
        self.steps.append(row)
        with (self.output / "all_steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        return norm


def select_and_gates(rows):
    if set(rows) != set(CELLS):
        raise ValueError("All 21 allocated cells are required")
    selected = {
        f: min(
            (c for c, s in CELLS.items() if s["form"] == f),
            key=lambda c: (rows[c]["validation_loss"], CELLS[c]["rate"]),
        )
        for f in FORMS
    }
    complete = len(rows) == 21 and all(
        r["status"] == "SCREENED" and r["all_finite"] for r in rows.values()
    )
    gates = {}
    for f in ("gelu_same", "gelu_matched"):
        candidate = rows[selected[f]]
        gates[f] = {
            "all_cells_complete_finite": complete,
            "at_least_70_percent_fewer_ffn_weights": candidate["ffn_reduction_percent"] >= 70,
            **{
                f"nll_within_one_percent_{ref}": candidate["validation_loss"]
                <= 1.01 * rows[selected[ref]]["validation_loss"]
                for ref in ("full_swiglu", "full_gelu")
            },
            **{
                f"nll_beats_{ref}": candidate["validation_loss"]
                < rows[selected[ref]]["validation_loss"]
                for ref in ("narrow_swiglu", "narrow_gelu")
            },
            **{
                f"memory_within_ten_percent_{ref}": candidate["peak_allocated_vram_bytes"]
                <= 1.1 * rows[selected[ref]]["peak_allocated_vram_bytes"]
                for ref in ("full_swiglu", "full_gelu")
            },
        }
    return selected, gates


def worker(cell):
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    assert environment() == protocol["environment"]
    assert all(sha(CACHE / n) == h for n, h in protocol["data_files"].items())
    spec = CELLS[cell]
    mc, tc = configuration(spec["form"], spec["rate"])
    output = ROOT / "runs" / cell
    assert not output.exists()
    observer = Observer(output, tc, MODES[spec["form"]])

    def full_provenance():
        prov = provenance()
        prov["source_files"] = dict(protocol["sources"])
        prov["source_hash"] = hashlib.sha256(
            json.dumps(prov["source_files"], sort_keys=True).encode()
        ).hexdigest()
        return prov

    with (
        patch.object(trainer, "Transformer", observer.constructor),
        patch.object(trainer, "initialize_dense_width", observer.initialize),
        patch.object(trainer, "provenance", full_provenance),
        patch.object(trainer, "forward_flops", operation_counts),
        patch.object(torch.nn.utils, "clip_grad_norm_", observer.clip),
    ):
        metrics = trainer.train(mc, tc, CACHE, output)
    assert observer.pending is None and len(observer.steps) == 200
    assert metrics["validation_tokens"] == 322688 and metrics["training_tokens"] == 409600
    assert (
        metrics["clipped_step_fraction"]
        == sum(r["gradient_norm_pre_clip"] > 1 for r in observer.steps) / 200
    )
    ckpt = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 200 and finite_tree(ckpt)
    assert all(float(v["step"]) == 200 for v in ckpt["optimizer"]["state"].values())
    for phase in ("initial", "final"):
        diagnostics = read(output / (phase + "_diagnostics.json"))
        assert finite_tree(diagnostics) and all(v["finite"] for v in diagnostics.values())
    history = list(map(json.loads, (output / "history.jsonl").read_text().splitlines()))
    for row in history:
        assert all(
            row[k] == observer.steps[row["step"] - 1][k]
            for k in ("step", "training_loss", "gradient_norm_pre_clip", "learning_rate")
        )
    prior = read("results/ungated_resource_recovery_v1/result.json")
    initial_root = Path(prior["origins"][spec["form"] + "_none"]["root"])
    assert (
        read(output / "initial_state_signature.json")["weights"]
        == read(initial_root / "workers" / (spec["form"] + "_none") / "initial_signature.json")[
            "weights"
        ]
    )
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    write_new(
        output / "qualification.json",
        {
            "status": "PASS",
            "all_finite": True,
            "initial_weights_match_h075": True,
            "all_steps_match_shared_history": True,
            "execution_mode": MODES[spec["form"]],
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "metrics_sha256": sha(output / "metrics.json"),
            "checkpoint_sha256": sha(output / "checkpoint.pt"),
            "source_archive_sha256": sha(output / "source.zip"),
            "all_steps_sha256": sha(output / "all_steps.jsonl"),
            "sampling_rng": tensor_record(ckpt["sampling_rng"]),
            "final_weights": digest(ckpt["model"]),
            "final_optimizer": digest(ckpt["optimizer"]),
            "official_test_scored": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "cell": cell,
                "validation_loss": metrics["validation_loss"],
                "peak_mib": metrics["peak_allocated_vram_bytes"] / 2**20,
            }
        ),
        flush=True,
    )


def finish():
    protocol = read(ROOT / "protocol.json")
    rows = {}
    for cell in CELLS:
        folder = ROOT / "runs" / cell
        q, m = read(folder / "qualification.json"), read(folder / "metrics.json")
        event = read(ROOT / "processes" / (cell + ".json"))
        assert (
            q["status"] == event["status"] == "PASS"
            and event["source_unchanged"]
            and event["returncode"] == 0
        )
        assert (
            sha(folder / "metrics.json") == q["metrics_sha256"]
            and sha(folder / "checkpoint.pt") == q["checkpoint_sha256"]
        )
        rows[cell] = {**m, "all_finite": q["all_finite"], "execution_mode": q["execution_mode"]}
    selected, gates = select_and_gates(rows)
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    result = {
        "status": "complete",
        "rows": rows,
        "selected_cells": selected,
        "gates": gates,
        "earns_longer_comparison": {f: all(g.values()) for f, g in gates.items()},
        "language_trials": 21,
        "language_optimizer_updates": 4200,
        "language_training_targets": 8601600,
        "shared_validation_target_exposures": 47435136,
        "qualification_cpu_updates": 8,
        "qualification_cpu_training_targets": 256,
        "scientific_attempts": 1,
        "scientific_retries": 0,
        "official_test_scored": False,
        "research_goal_achieved": False,
    }
    write_new(ROOT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    parser.add_argument("--cell", choices=list(CELLS))
    args = parser.parse_args()
    worker(args.cell) if args.command == "worker" else finish()
