"""H081 isolated BLAST/optimizer screen; delegates language training to src/core."""

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from torch import nn

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.blast_operator_v1.source.model import BlastFFN, BlastLinear
from results.ungated_lm_screen_v1.source import study as previous
from src.blockshuffle_ffn import BlockShuffleLinear
from src.core import benchmark, trainer
from src.core.config import ModelConfig
from src.core.diagnostics import inspect_layers
from src.core.native_recompute_audit import digest, finite_tree, tensor_record
from src.core.optimization import parameter_groups as legacy_groups
from src.core.reproducibility import environment, provenance
from src.core.transformer import Transformer

ROOT = Path("results/blast_learning_screen_v1")
PLAN = Path("research/blast_learning_screen_plan.md")
CACHE = previous.CACHE
FORMS = ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain",
         "plain_calibrated", "blast_swiglu", "blast_gelu")
NEW_FORMS = FORMS[-3:]
RATES = (0.0003, 0.0006, 0.0012)
MODES = {f: "block_inner" if f == "narrow_gelu" else "block" for f in FORMS}
CELLS = {f"{f}_lr{round(r * 1e6)}": {"form": f, "rate": r} for f in FORMS for r in RATES}


@dataclass(frozen=True)
class StudyConfig(ModelConfig):
    factor_calibration: str = "initial_rms_product"
    rank: int = 48

    def validate(self):
        if self.factor_calibration != "initial_rms_product":
            raise ValueError("Unknown isolated factor calibration")
        if self.variant.startswith("blast_"):
            if self.variant not in ("blast_swiglu", "blast_gelu"):
                raise ValueError("Unknown BLAST form")
            hidden = 1984 if self.variant == "blast_swiglu" else 3200
            if (self.width, self.layers, self.groups, self.rank, self.hidden) != (384, 8, 8, 48, hidden):
                raise ValueError("H081 supports only its frozen BLAST dimensions")
            base = {k: v for k, v in asdict(self).items() if k in ModelConfig.__dataclass_fields__}
            ModelConfig(**{**base, "variant": "blockshuffle_swiglu"}).validate()
        else:
            if self.variant != "blockshuffle_swiglu":
                raise ValueError("Only BlockShuffle can use this isolated non-BLAST config")
            super().validate()

    @property
    def ffn_parameters(self):
        if self.variant.startswith("blast_"):
            projections = 3 if self.variant == "blast_swiglu" else 2
            return projections * self.rank * (self.width + self.ffn_width + self.groups**2)
        return super().ffn_parameters


class LanguageBlastFFN(BlastFFN):
    """Use the qualified forward/initializer with full layer-local seed names."""

    def __init__(self, form, seed, layer):
        nn.Module.__init__(self)
        self.form, self.width, self.hidden = form, 384, 1984 if form == "swiglu" else 3200
        prefix = f"blocks.{layer}.ffn"
        self.up = BlastLinear(384, self.hidden, seed=seed, name=prefix + ".up")
        self.gate = (BlastLinear(384, self.hidden, seed=seed, name=prefix + ".gate")
                     if form == "swiglu" else None)
        self.down = BlastLinear(self.hidden, 384, seed=seed, name=prefix + ".down",
                                residual_scale=0.25)


def configuration(form, rate):
    origin = "plain" if form in NEW_FORMS else form
    mc, tc = previous.configuration(origin, rate)
    if form in NEW_FORMS:
        values = asdict(mc)
        if form.startswith("blast_"):
            values.update(variant=form, hidden=1984 if form == "blast_swiglu" else 3200)
        mc = StudyConfig(**values)
        tc = replace(tc, ffn_lr_mode="uniform", ffn_decay_mode="product")
    mc.validate()
    tc.validate()
    return mc, tc


def make_model(mc, seed, mode):
    if not mc.variant.startswith("blast_"):
        return previous.make_model(mc, seed, mode)
    # Discard temporary dense FFNs before any optimizer/forward. Common tensors
    # are still initialized by the unchanged, name-local shared constructor.
    surrogate = ModelConfig(variant="swiglu_narrow", hidden=304, width=384, layers=8,
                            heads=mc.heads, context=mc.context, vocab_size=mc.vocab_size)
    model = Transformer(surrogate, seed)
    for i, block in enumerate(model.blocks):
        block.ffn = LanguageBlastFFN(mc.variant.removeprefix("blast_"), seed, i)
    model.config = mc
    model.set_recompute_scope(mode)
    assert sum(p.numel() for p in model.parameters()) == mc.total_parameters
    return model


def calibration(model):
    """Fixed initial parameter-relative scales, not functional natural gradients."""
    entries = {}
    if not isinstance(model.config, StudyConfig):
        return entries
    for name, module in model.named_modules():
        if isinstance(module, BlastLinear):
            parts = tuple(module.named_parameters(recurse=False))
        elif isinstance(module, BlockShuffleLinear):
            parts = (("first.weight", module.first.weight), ("second.weight", module.second.weight))
        else:
            continue
        count = len(parts)
        sigma = 0.02 / math.sqrt(2 * model.config.layers) if name.endswith(".down") else 0.02
        for label, parameter in parts:
            rms = parameter.detach().cpu().double().square().mean().sqrt().item()
            assert math.isfinite(rms) and rms > 0
            entries[name + "." + label] = {"initial_rms": rms, "dense_sigma": sigma,
                                           "factors": count, "lr_scale": rms / (count * sigma)}
    return entries


def parameter_groups(model, tc):
    if not isinstance(model.config, StudyConfig):
        return legacy_groups(model, tc)
    entries = calibration(model)
    grouped = {}
    for name, parameter in model.named_parameters():
        entry = entries.get(name)
        scale = entry["lr_scale"] if entry else 1.0
        decay = tc.weight_decay if parameter.ndim >= 2 else 0.0
        if entry:
            decay /= entry["factors"] * scale
        group = grouped.setdefault((decay, scale), {
            "params": [], "lr_scale": scale, "lr": tc.learning_rate * scale,
            "weight_decay": decay,
        })
        group["params"].append(parameter)
    model._initial_factor_calibration = entries
    return list(grouped.values())


def operation_counts(mc):
    ffn = 2 * mc.unique_ffn_parameters
    total = mc.layers * (8 * mc.width**2 + 4 * mc.context * mc.width) + ffn + 2 * mc.width * mc.vocab_size
    return {"ffn_matrix_forward_flops_per_token": ffn,
            "model_matrix_forward_flops_per_token": total,
            "training_matrix_flops_per_token_estimate": 3 * total,
            "flop_limitations": "Matrix MACs only; excludes activation, rearrangement, norm, softmax, loss and optimizer"}


@torch.no_grad()
def diagnostics(model, x, device, precision):
    records = inspect_layers(model, x, device, precision)
    handles = []
    for i, block in enumerate(model.blocks):
        def hook(module, inputs, output, i=i, gated=block.ffn.gate is not None):
            z = output.detach().reshape(-1, output.shape[-1])[:256].float()
            if gated:
                sigmoid = z.sigmoid()
                slope = sigmoid * (1 + z * (1 - sigmoid))
            else:
                slope = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z.square()/2) / math.sqrt(2*math.pi)
            records[f"layer_{i}.activation_slope"] = {
                "activation": "silu" if gated else "gelu", "samples": z.numel(),
                "absolute_mean": slope.abs().mean().item(), "absolute_max": slope.abs().max().item(),
                "below_1e_minus3_fraction": (slope.abs() < 0.001).float().mean().item(),
                "finite": bool(torch.isfinite(slope).all()),
            }
        handles.append(block.ffn.up.register_forward_hook(hook))
    try:
        with benchmark.autocast(device, precision):
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return records


class Observer(previous.Observer):
    def constructor(self, mc, seed):
        model = make_model(mc, seed, self.mode)
        observer = self

        def loss(this, x, y):
            value = Transformer.loss(this, x, y)
            if torch.is_grad_enabled():
                assert observer.pending is None
                observer.pending = value.detach()
            return value

        model.loss = MethodType(loss, model)
        return model


def select_and_gates(rows):
    if set(rows) != set(CELLS):
        raise ValueError("All 24 allocated cells are required")
    selected = {f: min((c for c, s in CELLS.items() if s["form"] == f),
                       key=lambda c: (rows[c]["validation_loss"], CELLS[c]["rate"])) for f in FORMS}
    complete = all(r["status"] == "SCREENED" and r["all_finite"] for r in rows.values())
    gates = {}
    for f in NEW_FORMS:
        c = rows[selected[f]]
        gates[f] = {
            "all_cells_complete_finite": complete,
            "at_least_70_percent_fewer_ffn_weights": c["ffn_reduction_percent"] >= 70,
            **{f"nll_within_one_percent_{ref}": c["validation_loss"] <= 1.01 * rows[selected[ref]]["validation_loss"]
               for ref in ("full_swiglu", "full_gelu")},
            **{f"nll_beats_{ref}": c["validation_loss"] < rows[selected[ref]]["validation_loss"]
               for ref in ("narrow_swiglu", "narrow_gelu")},
            **{f"memory_within_ten_percent_{ref}": c["peak_allocated_vram_bytes"] <= 1.1 * rows[selected[ref]]["peak_allocated_vram_bytes"]
               for ref in ("full_swiglu", "full_gelu")},
        }
    return selected, gates


def durable_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value, exclusive=not Path(path).exists())


def flush_files(folder):
    for path in folder.iterdir():
        if not path.is_file():
            continue
        before = sha(path)
        with path.open("r+b") as f:
            f.flush()
            os.fsync(f.fileno())
        assert sha(path) == before
        if path.suffix in (".zip", ".pt"):
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
        elif path.suffix == ".json":
            read(path)
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                json.loads(line)


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
        prov["source_files"] = protocol["sources"]
        prov["source_hash"] = hashlib.sha256(json.dumps(prov["source_files"], sort_keys=True).encode()).hexdigest()
        return prov

    def groups(model, config):
        value = parameter_groups(model, config)
        write_json(output / "factor_calibration.json", getattr(model, "_initial_factor_calibration", {}))
        return value

    with (patch.object(trainer, "Transformer", observer.constructor),
          patch.object(trainer, "initialize_dense_width", observer.initialize),
          patch.object(trainer, "parameter_groups", groups),
          patch.object(trainer, "provenance", full_provenance),
          patch.object(trainer, "forward_flops", operation_counts),
          patch.object(trainer, "inspect_layers", diagnostics),
          patch.object(trainer, "write_json", durable_json),
          patch.object(torch.nn.utils, "clip_grad_norm_", observer.clip)):
        metrics = trainer.train(mc, tc, CACHE, output)
    assert observer.pending is None and len(observer.steps) == 200
    assert metrics["training_tokens"] == 409600 and metrics["validation_tokens"] == 322688
    assert metrics["clipped_step_fraction"] == sum(r["gradient_norm_pre_clip"] > 1 for r in observer.steps) / 200
    flush_files(output)
    ckpt = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 200 and finite_tree(ckpt)
    assert all(float(v["step"]) == 200 for v in ckpt["optimizer"]["state"].values())
    for phase in ("initial", "final"):
        values = read(output / (phase + "_diagnostics.json"))
        assert finite_tree(values) and all(v["finite"] for v in values.values())
    history = list(map(json.loads, (output / "history.jsonl").read_text().splitlines()))
    assert all(all(row[k] == observer.steps[row["step"] - 1][k]
                   for k in ("step", "training_loss", "gradient_norm_pre_clip", "learning_rate")) for row in history)
    if spec["form"] not in NEW_FORMS:
        old = Path("results/ungated_lm_screen_v1/runs") / cell
        assert read(output / "initial_state_signature.json")["weights"] == read(old / "initial_state_signature.json")["weights"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    write_json(output / "qualification.json", {
        "status": "PASS", "all_finite": True, "execution_mode": MODES[spec["form"]],
        "artifact_hashes": {n.name: sha(n) for n in sorted(output.iterdir()) if n.is_file()},
        "sampling_rng": tensor_record(ckpt["sampling_rng"]), "final_weights": digest(ckpt["model"]),
        "final_optimizer": digest(ckpt["optimizer"]), "official_test_scored": False,
    })
    print(json.dumps({"status": "PASS", "cell": cell, "validation_loss": metrics["validation_loss"],
                      "peak_mib": metrics["peak_allocated_vram_bytes"] / 2**20}), flush=True)


def finish():
    protocol = read(ROOT / "protocol.json")
    rows = {}
    for cell in CELLS:
        folder = ROOT / "runs" / cell
        q, m = read(folder / "qualification.json"), read(folder / "metrics.json")
        event = read(ROOT / "processes" / (cell + ".json"))
        assert q["status"] == event["status"] == "PASS" and event["returncode"] == 0 and event["source_unchanged"]
        assert all(sha(folder / n) == h for n, h in q["artifact_hashes"].items())
        rows[cell] = {**m, "all_finite": True, "execution_mode": q["execution_mode"]}
    selected, gates = select_and_gates(rows)
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    result = {"status": "COMPLETE", "rows": rows, "selected_cells": selected, "gates": gates,
              "earns_longer_comparison": {f: all(g.values()) for f, g in gates.items()},
              "language_trials": 24, "language_optimizer_updates": 4800,
              "language_training_targets": 9830400, "shared_validation_target_exposures": 54211584,
              "qualification_updates": 24, "qualification_targets": 24960,
              "scientific_attempts": 1, "scientific_retries": 0,
              "official_test_scored": False, "research_goal_achieved": False}
    write_json(ROOT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    parser.add_argument("--cell", choices=list(CELLS))
    args = parser.parse_args()
    worker(args.cell) if args.command == "worker" else finish()
