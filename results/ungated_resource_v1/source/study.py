"""H074 isolated full-model resource experiment; active model registry is unchanged."""

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.blockshuffle_ffn import BlockShuffleFFN, swiglu_product
from src.core import transformer as decoder
from src.core.benchmark import autocast
from src.core.config import ModelConfig, TrainConfig
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.native_recompute_audit import (
    digest,
    finite_tree,
    read,
    sha,
    tensor_record,
    tokens,
    write_new,
)
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
from src.core.reproducibility import environment

ROOT = Path("results/ungated_resource_v1")
PLAN = Path("research/ungated_resource_plan.md")
FORMS = {
    "full_swiglu": ("swiglu", 1024),
    "full_gelu": ("gelu", 1536),
    "narrow_swiglu": ("swiglu_narrow", 304),
    "narrow_gelu": ("gelu_narrow", 456),
    "plain": ("blockshuffle_swiglu", 2048),
    "gelu_same": ("blockshuffle_gelu", 2048),
    "gelu_matched": ("blockshuffle_gelu", 3264),
}
MODES = ("none", "block", "block_inner")
CELLS = {f"{f}_{m}": {"form": f, "mode": m} for f in FORMS for m in MODES}


@dataclass(frozen=True)
class UngatedConfig(ModelConfig):
    variant: str = "blockshuffle_gelu"

    def validate(self):
        if self.variant != "blockshuffle_gelu":
            raise ValueError("Isolated configuration requires blockshuffle_gelu")
        ModelConfig(**{**asdict(self), "variant": "blockshuffle_swiglu"}).validate()

    @property
    def ffn_parameters(self):
        return 2 * min(self.width, self.ffn_width) * (self.width + self.ffn_width) // self.groups


def make_model(form):
    variant, hidden = FORMS[form]
    cls = UngatedConfig if variant == "blockshuffle_gelu" else ModelConfig
    mc = cls(variant=variant, width=384, layers=8, heads=6, hidden=hidden)
    factory = decoder.make_ffn

    def isolated_factory(config):
        if isinstance(config, UngatedConfig):
            config.validate()
            return BlockShuffleFFN(config.width, config.ffn_width, config.groups, gated=False)
        return factory(config)

    with patch.object(decoder, "make_ffn", isolated_factory):
        model = decoder.Transformer(mc, 17)
    narrow = form.startswith("narrow_")
    tc = TrainConfig(
        steps=20,
        seed=17,
        batch_size=16,
        learning_rate=0.0012,
        precision="bf16",
        ffn_lr_mode="fan_in",
        ffn_width_init_mode="fan_in" if narrow else "none",
        ffn_width_lr_mode="fan_in" if narrow else "none",
    )
    tc.validate()
    initialize_dense_width(model, tc)
    return model, mc, tc


def inner_forward(self, x):
    if not self.training or not torch.is_grad_enabled():
        return type(self).forward(self, x)
    u = self.up(x)
    if self.gate is None:
        z = checkpoint(F.gelu, u, use_reentrant=False, preserve_rng_state=False)
    else:
        z = checkpoint(
            swiglu_product, u, self.gate(x), use_reentrant=False, preserve_rng_state=False
        )
    return self.down(z)


def configure(model, optimizer, mode):
    if mode not in MODES:
        raise ValueError(mode)
    identities = {n: id(p) for n, p in model.named_parameters()}
    opt_ids = [[id(p) for p in g["params"]] for g in optimizer.param_groups]
    initial = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    model.set_recompute_scope("none" if mode == "none" else "block")
    if mode == "block_inner":
        for block in model.blocks:
            block.ffn.forward = MethodType(inner_forward, block.ffn)
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    assert opt_ids == [[id(p) for p in g["params"]] for g in optimizer.param_groups]
    assert initial == digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    return {
        "parameter_objects_preserved": True,
        "optimizer_references_preserved": True,
        "state_preserved": True,
        "mode": mode,
    }


@torch.no_grad()
def diagnostics(model, x):
    records = inspect_layers(model, x, "cuda", "bf16")
    slopes, handles = {}, []
    for index, block in enumerate(model.blocks):

        def hook(module, inputs, output, index=index, gated=block.ffn.gate is not None):
            z = output.detach().reshape(-1, output.shape[-1])[:256].float()
            if gated:
                sigmoid = z.sigmoid()
                slope = sigmoid * (1 + z * (1 - sigmoid))
            else:
                slope = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(
                    -z.square() / 2
                ) / math.sqrt(2 * math.pi)
            slopes[f"layer_{index}"] = {
                "activation": "silu" if gated else "gelu",
                "samples": z.numel(),
                "slope_abs_mean": slope.abs().mean().item(),
                "slope_abs_max": slope.abs().max().item(),
                "slope_below_1e_minus3_fraction": (slope.abs() < 0.001).float().mean().item(),
                "finite": bool(torch.isfinite(slope).all()),
            }

        handles.append(block.ffn.up.register_forward_hook(hook))
    try:
        with autocast("cuda", "bf16"):
            model(x)
    finally:
        for h in handles:
            h.remove()
    assert all(r["finite"] for r in records.values()) and all(r["finite"] for r in slopes.values())
    return {"layers": records, "sampled_activation_slopes": slopes}


def choose_modes(rows):
    return {
        f: min(
            MODES,
            key=lambda m: (
                rows[f"{f}_{m}"]["peak_allocated_bytes"],
                rows[f"{f}_{m}"]["median_step_ms"],
                MODES.index(m),
            ),
        )
        for f in FORMS
    }


def derive_gates(rows, exact):
    chosen = choose_modes(rows)
    gates = {}
    for form in ("gelu_same", "gelu_matched"):
        mode = chosen[form]
        row = rows[f"{form}_{mode}"]
        gates[form] = {
            "all_workers_and_fidelity_exact": exact
            and all(r["status"] == "PASS" for r in rows.values()),
            "at_least_70_percent_fewer_ffn_weights": row["ffn_reduction_percent"] >= 70,
            "all_diagnostics_and_state_finite": all(r["all_finite"] for r in rows.values()),
            **{
                f"memory_within_ten_percent_{f}": row["peak_allocated_bytes"]
                <= 1.1 * rows[f"{f}_{chosen[f]}"]["peak_allocated_bytes"]
                for f in ("full_swiglu", "full_gelu")
            },
            "time_within_25_percent_plain_same_mode": row["median_step_ms"]
            <= 1.25 * rows[f"plain_{mode}"]["median_step_ms"],
        }
    return chosen, gates


def worker(cell):
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    assert environment() == protocol["environment"]
    form, mode = CELLS[cell].values()
    folder = ROOT / "workers" / cell
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, mc, tc = make_model(form)
    actual_ffn = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
    actual_total = sum(p.numel() for p in model.parameters())
    assert (actual_ffn, actual_total) == (mc.unique_ffn_parameters, mc.total_parameters)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    adapter = configure(model, optimizer, mode)
    write_new(folder / "adapter.json", adapter)
    stream = tokens()
    assert tensor_record(stream) == protocol["token_record"]
    write_new(
        folder / "config.json",
        {
            "cell": cell,
            "form": form,
            "mode": mode,
            "model": asdict(mc),
            "training": asdict(tc),
            "actual_ffn_parameters": actual_ffn,
            "actual_total_parameters": actual_total,
            "optimizer_parameter_groups": group_summary(optimizer.param_groups),
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "environment": environment(),
            "synthetic": True,
            "start_step": 0,
            "updates": 20,
            "corpus_targets": 0,
        },
    )
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 0},
        folder / "initial.pt",
    )
    raw = stream[0].cuda()
    x, y = raw[:, :-1], raw[:, 1:]
    write_new(folder / "initial_diagnostics.json", diagnostics(model, x))
    cpu_rng, cuda_rng = (
        tensor_record(torch.get_rng_state()),
        tensor_record(torch.cuda.get_rng_state()),
    )
    initial = {
        "weights": digest(model.state_dict()),
        "optimizer": digest(optimizer.state_dict()),
        "groups": group_summary(optimizer.param_groups),
        "non_ffn": digest({n: p for n, p in model.state_dict().items() if ".ffn." not in n}),
    }
    outputs = []
    handle = model.register_forward_hook(
        lambda module, args, output: outputs.append(output.detach())
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast("cuda", "bf16"):
        loss = model.loss(x, y)
    handle.remove()
    assert len(outputs) == 1
    loss.backward()
    initial.update(
        logits=tensor_record(outputs.pop()),
        loss=loss.item(),
        gradients={n: tensor_record(p.grad) for n, p in model.named_parameters()},
        layer_gradients=gradient_stats(model),
    )
    assert finite_tree([p.grad for p in model.parameters()])
    write_new(folder / "initial_signature.json", initial)
    optimizer.zero_grad(set_to_none=True)
    del loss, raw, x, y
    torch.cuda.empty_cache()
    history = []
    for step, batch in enumerate(stream):
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw = batch.cuda()
        x, y = raw[:, :-1], raw[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", "bf16"):
            loss = model.loss(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        row = {
            "step": step + 1,
            "loss": loss.item(),
            "norm_pre_clip": norm.item(),
            "seconds": duration,
        }
        assert finite_tree(row)
        history.append(row)
        with (folder / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if step == 9:
            torch.cuda.reset_peak_memory_stats()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    assert finite_tree(model.state_dict()) and finite_tree(optimizer.state_dict())
    assert (
        tensor_record(torch.get_rng_state()) == cpu_rng
        and tensor_record(torch.cuda.get_rng_state()) == cuda_rng
    )
    write_new(folder / "final_diagnostics.json", diagnostics(model, x))
    write_new(folder / "final_gradient_stats.json", gradient_stats(model))
    final = {"weights": digest(model.state_dict()), "optimizer": digest(optimizer.state_dict())}
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(mc),
            "training_config": asdict(tc),
            "step": 20,
            "synthetic": True,
            "mode": mode,
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        folder / "checkpoint.pt",
    )
    durations = [r["seconds"] for r in history[10:]]
    result = {
        "status": "PASS",
        "cell": cell,
        "form": form,
        "mode": mode,
        "ffn_parameters": actual_ffn,
        "total_parameters": actual_total,
        "ffn_projection_mac_per_token_per_layer": actual_ffn // 8,
        "ffn_reduction_percent": 100 * (1 - actual_ffn / 9437184),
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "median_step_ms": 1000 * statistics.median(durations),
        "step_seconds": durations,
        "training_tokens_per_second": 20480 / sum(durations),
        "clipped_step_fraction": sum(r["norm_pre_clip"] > 1 for r in history) / 20,
        "initial_weights_sha256": sha(folder / "initial.pt"),
        "initial_signature_sha256": sha(folder / "initial_signature.json"),
        "checkpoint_sha256": sha(folder / "checkpoint.pt"),
        "history_sha256": sha(folder / "history.jsonl"),
        "final": final,
        "token_record": tensor_record(stream),
        "cpu_rng": cpu_rng,
        "cuda_rng": cuda_rng,
        "rng_unchanged": True,
        "all_finite": True,
        "updates": 20,
        "synthetic_training_targets": 40960,
        "initial_probe_targets": 2048,
        "corpus_targets": 0,
    }
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    write_new(folder / "result.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "cell", "peak_allocated_bytes", "median_step_ms")}
        ),
        flush=True,
    )


def finish():
    protocol = read(ROOT / "protocol.json")
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    rows, comparisons = {}, {}
    for cell in CELLS:
        folder = ROOT / "workers" / cell
        row, event = read(folder / "result.json"), read(ROOT / "processes" / (cell + ".json"))
        assert (
            row["status"] == event["status"] == "PASS"
            and event["source_unchanged"]
            and event["returncode"] == 0
        )
        ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert ckpt["step"] == 20 and finite_tree(ckpt)
        assert {"weights": digest(ckpt["model"]), "optimizer": digest(ckpt["optimizer"])} == row[
            "final"
        ]
        assert sha(folder / "checkpoint.pt") == row["checkpoint_sha256"]
        rows[cell] = row
    for form in FORMS:
        base = ROOT / "workers" / f"{form}_none"
        baseline = rows[f"{form}_none"]
        for mode in MODES[1:]:
            folder = ROOT / "workers" / f"{form}_{mode}"
            row = rows[f"{form}_{mode}"]
            histories = [
                [
                    {k: r[k] for k in ("step", "loss", "norm_pre_clip")}
                    for r in map(json.loads, (p / "history.jsonl").read_text().splitlines())
                ]
                for p in (base, folder)
            ]
            comparisons[f"{form}_{mode}"] = {
                "initial_signature": read(base / "initial_signature.json")
                == read(folder / "initial_signature.json"),
                "all_losses_and_norms": histories[0] == histories[1],
                "final_weights_and_moments": baseline["final"] == row["final"],
                "token_rng": all(
                    baseline[k] == row[k] for k in ("token_record", "cpu_rng", "cuda_rng")
                ),
            }
    exact = all(all(r.values()) for r in comparisons.values())
    chosen, gates = derive_gates(rows, exact)
    result = {
        "status": "complete",
        "rows": rows,
        "comparisons": comparisons,
        "all_comparisons_exact": exact,
        "selected_modes": chosen,
        "gates": gates,
        "earns_language_screen": {f: all(g.values()) for f, g in gates.items()},
        "synthetic_updates": 420,
        "synthetic_training_targets": 860160,
        "initial_probe_targets": 43008,
        "corpus_targets": 0,
        "research_goal_achieved": False,
        "scientific_attempts": 1,
    }
    write_new(ROOT / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("rows", "comparisons")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    parser.add_argument("--cell", choices=list(CELLS))
    args = parser.parse_args()
    worker(args.cell) if args.command == "worker" else finish()
