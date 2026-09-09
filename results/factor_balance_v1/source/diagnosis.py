"""H069: read-only internal factor gauge diagnosis, no training or corpus scoring."""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from src.blockshuffle_ffn import BlockShuffleFFN, unshuffle_channels
from src.core.config import ModelConfig
from src.core.reproducibility import sha256, write_json
from src.core.structured_linear import shuffle_channels
from src.core.transformer import Transformer

ROOT = Path("results/factor_balance_v1")
PLAN = Path("research/factor_balance_plan.md")
SEEDS = (17, 29, 43)
LAYERS = (0, 3, 7)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_new(p, value):
    assert not p.exists(), p
    write_json(p, value)


def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((t.dtype, tuple(t.shape))).encode() + t.numpy().tobytes()).hexdigest()


def scales(a, b, s1, s2):
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and (a > 0).all() and (b > 0).all()
    exponent = torch.round(0.25 * torch.log2((b.square() / s2) / (a.square() / s1)))
    assert exponent.abs().max() <= 8
    c = 2.0**exponent
    original = a.square() / s1 + b.square() / s2
    optimal = 2 * a * b / math.sqrt(s1 * s2)
    after = a.square() * c.square() / s1 + b.square() / c.square() / s2
    ratio = (a * c / math.sqrt(s1)) / (b / c / math.sqrt(s2))
    assert (after <= original * (1 + 1e-12)).all()
    assert (after <= 1.25 * optimal * (1 + 1e-12)).all()
    assert (ratio >= 0.5 * (1 - 1e-12)).all() and (ratio <= 2 * (1 + 1e-12)).all()
    return c, exponent, original, optimal, after, ratio


@torch.no_grad()
def balance(projection):
    first, second = projection.first.weight, projection.second.weight
    g = projection.groups
    a = first.double().norm(dim=-1).flatten()
    b = unshuffle_channels(second.double().norm(dim=1).flatten(), g)
    s1 = projection.input_width / (2 * first.shape[-1])
    s2 = projection.input_width / (2 * second.shape[-1])
    c, e, original, optimal, after, ratio = scales(a, b, s1, s2)
    fa = c.to(first).reshape(g, -1, 1)
    fb = shuffle_channels(1 / c, g).to(second).reshape(g, 1, -1)
    first.mul_(fa)
    second.mul_(fb)
    actual_a = first.double().norm(dim=-1).flatten()
    actual_b = unshuffle_channels(second.double().norm(dim=1).flatten(), g)
    torch.testing.assert_close(actual_a, a * c, rtol=1e-12, atol=0)
    torch.testing.assert_close(actual_b, b / c, rtol=1e-12, atol=0)
    weighted_ratio = (a / math.sqrt(s1)) / (b / math.sqrt(s2))
    record = {
        "s1": s1,
        "s2": s2,
        "incoming_norms": a.tolist(),
        "outgoing_norms": b.tolist(),
        "unweighted_norm_ratio": (a / b).tolist(),
        "weighted_norm_ratio": weighted_ratio.tolist(),
        "exponents": e.int().tolist(),
        "energy_before": original.tolist(),
        "energy_optimal": optimal.tolist(),
        "energy_after": after.tolist(),
        "weighted_norm_ratio_after": ratio.tolist(),
        "channels_outside_fourfold": int(((weighted_ratio < 0.25) | (weighted_ratio > 4)).sum()),
        "sum_energy_before": original.sum().item(),
        "sum_energy_after": after.sum().item(),
        "max_energy_over_optimum": (after / optimal).max().item(),
        "all_bounds_pass": True,
    }
    return record, {"first.weight": fa, "second.weight": fb}


def initialized_ffn(seed, layer):
    ffn = BlockShuffleFFN(384, 2048, 8)
    for name in ("up", "gate", "down"):
        getattr(ffn, name).initialize(
            seed, f"blocks.{layer}.ffn.{name}", 0.25 if name == "down" else 1.0
        )
    ffn.recompute_gate = True
    ffn.gate_recompute_method = "native"
    return ffn


def value_and_grads(ffn, x, incoming, device):
    x = x.to(device).requires_grad_(True)
    ffn = ffn.to(device).train()
    names = [n for n, _ in ffn.named_parameters()]
    with torch.autocast(device, enabled=device == "cuda", dtype=torch.bfloat16):
        y = ffn(x)
    gradients = torch.autograd.grad(
        y, (x, *ffn.parameters()), incoming.to(device=device, dtype=y.dtype)
    )
    return {
        "output": y.detach().cpu(),
        "input_gradient": gradients[0].cpu(),
        **{n: g.cpu() for n, g in zip(names, gradients[1:])},
    }


def compare_case(reference, seed, step, layer, device):
    name = f"s{seed}_{step}_l{layer}_{device}"
    candidate = copy.deepcopy(reference)
    factors = {}
    for label in ("up", "gate", "down"):
        _, mapping = balance(getattr(candidate, label))
        factors.update({f"{label}.{n}": t for n, t in mapping.items()})
    shape = (16, 128, 384) if device == "cuda" else (2, 7, 384)
    generator = torch.Generator().manual_seed(91000 + 100 * seed + layer)
    x = torch.randn(shape, generator=generator)
    incoming = torch.randn(shape, generator=generator)
    baseline = value_and_grads(copy.deepcopy(reference), x.clone(), incoming, device)
    transformed = value_and_grads(candidate, x.clone(), incoming, device)
    mapped = {n: t * factors[n] if n in factors else t for n, t in transformed.items()}
    comparisons = {}
    for n, a in baseline.items():
        b = mapped[n]
        comparisons[n] = {
            "exact": torch.equal(a, b),
            "max_abs_error": (a - b).abs().max().item(),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "reference_sha256": tensor_hash(a),
            "mapped_sha256": tensor_hash(b),
        }
    raw = {
        "input": x,
        "incoming_gradient": incoming,
        "baseline": baseline,
        "transformed": transformed,
        "mapped": mapped,
        "factor_scales": factors,
    }
    path = ROOT / "checks" / (name + ".pt")
    path.parent.mkdir(exist_ok=True)
    assert not path.exists()
    torch.save(raw, path)
    record = {
        "seed": seed,
        "step": step,
        "layer": layer,
        "device": device,
        "input_sha256": tensor_hash(x),
        "comparisons": comparisons,
        "all_exact_finite": all(v["exact"] and v["finite"] for v in comparisons.values()),
        "artifact_sha256": sha256(path),
        "optimizer_updates": 0,
    }
    write_new(ROOT / "checks" / (name + ".json"), record)
    print(json.dumps({"case": name, "all_exact_finite": record["all_exact_finite"]}), flush=True)
    return record


def run():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    protocol = read(ROOT / "protocol.json")
    assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
    projections, checks, states = [], [], {}
    for seed in SEEDS:
        for step in (0, 800, 3200):
            if step:
                spec = protocol["checkpoints"][f"s{seed}_{step}"]
                folder = Path(spec["path"])
                assert sha256(folder / "checkpoint.pt") == spec["checkpoint_sha256"]
                assert sha256(folder / "metrics.json") == spec["metrics_sha256"]
                metrics = read(folder / "metrics.json")
                assert metrics["training"]["seed"] == seed and metrics["training"]["steps"] == step
                config = ModelConfig(**metrics["model"])
                model = Transformer(config, seed)
                stored = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(stored["model"], strict=True)
                del stored
                ffns = [b.ffn for b in model.blocks]
                for ffn in ffns:
                    ffn.recompute_gate = True
                    ffn.gate_recompute_method = "native"
            else:
                ffns = [initialized_ffn(seed, i) for i in range(8)]
            assert sum(p.numel() for ffn in ffns for p in ffn.parameters()) == 2801664
            rows = []
            for layer, ffn in enumerate(ffns):
                for label in ("up", "gate", "down"):
                    projection = copy.deepcopy(getattr(ffn, label))
                    row, _ = balance(projection)
                    row.update(seed=seed, step=step, layer=layer, projection=label)
                    rows.append(row)
                if layer in LAYERS and (step == 3200 or (step == 0 and seed == 17)):
                    for device in ("cpu", "cuda"):
                        checks.append(compare_case(ffn, seed, step, layer, device))
            summary = {
                "energy_before": sum(v["sum_energy_before"] for v in rows),
                "energy_after": sum(v["sum_energy_after"] for v in rows),
                "channels_outside_fourfold": sum(v["channels_outside_fourfold"] for v in rows),
                "channels": 9216,
                "projection_records": 24,
            }
            summary["energy_reduction_fraction"] = (
                1 - summary["energy_after"] / summary["energy_before"]
            )
            summary["outside_fourfold_fraction"] = summary["channels_outside_fourfold"] / 9216
            states[f"s{seed}_{step}"] = summary
            projections.extend(rows)
            write_new(
                ROOT / "states" / f"s{seed}_{step}.json", {"summary": summary, "projections": rows}
            )
            print(json.dumps({"state": f"s{seed}_{step}", **summary}), flush=True)
    assert len(projections) == 216 and len(checks) == 24
    numerical = all(c["all_exact_finite"] for c in checks)
    energy = all(states[f"s{s}_3200"]["energy_reduction_fraction"] >= 0.2 for s in SEEDS)
    imbalance = all(states[f"s{s}_3200"]["outside_fourfold_fraction"] >= 0.1 for s in SEEDS)
    earns = numerical and energy and imbalance
    result = {
        "status": "complete",
        "states": states,
        "checks": checks,
        "projection_records": 216,
        "all_local_exact": numerical,
        "every_seed_energy_gate": energy,
        "every_seed_imbalance_gate": imbalance,
        "earns_optimizer_state_update_qualification": earns,
        "scientific_verdict": "EARNS_STATE_QUALIFICATION"
        if earns
        else "REJECTED_AS_OPTIMIZER_MOTIVATION",
        "optimizer_updates": 0,
        "corpus_targets": 0,
        "new_active_variants": 0,
        "research_goal_achieved": False,
    }
    assert all(sha256(Path(n)) == h for n, h in protocol["sources"].items())
    for spec in protocol["checkpoints"].values():
        assert sha256(Path(spec["path"]) / "checkpoint.pt") == spec["checkpoint_sha256"]
    write_new(ROOT / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("states", "checks")}), flush=True
    )


if __name__ == "__main__":
    run()
