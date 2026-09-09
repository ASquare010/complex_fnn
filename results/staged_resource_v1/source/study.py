"""H066 adapters and analysis, reusing H063's unchanged worker loop."""

import argparse
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.rational_staged_v1.source.candidate import StagedRationalActivation
from src.core import native_recompute_audit as common
from src.dense_ffn import DenseFFN
from src.rational_blockshuffle_ffn import RationalResidualActivation

ROOT = Path("results/staged_resource_v1")
PLAN = Path("research/staged_resource_plan.md")
PLAN_SHA = "2a5addfe6acc557559cca038b9e9151cd481609d76ab487c9d480c128c3e38f4"
CELLS = {
    "full_swiglu_native": {"recipe": "full_swiglu", "scope": "block", "mode": "native"},
    "full_swiglu_inner": {"recipe": "full_swiglu", "scope": "block", "mode": "inner"},
    "full_gelu_native": {"recipe": "full_gelu", "scope": "block", "mode": "native"},
    "full_gelu_inner": {"recipe": "full_gelu", "scope": "block", "mode": "inner"},
    "rational_native": {"recipe": "rational", "scope": "block", "mode": "native"},
    "rational_staged": {"recipe": "rational", "scope": "block", "mode": "staged"},
}


def dense_product(u, v):
    return F.silu(u) * v


def dense_inner_forward(self, x):
    if not self.training or not torch.is_grad_enabled():
        return DenseFFN.forward(self, x)
    u = self.up(x)
    if self.gate is not None:
        v = self.gate(x)
        z = checkpoint(dense_product, u, v, use_reentrant=False, preserve_rng_state=False)
    else:
        assert self.activation == "gelu"
        z = checkpoint(F.gelu, u, use_reentrant=False, preserve_rng_state=False)
    return self.down(z)


def worker(cell):
    spec = CELLS[cell]
    original_loader = common.load_model

    def load_model(recipe, scope, protocol):
        loaded = original_loader(recipe, scope, protocol)
        model, optimizer = loaded[:2]
        before = common.digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        identities = {n: id(p) for n, p in model.named_parameters()}
        optimizer_ids = [[id(p) for p in group["params"]] for group in optimizer.param_groups]
        for block in model.blocks:
            if spec["mode"] == "staged":
                assert type(block.ffn.curve) is RationalResidualActivation
                block.ffn.curve.residual = MethodType(
                    StagedRationalActivation.residual, block.ffn.curve
                )
            elif spec["mode"] == "inner":
                assert type(block.ffn) is DenseFFN
                block.ffn.forward = MethodType(dense_inner_forward, block.ffn)
        assert identities == {n: id(p) for n, p in model.named_parameters()}
        assert optimizer_ids == [
            [id(p) for p in group["params"]] for group in optimizer.param_groups
        ]
        assert (
            common.digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            == before
        )
        common.write_new(
            ROOT / "workers" / cell / "adapter.json",
            {
                "mode": spec["mode"],
                "parameter_objects_preserved": True,
                "optimizer_references_preserved": True,
                "weights_and_moments_preserved": True,
                "scope": scope,
            },
        )
        return loaded

    with (
        patch.object(common, "ROOT", ROOT),
        patch.object(common, "PLAN", PLAN),
        patch.object(common, "load_model", load_model),
    ):
        common.worker(cell)


def finish():
    protocol = common.read(ROOT / "protocol.json")
    assert common.sha(PLAN) == protocol["plan_sha256"] == PLAN_SHA
    assert all(common.sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    rows, comparisons = {}, {}
    for cell in CELLS:
        folder = ROOT / "workers" / cell
        row = common.read(folder / "result.json")
        process = common.read(ROOT / "processes" / f"{cell}.json")
        assert (
            row["status"] == process["status"] == "PASS"
            and process["returncode"] == 0
            and process["source_unchanged"]
        )
        assert row["updates"] == 20 and row["all_finite"] and row["rng_unchanged"]
        assert common.sha(folder / "checkpoint.pt") == row["checkpoint_sha256"]
        assert common.sha(folder / "history.jsonl") == row["history_sha256"]
        assert common.sha(folder / "initial_signature.json") == row["initial_signature_sha256"]
        ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert ckpt["step"] == 220 and ckpt["synthetic"] and common.finite_tree(ckpt)
        assert common.digest(ckpt["model"]) == row["final"]["weights"]
        assert common.digest(ckpt["optimizer"]) == row["final"]["optimizer"]
        rows[cell] = row
        del ckpt
    for recipe, treatment in (
        ("full_swiglu", "inner"),
        ("full_gelu", "inner"),
        ("rational", "staged"),
    ):
        native, changed = rows[f"{recipe}_native"], rows[f"{recipe}_{treatment}"]
        comparisons[recipe] = {
            "initial_signature": common.read(
                ROOT / "workers" / f"{recipe}_native" / "initial_signature.json"
            )
            == common.read(ROOT / "workers" / f"{recipe}_{treatment}" / "initial_signature.json"),
            "all_losses_and_norms": native["history_sha256"] == changed["history_sha256"],
            "final_weights": native["final"]["weights"] == changed["final"]["weights"],
            "final_optimizer": native["final"]["optimizer"] == changed["final"]["optimizer"],
            "token_rng": all(
                native[k] == changed[k] for k in ("token_record", "cpu_rng", "cuda_rng")
            ),
        }
    candidate, native = rows["rational_staged"], rows["rational_native"]
    limits = {
        recipe: 1.1
        * min(rows[f"{recipe}_{mode}"]["peak_allocated_bytes"] for mode in ("native", "inner"))
        for recipe in ("full_swiglu", "full_gelu")
    }
    gates = {
        "all_full_model_numerical_comparisons_exact": all(
            all(c.values()) for c in comparisons.values()
        ),
        "at_least_70_percent_fewer_ffn_weights": candidate["ffn_reduction_percent"] >= 70,
        "at_least_ten_percent_native_memory_reduction": candidate["peak_allocated_bytes"]
        <= 0.9 * native["peak_allocated_bytes"],
        **{
            f"memory_within_ten_percent_best_{n}": candidate["peak_allocated_bytes"] <= limit
            for n, limit in limits.items()
        },
        "additional_update_time_at_most_25_percent": candidate["median_step_ms"]
        <= 1.25 * native["median_step_ms"],
    }
    for row in protocol["inputs"].values():
        assert all(
            common.sha(Path("results/runs") / row["run"] / n) == h for n, h in row["files"].items()
        )
    result = {
        "status": "complete",
        "verdict": "QUALIFIED_RESOURCE_AND_FIDELITY" if all(gates.values()) else "REJECTED",
        "results": rows,
        "comparisons": comparisons,
        "gates": gates,
        "dense_memory_limits_bytes": limits,
        "plan_sha256": PLAN_SHA,
        "earns_native_language_repeat": all(gates.values()),
        "synthetic_updates": 120,
        "synthetic_training_targets": 245760,
        "initial_probe_targets": 12288,
        "corpus_scoring": False,
        "research_goal_achieved": False,
    }
    common.write_new(ROOT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    parser.add_argument("--cell", choices=list(CELLS))
    args = parser.parse_args()
    worker(args.cell) if args.command == "worker" else finish()
