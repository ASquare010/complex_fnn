"""H064: zero-update allocator attribution with exact untraced controls."""

import argparse
import ast
import json
import pickle
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.allocation_trace import replay_allocations, summarize_peak
from src.core.benchmark import autocast
from src.core.native_recompute_audit import (
    digest,
    finite_tree,
    load_model,
    read,
    sha,
    tensor_record,
    tokens,
    write_new,
)
from src.core.reproducibility import environment, provenance

ROOT = Path("results/rational_memory_v1")
PLAN = Path("research/rational_memory_plan.md")
PLAN_SHA = "2870fab288f10160e887b9efeb095528199542c012abdba89b9f3eb93ecd3c73"
CELLS = ("plain_untraced", "plain_traced", "rational_untraced", "rational_traced")
CAP = 50000


def regions():
    definitions = {
        "src/rational_blockshuffle_ffn/__init__.py": {
            "LearnableActivation.forward": "rational_pointwise",
            "RationalResidualActivation.residual": "rational_pointwise",
            "RationalBlockShuffleFFN.product": "rational_pointwise",
        },
        "src/blockshuffle_ffn/__init__.py": {
            "BlockShuffleLinear.forward": "projection",
            "swiglu_product": "plain_gate",
            "RecomputedSwiGLU.forward": "plain_gate",
            "RecomputedSwiGLU.backward": "plain_gate",
        },
        "src/core/structured_linear.py": {"GroupedLinear.forward": "projection"},
        "src/core/transformer.py": {
            "RMSNorm.forward": "norm",
            "Attention.forward": "attention",
            "Attention.rotate": "attention",
        },
    }
    result = []
    for path, selected in definitions.items():
        tree = ast.parse(Path(path).read_bytes())
        for parent in tree.body:
            pairs = (
                [
                    (f"{parent.name}.{n.name}", n)
                    for n in parent.body
                    if isinstance(n, ast.FunctionDef)
                ]
                if isinstance(parent, ast.ClassDef)
                else [(parent.name, parent)]
                if isinstance(parent, ast.FunctionDef)
                else []
            )
            for name, node in pairs:
                if name in selected:
                    result.append(
                        {
                            "path": path,
                            "function": name,
                            "start": node.lineno,
                            "end": node.end_lineno,
                            "category": selected[name],
                        }
                    )
    return result


def preflight():
    assert sha(PLAN) == PLAN_SHA and not (ROOT / "protocol.json").exists()
    before = read(ROOT / "before.json")
    assert sha("results/verification/native_recompute_final_v1.json") == before["h063_final_sha256"]
    assert sha("results/native_recompute_v1/protocol.json") == before["h063_protocol_sha256"]
    assert all(sha(n) == h for n, h in before["h063_source_hashes"].items())
    suite = read(ROOT / "processes/full_tests_after_compiler_failure.json")
    assert suite["status"] == "PASS" and suite["source_unchanged"] and suite["returncode"] == 0
    assert all(sha(n) == h for n, h in suite["sources"].items())
    for row in before["inputs"].values():
        assert all(sha(Path("results/runs") / row["run"] / n) == h for n, h in row["files"].items())
    prov = provenance()
    protocol = {
        "plan_sha256": PLAN_SHA,
        "provenance": prov,
        "inputs": before["inputs"],
        "environment": environment(),
        "regions": regions(),
        "cells": list(CELLS),
        "token_record": tensor_record(tokens()[0]),
        "scope": "block",
        "updates": 0,
        "forward_backward_evaluations": 8,
        "synthetic_target_exposures": 16384,
        "corpus_scoring": False,
        "trace_cap": CAP,
        "allocator_source_audit_sha256": sha(ROOT / "allocator_source_audit.json"),
    }
    write_new(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    write_new(
        ROOT / "preflight.json",
        {
            "status": "PASS",
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "all_h063_sources_unchanged": True,
        },
    )
    print(json.dumps({"status": "PASS", "workers": 4, "optimizer_updates": 0}), flush=True)


def counters():
    stats = torch.cuda.memory_stats()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "requested": stats["requested_bytes.all.current"],
        "peak_requested": stats["requested_bytes.all.peak"],
    }


def known_allocations(model, optimizer, raw):
    known = {}

    def add(value, category, name):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            known[str(value.untyped_storage().data_ptr())] = {
                "category": category,
                "name": name,
                "tensor_storage_bytes": value.untyped_storage().nbytes(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }

    for n, p in model.named_parameters():
        add(p, "model_parameters", n)
    for n, p in model.named_buffers():
        add(p, "model_buffers", n)
    for i, state in enumerate(optimizer.state.values()):
        for n, p in state.items():
            add(p, "optimizer_moments", f"optimizer.{i}.{n}")
    add(raw, "synthetic_input", "token batch")
    return known


def worker(cell):
    assert cell in CELLS
    protocol = read(ROOT / "protocol.json")
    pre = read(ROOT / "preflight.json")
    assert pre["status"] == "PASS" and sha(ROOT / "protocol.json") == pre["protocol_sha256"]
    assert sha(PLAN) == protocol["plan_sha256"] and environment() == protocol["environment"]
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    directory = ROOT / "workers" / cell
    directory.mkdir(parents=True, exist_ok=False)
    recipe = "rational" if cell.startswith("rational") else "blockshuffle"
    traced = cell.endswith("_traced")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, optimizer, mc, tc, _ = load_model(recipe, "block", protocol)
    raw = tokens()[0].cuda()
    x, y = raw[:, :-1], raw[:, 1:]
    assert tensor_record(raw) == protocol["token_record"]
    state_before = {
        "model": digest(model.state_dict()),
        "optimizer": digest(optimizer.state_dict()),
    }
    rng_before = {
        "cpu": tensor_record(torch.get_rng_state()),
        "cuda": tensor_record(torch.cuda.get_rng_state()),
    }
    with autocast("cuda", "bf16"):
        warmup = model.loss(x, y)
    warmup.backward()
    assert finite_tree([p.grad for p in model.parameters()])
    optimizer.zero_grad(set_to_none=True)
    del warmup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    write_new(
        directory / "config.json",
        {
            "cell": cell,
            "recipe": recipe,
            "traced": traced,
            "model": asdict(mc),
            "training": asdict(tc),
            "scope": "block",
            "optimizer_updates": 0,
            "synthetic": True,
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "input": protocol["inputs"][recipe],
            "environment": environment(),
        },
    )
    write_new(directory / "known_allocations.json", known_allocations(model, optimizer, raw))
    logits = []
    # Copy/hash the output at its existing lifetime; retain only CPU metadata during backward.
    handle = model.register_forward_hook(
        lambda module, args, output: logits.append(tensor_record(output))
    )
    snapshots = {}
    torch.cuda.reset_peak_memory_stats()
    if traced:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="alloc", stacks="all", max_entries=CAP, clear_history=True
        )
    marks = {"baseline": counters()}
    if traced:
        snapshots["baseline"] = torch.cuda.memory._snapshot()
    with autocast("cuda", "bf16"):
        loss = model.loss(x, y)
    handle.remove()
    assert len(logits) == 1
    torch.cuda.synchronize()
    marks["forward_end"] = counters()
    if traced:
        snapshots["forward_end"] = torch.cuda.memory._snapshot()
    loss.backward()
    torch.cuda.synchronize()
    marks["backward_end"] = counters()
    if traced:
        snapshots["backward_end"] = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
    signature = {
        "loss": loss.item(),
        "logits": logits[0],
        "gradients": {n: tensor_record(p.grad) for n, p in model.named_parameters()},
        "weights": digest(model.state_dict()),
        "optimizer": digest(optimizer.state_dict()),
    }
    assert finite_tree([p.grad for p in model.parameters()])
    assert {"model": signature["weights"], "optimizer": signature["optimizer"]} == state_before
    assert {
        "cpu": tensor_record(torch.get_rng_state()),
        "cuda": tensor_record(torch.cuda.get_rng_state()),
    } == rng_before
    write_new(directory / "signature.json", signature)
    snapshot_hashes = {}
    for name, snapshot in snapshots.items():
        path = directory / f"{name}.pickle"
        with path.open("xb") as handle:
            pickle.dump(snapshot, handle, protocol=5)
        snapshot_hashes[path.name] = sha(path)
    result = {
        "status": "PASS",
        "cell": cell,
        "recipe": recipe,
        "traced": traced,
        "marks": marks,
        "snapshot_hashes": snapshot_hashes,
        "signature_sha256": sha(directory / "signature.json"),
        "source_unchanged": True,
        "rng_unchanged": True,
        "weights_moments_unchanged": True,
        "all_gradients_finite": True,
        "optimizer_updates": 0,
        "synthetic_target_exposures": 4096,
        "corpus_scoring": False,
        "token_record": tensor_record(raw),
        "rng": rng_before,
    }
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    write_new(directory / "result.json", result)
    print(json.dumps({"status": "PASS", "cell": cell, "marks": marks}), flush=True)


def load_snapshot(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def finish():
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == PLAN_SHA and all(
        sha(n) == h for n, h in protocol["provenance"]["source_files"].items()
    )
    comparisons = {}
    analyses = {}
    rows = {}
    for cell in CELLS:
        directory = ROOT / "workers" / cell
        row = read(directory / "result.json")
        process = read(ROOT / "processes" / f"worker_{cell}.json")
        assert (
            row["status"] == process["status"] == "PASS"
            and process["returncode"] == 0
            and process["source_unchanged"]
        )
        assert (
            row["optimizer_updates"] == 0
            and row["weights_moments_unchanged"]
            and row["rng_unchanged"]
        )
        assert sha(directory / "signature.json") == row["signature_sha256"]
        assert all(sha(directory / n) == h for n, h in row["snapshot_hashes"].items())
        rows[cell] = row
    for recipe in ("plain", "rational"):
        plain, traced = rows[f"{recipe}_untraced"], rows[f"{recipe}_traced"]
        equality = {
            "signature_exact": plain["signature_sha256"] == traced["signature_sha256"],
            "peak_allocated_exact": plain["marks"]["backward_end"]["peak_allocated"]
            == traced["marks"]["backward_end"]["peak_allocated"],
            "phase_allocations_exact": all(
                plain["marks"][p]["allocated"] == traced["marks"][p]["allocated"]
                for p in plain["marks"]
            ),
            "token_rng_exact": all(plain[k] == traced[k] for k in ("token_record", "rng")),
        }
        comparisons[recipe] = equality
        directory = ROOT / "workers" / f"{recipe}_traced"
        baseline, middle, final = [
            load_snapshot(directory / f"{name}.pickle")
            for name in ("baseline", "forward_end", "backward_end")
        ]
        boundary = len(middle["device_traces"][0])
        replay = replay_allocations(baseline, final, forward_end=boundary, cap=CAP)
        phase_replay = replay_allocations(baseline, middle, forward_end=boundary, cap=CAP)
        accounting = {
            "baseline_exact": replay["initial_bytes"] == traced["marks"]["baseline"]["allocated"],
            "forward_end_exact": phase_replay["final_bytes"]
            == traced["marks"]["forward_end"]["allocated"],
            "forward_peak_exact": phase_replay["peak_bytes"]
            == traced["marks"]["forward_end"]["peak_allocated"],
            "backward_end_exact": replay["final_bytes"]
            == traced["marks"]["backward_end"]["allocated"],
            "global_peak_exact": replay["peak_bytes"]
            == traced["marks"]["backward_end"]["peak_allocated"],
        }
        attribution = summarize_peak(
            replay, read(directory / "known_allocations.json"), protocol["regions"]
        )
        write_new(directory / "replay.json", replay)
        write_new(directory / "peak_attribution.json", attribution)
        analyses[recipe] = {
            "accounting": accounting,
            "attribution_valid": all(equality.values()) and all(accounting.values()),
            "peak_bytes": replay["peak_bytes"],
            "peak_phase": replay["peak_phase"],
            "peak_event": replay["peak_event"],
            "events": replay["events"],
            "category_bytes": attribution["category_bytes"],
            "site_bytes": attribution["site_bytes"],
            "requested_bytes_at_allocated_peak": attribution["requested_bytes_at_allocated_peak"],
        }
    for row in protocol["inputs"].values():
        assert all(sha(Path("results/runs") / row["run"] / n) == h for n, h in row["files"].items())
    result = {
        "status": "complete",
        "comparisons": comparisons,
        "analyses": analyses,
        "attribution_valid": all(row["attribution_valid"] for row in analyses.values()),
        "optimizer_updates": 0,
        "synthetic_target_exposures": 16384,
        "corpus_scoring": False,
        "plan_sha256": PLAN_SHA,
        "research_goal_achieved": False,
        "repair_implementation_earned": False,
    }
    write_new(ROOT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "worker", "finish"))
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.cell)
    else:
        {"preflight": preflight, "finish": finish}[args.command]()
