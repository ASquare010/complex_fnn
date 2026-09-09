"""Recover only the four missing H074 cells using its unchanged worker and gates."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from results.ungated_resource_v1.source import study as original
from src.core.native_recompute_audit import digest, finite_tree, read, sha, write_new

ROOT = Path("results/ungated_resource_recovery_v1")
PLAN = Path("research/ungated_resource_recovery_plan.md")
RECOVERY_CELLS = (
    "gelu_same_block_inner",
    "gelu_matched_none",
    "gelu_matched_block",
    "gelu_matched_block_inner",
)


def verify():
    p = read(ROOT / "protocol.json")
    assert sha(PLAN) == p["recovery_plan_sha256"]
    assert sha(original.PLAN) == p["plan_sha256"]
    assert all(sha(n) == h for n, h in p["sources"].items())
    for probe in ("cpu_small", "full_1", "full_2"):
        assert read(ROOT / "probes" / (probe + ".json"))["status"] == "PASS"
        assert read(ROOT / "processes" / (probe + ".json"))["status"] == "PASS"
    for cell, files in p["preserved_h074_worker_hashes"].items():
        assert all(
            sha(Path("results/ungated_resource_v1/workers") / cell / n) == h
            for n, h in files.items()
        )
    return p


def worker(cell):
    assert cell in RECOVERY_CELLS
    verify()
    with patch.object(original, "ROOT", ROOT):
        original.worker(cell)


def finish():
    protocol = verify()
    rows, comparisons, origin_records = {}, {}, {}
    for cell, spec in original.CELLS.items():
        origin = Path(protocol["origins"][cell])
        folder = origin / "workers" / cell
        row = read(folder / "result.json")
        event = read(origin / "processes" / (cell + ".json"))
        assert (
            row["status"] == event["status"] == "PASS"
            and event["returncode"] == 0
            and event["source_unchanged"]
        )
        assert sha(folder / "checkpoint.pt") == row["checkpoint_sha256"]
        ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert ckpt["step"] == 20 and ckpt["synthetic"] and finite_tree(ckpt)
        assert {"weights": digest(ckpt["model"]), "optimizer": digest(ckpt["optimizer"])} == row[
            "final"
        ]
        rows[cell] = row
        origin_records[cell] = {
            "root": origin.as_posix(),
            "process_sha256": sha(origin / "processes" / (cell + ".json")),
            "result_sha256": sha(folder / "result.json"),
            "checkpoint_sha256": sha(folder / "checkpoint.pt"),
        }
    for form in original.FORMS:
        baseline = rows[f"{form}_none"]
        base = Path(protocol["origins"][f"{form}_none"]) / "workers" / f"{form}_none"
        for mode in original.MODES[1:]:
            cell = f"{form}_{mode}"
            row = rows[cell]
            folder = Path(protocol["origins"][cell]) / "workers" / cell
            histories = [
                [
                    {k: r[k] for k in ("step", "loss", "norm_pre_clip")}
                    for r in map(json.loads, (p / "history.jsonl").read_text().splitlines())
                ]
                for p in (base, folder)
            ]
            comparisons[cell] = {
                "initial_signature": read(base / "initial_signature.json")
                == read(folder / "initial_signature.json"),
                "all_losses_and_norms": histories[0] == histories[1],
                "final_weights_and_moments": baseline["final"] == row["final"],
                "token_rng": all(
                    baseline[k] == row[k] for k in ("token_record", "cpu_rng", "cuda_rng")
                ),
            }
    exact = all(all(v.values()) for v in comparisons.values())
    selected, gates = original.derive_gates(rows, exact)
    result = {
        "status": "complete",
        "recovery_completed": True,
        "rows": rows,
        "origins": origin_records,
        "comparisons": comparisons,
        "all_comparisons_exact": exact,
        "selected_modes": selected,
        "gates": gates,
        "earns_language_screen": {f: all(g.values()) for f, g in gates.items()},
        "new_optimizer_updates": 80,
        "combined_optimizer_updates": 420,
        "new_synthetic_training_targets": 163840,
        "combined_synthetic_training_targets": 860160,
        "new_initial_probe_targets": 8192,
        "combined_initial_probe_targets": 43008,
        "construction_probes": 3,
        "construction_probe_updates": 0,
        "corpus_targets": 0,
        "original_worker_launches": 18,
        "recovery_worker_launches": 4,
        "combined_worker_launches": 22,
        "completed_cell_reruns": 0,
        "explicit_preupdate_failed_cell_retries": 1,
        "research_goal_achieved": False,
        "failure_cause_identified": False,
    }
    verify()
    write_new(ROOT / "result.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "origins", "comparisons")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    parser.add_argument("--cell", choices=RECOVERY_CELLS)
    args = parser.parse_args()
    worker(args.cell) if args.command == "worker" else finish()
