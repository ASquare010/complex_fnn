"""Post-hoc zero-predictor and generalization-gap diagnostic; no changed gates."""

import json
import statistics
from pathlib import Path

import torch

from src.core.reproducibility import sha256, write_json

ROOT = Path("results/rotated_shuffle_fit_v1")
OUT = Path("results/verification/rotated_shuffle_fit_baseline_v1.json")
assert not OUT.exists()
torch.set_num_threads(4)
r = json.loads((ROOT / "result.json").read_text(encoding="utf-8"))
data = torch.load(ROOT / "data.pt", map_location="cpu", weights_only=True)
rows = {}
for task, y in data["targets"].items():
    zero = (y[5120:].double().square().mean()).item()
    train_zero = (y[:4096].double().square().mean()).item()
    forms = {}
    for form in r["task_mean_mse"][task]:
        selected = [s for s in r["selected_rows"] if s["task"] == task and s["form"] == form]
        train = []
        for selected_row in selected:
            record = json.loads(
                (ROOT / "cells" / selected_row["cell"] / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            train.append(record["train_mse"])
        heldout = r["task_mean_mse"][task][form]
        forms[form] = {
            "mean_train_mse": statistics.mean(train),
            "mean_heldout_mse": heldout,
            "heldout_over_zero": heldout / zero,
            "train_over_train_zero": statistics.mean(train) / train_zero,
            "mean_heldout_minus_train": heldout - statistics.mean(train),
            "selected_seeds_beating_zero": sum(s["heldout_mse"] < zero for s in selected),
        }
    rows[task] = {"heldout_zero_mse": zero, "train_zero_mse": train_zero, "forms": forms}
record = {
    "status": "PASS",
    "post_hoc_diagnostic": True,
    "decision_gates_changed": False,
    "optimizer_updates": 0,
    "corpus_targets": 0,
    "data_sha256": sha256(ROOT / "data.pt"),
    "result_sha256": sha256(ROOT / "result.json"),
    "rows": rows,
}
write_json(OUT, record)
print(
    json.dumps(
        {
            "status": "PASS",
            "zero_baselines": {t: v["heldout_zero_mse"] for t, v in rows.items()},
            "plain_heldout_over_zero": {
                t: v["forms"]["plain"]["heldout_over_zero"] for t, v in rows.items()
            },
        }
    )
)
