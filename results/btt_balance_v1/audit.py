"""CPU FP64 dense-matrix evaluation independent of the trained BTT contractions."""

import json
import math
import statistics as st
from pathlib import Path

import torch
from torch.nn import functional as F

from results.btt_balance_v1.model import BTT, Model
from results.btt_balance_v1.study import ROOT, SEEDS, TASKS, data, read, sha, verify, write


@torch.no_grad()
def dense_score(row, dataset, start, end):
    model = Model(row["arm"], row["seed"]).double()
    model.load_state_dict(torch.load(row["selected_path"], weights_only=True))
    matrices = []
    for module, din in ((model.up, 64), (model.down, model.bias_up.numel())):
        if isinstance(module, BTT):
            right, left = module.cores()
            # Explicit four-index weight construction, not the two training GEMMs.
            matrix = (
                (right.permute(0, 2, 1)[:, :, :, None] * left.permute(1, 0, 2)[:, :, None, :])
                .permute(0, 2, 1, 3)
                .reshape(din, -1)
            )
        elif isinstance(module, torch.nn.Linear):
            matrix = module.weight.T
        else:
            # The retained grouped/permutation map is linear; recover each basis image.
            matrix = module(torch.eye(din, dtype=torch.float64))
        matrices.append(matrix)
    x, y = dataset["x"][start:end].double(), dataset["y"][start:end].double()
    predicted = F.gelu(x @ matrices[0] + model.bias_up) @ matrices[1] + model.bias_down
    return (predicted - y).square().mean().item()


def main():
    verify()
    result = read(ROOT / "result.json")
    assert len(result["runs"]) == 84 and len(result["selected"]) == len(result["profiles"]) == 42
    datasets = {}
    for task in TASKS:
        for seed in SEEDS:
            actual = torch.load(ROOT / f"data_{task}_{seed}.pt", weights_only=True)
            regenerated = data(task, seed)
            assert all(torch.equal(actual[k], regenerated[k]) for k in actual)
            datasets[task, seed] = actual
    scores = []
    for row in result["runs"]:
        assert len(row["trace"]) == 600
        assert all(math.isfinite(r[k]) for r in row["trace"] for k in ("loss", "norm"))
        assert sha(row["selected_path"]) == row["selected_sha"]
        assert sha(Path(row["selected_path"]).parent / "final.pt") == row["final_sha"]
        endpoint = min(row["endpoints"], key=lambda v: v["validation"])
        assert endpoint["step"] == row["step"] and endpoint["validation"] == row["validation"]
        dataset = datasets[row["task"], row["seed"]]
        validation = dense_score(row, dataset, 4096, 5120)
        report = dense_score(row, dataset, 5120, 6144)
        assert abs(validation / row["validation"] - 1) <= 1e-5
        assert abs(report / row["report"] - 1) <= 1e-5
        scores.append(
            dict(
                task=row["task"],
                seed=row["seed"],
                arm=row["arm"],
                lr=row["lr"],
                validation=validation,
                report=report,
            )
        )
    for chosen in result["selected"]:
        candidates = [
            r
            for r in result["runs"]
            if (r["task"], r["seed"], r["arm"]) == (chosen["task"], chosen["seed"], chosen["arm"])
        ]
        assert chosen == min(candidates, key=lambda r: r["validation"])
    qualified, decisions, shape, stats = {}, {}, {}, {}
    for task in TASKS:
        taskrows = [r for r in result["selected"] if r["task"] == task]
        qualified[task] = all(
            r["report"] <= (0.25 if task == "teacher" else 0.5)
            for r in taskrows
            if r["arm"] == "wide"
        )
        pairs = []
        for seed in SEEDS:
            rows = {r["arm"]: r for r in taskrows if r["seed"] == seed}
            profiles = {
                r["arm"]: r for r in result["profiles"] if r["task"] == task and r["seed"] == seed
            }
            b, w, n = rows["balanced"], rows["wide"], rows["narrow33"]
            bp, wp = profiles["balanced"], profiles["wide"]
            ratios = dict(
                wide_mse=b["report"] / w["report"],
                narrow_mse=b["report"] / n["report"],
                greedy_mse=b["report"] / rows["greedy"]["report"],
                parameters=b["parameters"] / w["parameters"],
                memory=bp["memory"]["allocated"] / wp["memory"]["allocated"],
                cuda=bp["median_ms"] / wp["median_ms"],
                wall=bp["median_wall_ms"] / wp["median_wall_ms"],
            )
            gates = dict(
                task=qualified[task],
                wide_quality=ratios["wide_mse"] <= 1.05,
                narrow_quality=ratios["narrow_mse"] <= 0.95,
                parameters=ratios["parameters"] <= 0.75,
                memory=ratios["memory"] <= 0.90,
                cuda=ratios["cuda"] <= 1.15,
                wall=ratios["wall"] <= 1.15,
            )
            pairs.append(dict(seed=seed, ratios=ratios, gates=gates, passed=all(gates.values())))
        decisions[task] = pairs
        balanced = [r["report"] for r in taskrows if r["arm"] == "balanced"]
        greedy = [r["report"] for r in taskrows if r["arm"] == "greedy"]
        ratio = st.mean(balanced) / st.mean(greedy)
        shape[task] = dict(
            mean_ratio=ratio,
            passed=qualified[task]
            and ratio <= 0.95
            and all(p["ratios"]["greedy_mse"] <= 1.05 for p in pairs),
        )
        stats[task] = {}
        for arm in {r["arm"] for r in taskrows}:
            values = [r["report"] for r in taskrows if r["arm"] == arm]
            stats[task][arm] = dict(
                mean=st.mean(values), median=st.median(values), variance=st.variance(values)
            )
    write(
        ROOT / "audit.json",
        dict(
            passed=True,
            independent_scores=168,
            datasets_regenerated=6,
            scores=scores,
            optimizer_updates=0,
            gpu_work=False,
        ),
    )
    write(
        ROOT / "summary.json",
        dict(
            study="H164",
            qualified_tasks=qualified,
            decisions=decisions,
            shape_hypothesis=shape,
            statistics=stats,
            shape_passed=all(v["passed"] for v in shape.values()),
            promotion=all(p["passed"] for rows in decisions.values() for p in rows),
            broad_goal_achieved=False,
        ),
    )
    print(json.dumps(dict(qualified=qualified, shape=shape, decisions=decisions)))


if __name__ == "__main__":
    main()
