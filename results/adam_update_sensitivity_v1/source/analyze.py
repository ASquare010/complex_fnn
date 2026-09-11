"""Predeclared CPU FP64 direction analysis; no new optimizer or model work."""

import itertools
import math

import torch

from results.adam_update_sensitivity_v1.source.prepare import ROOT, hashes, read
from src.core.reproducibility import write_json

torch.set_num_threads(4)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def distance(a, b):
    error = left = right = 0.0
    for key in a:
        x, y = a[key].double(), b[key].double()
        error += (x - y).square().sum().item()
        left += x.square().sum().item()
        right += y.square().sum().item()
    return math.sqrt(error) / max((math.sqrt(left) + math.sqrt(right)) / 2, 1e-12)


def direction(a, epsilon):
    return {k: v.double() / (v.double().abs() + epsilon) for k, v in a.items()}


def pair_metrics(a, b, epsilon):
    counts, errors, opposite = [0] * 4, [0.0] * 4, 0
    da, db = direction(a, epsilon), direction(b, epsilon)
    for key in a:
        x, y = a[key].double(), b[key].double()
        level = torch.maximum(x.abs(), y.abs()) / epsilon
        index = (level > 1).int() + (level > 10).int() + (level > 100).int()
        err = (da[key] - db[key]).square()
        opposite += ((x * y) < 0).sum().item()
        for i in range(4):
            mask = index == i
            counts[i] += mask.sum().item()
            errors[i] += err[mask].sum().item()
    total = sum(errors)
    clipped_distance, direction_distance = distance(a, b), distance(da, db)
    return dict(
        clipped_distance=clipped_distance,
        direction_distance=direction_distance,
        amplification=direction_distance / clipped_distance if clipped_distance else None,
        bin_counts=counts,
        bin_squared_errors=errors,
        bin_error_shares=[e / total if total else 0.0 for e in errors],
        near_zero_error_share=sum(errors[:2]) / total if total else 0.0,
        opposite_nonzero_signs=opposite,
    )


def run():
    protocol, result = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    assert not (ROOT / "analysis.json").exists() and result["status"] == "COMPLETE"
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(protocol[field])
    gpu = [c for c in result["cases"] if c["device"] == "cuda"]
    clipped = [load(c["clipped"]["path"]) for c in gpu]
    raw = [load(c["probe"]["path"]) for c in protocol["cases"]]
    initial = load(protocol["initial_path"])["model"]
    updated = [load(c["updated"]["path"])["model"] for c in gpu]
    updates = [{k: initial[k].double() - u[k].double() for k in initial} for u in updated]
    pairs, distortions, native_pairs, devices = [], [], [], []
    for epsilon in protocol["epsilons"]:
        for i, j in itertools.combinations(range(6), 2):
            pairs.append(
                dict(
                    left=gpu[i]["input_label"],
                    right=gpu[j]["input_label"],
                    epsilon=epsilon,
                    same_policy=gpu[i]["policy"] == gpu[j]["policy"],
                    **pair_metrics(clipped[i], clipped[j], epsilon),
                )
            )
        if epsilon != protocol["native_epsilon"]:
            for i in range(6):
                distortions.append(
                    dict(
                        label=gpu[i]["input_label"],
                        epsilon=epsilon,
                        distance=distance(
                            direction(clipped[i], epsilon), direction(clipped[i], 1e-8)
                        ),
                    )
                )
    for i, j in itertools.combinations(range(6), 2):
        native_pairs.append(
            dict(
                left=gpu[i]["input_label"],
                right=gpu[j]["input_label"],
                same_policy=gpu[i]["policy"] == gpu[j]["policy"],
                raw_gradient_distance=distance(raw[i], raw[j]),
                realized_update_distance=distance(updates[i], updates[j]),
                parameter_distance=distance(updated[i], updated[j]),
            )
        )
    for i, g in enumerate(gpu):
        c = next(
            c
            for c in result["cases"]
            if c["device"] == "cpu" and c["input_label"] == g["input_label"]
        )
        c_clip, c_params = load(c["clipped"]["path"]), load(c["updated"]["path"])["model"]
        c_update = {k: initial[k].double() - c_params[k].double() for k in initial}
        devices.append(
            dict(
                label=g["input_label"],
                clipped_distance=distance(c_clip, clipped[i]),
                parameter_distance=distance(c_params, updated[i]),
                realized_update_distance=distance(c_update, updates[i]),
            )
        )
    baseline = [p for p in pairs if p["epsilon"] == 1e-8 and not p["same_policy"]]
    strong = all(
        p["amplification"] is not None
        and p["amplification"] >= protocol["amplification_min"]
        and p["near_zero_error_share"] >= protocol["near_zero_share_min"]
        for p in baseline
    )
    conditioning = []
    for epsilon in protocol["epsilons"][1:]:
        alternatives = [p for p in pairs if p["epsilon"] == epsilon and not p["same_policy"]]
        ratios = [
            a["direction_distance"] / b["direction_distance"]
            for a, b in zip(alternatives, baseline, strict=True)
        ]
        worst = max(d["distance"] for d in distortions if d["epsilon"] == epsilon)
        conditioning.append(
            dict(
                epsilon=epsilon,
                max_error_ratio=max(ratios),
                max_direction_distortion=worst,
                passed=max(ratios) <= protocol["conditioning_ratio_max"]
                and worst <= protocol["distortion_max"],
            )
        )
    write_json(
        ROOT / "analysis.json",
        dict(
            pairs=pairs,
            distortions=distortions,
            native_pairs=native_pairs,
            device_pairs=devices,
            strong_near_zero_amplification=strong,
            conditioning=conditioning,
            optimizer_steps=0,
            forwards=0,
            backwards=0,
            broad_goal_achieved=False,
        ),
    )
    print("45 direction pairs; 15 native pairs; six device pairs; 12 distortions.", flush=True)


if __name__ == "__main__":
    run()
