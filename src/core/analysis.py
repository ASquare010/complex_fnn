"""Aggregate seed variability and paired differences without selecting best runs."""

import argparse
import json
import math
import statistics
from pathlib import Path

from src.core.report import cohort
from src.core.reproducibility import write_json


def summarize(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def analyze(root: Path) -> dict:
    cohorts = {}
    for path in sorted((root / "runs").glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        group = cohorts.setdefault(cohort(m), {})
        variant = m["model"]["variant"]
        # Explicit hidden/group settings distinguish configurations of a family.
        label = f"{variant}:h{m['model']['hidden']}:g{m['model']['groups']}"
        runs = group.setdefault(label, {})
        seed = m["training"]["seed"]
        if seed in runs:
            raise ValueError(
                f"Duplicate seed/config in cohort: {label}, {seed}; select a reviewed cohort explicitly"
            )
        runs[seed] = m
    output = {}
    for key, group in cohorts.items():
        aggregate, paired = {}, {}
        for label, runs in group.items():
            aggregate[label] = {
                metric: summarize([r[metric] for r in runs.values()])
                for metric in (
                    "validation_loss",
                    "perplexity",
                    "training_tokens_per_second",
                    "inference_tokens_per_second",
                    "peak_allocated_vram_bytes",
                    "clipped_step_fraction",
                )
            }
            aggregate[label]["seeds"] = sorted(runs)
            aggregate[label]["runs"] = [r["run"] for r in runs.values()]
            for control in (
                "gelu:h0:g8",
                "swiglu:h0:g8",
                "gelu_narrow:h0:g8",
                "swiglu_narrow:h0:g8",
                "shared_gelu:h0:g8",
                "shared_swiglu:h0:g8",
            ):
                if control not in group or label == control:
                    continue
                seeds = sorted(set(runs) & set(group[control]))
                deltas = [
                    runs[s]["validation_loss"] - group[control][s]["validation_loss"] for s in seeds
                ]
                relative = [
                    100 * (runs[s]["validation_loss"] / group[control][s]["validation_loss"] - 1)
                    for s in seeds
                ]
                result = {
                    "seeds": seeds,
                    "nll_difference": summarize(deltas),
                    "relative_nll_percent": summarize(relative),
                    "candidate_better_seeds": sum(x < 0 for x in deltas),
                }
                if len(deltas) == 3:
                    # Student-t(2) two-sided 95% critical value. Exploratory, normality assumed.
                    critical = math.sqrt(2) * 0.95 / math.sqrt(1 - 0.95**2)
                    margin = critical * statistics.stdev(deltas) / math.sqrt(3)
                    result["paired_nll_95pct_t_interval"] = [
                        statistics.mean(deltas) - margin,
                        statistics.mean(deltas) + margin,
                    ]
                paired[f"{label} minus {control}"] = result
        output[key] = {"aggregate": aggregate, "paired": paired}
    result = {
        "cohorts": output,
        "interpretation": "Exploratory short-budget seed uncertainty on one fixed validation set; t intervals assume normal paired seed differences, unadjusted for multiple comparisons. Not a scaling or generalization guarantee.",
    }
    write_json(root / "seed_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))
