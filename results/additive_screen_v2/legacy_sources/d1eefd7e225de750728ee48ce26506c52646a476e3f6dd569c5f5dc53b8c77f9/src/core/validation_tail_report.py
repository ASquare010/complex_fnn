"""Reproduce the held-out-tail comparison and exact target accounting."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.reproducibility import sha256, write_json
from src.core.scale_report import RECIPES


def report(root: Path = Path("results")):
    folder = root / "validation_tail_scale384_v1"
    data = json.loads((folder / "result.json").read_text())
    assert data["status"] == "complete" and len(data["records"]) == 12
    records = data["records"]
    per_seed = {}
    window_hashes = {}
    for name, r in records.items():
        path = folder / f"{name}.json"
        detail = json.loads(path.read_text())
        rows = detail["windows"]
        assert len(rows) == 1312 and [v["window"] for v in rows] == list(range(256, 1568))
        assert rows[0]["first_target"] == 32769 and rows[-1]["last_target"] == 200704
        assert sum(v["targets"] for v in rows) == 167936
        assert all(
            v["targets"] == 128 and v["last_target"] - v["first_target"] + 1 == 128 for v in rows
        )
        assert all(b["first_target"] == a["last_target"] + 1 for a, b in zip(rows, rows[1:]))
        assert abs(sum(v["nll"] * v["targets"] for v in rows) / 167936 - r["tail_nll"]) < 1e-12
        assert (
            abs((r["prefix_nll"] * 32768 + r["tail_nll"] * 167936) / 200704 - r["aggregate_nll"])
            < 1e-12
        )
        assert r["prefix_nll"] == r["prefix_reference_nll"]
        window_hashes[name] = sha256(path)
    stats = {}
    for key in RECIPES:
        stats[key] = {}
        for field in ("prefix_nll", "tail_nll", "aggregate_nll"):
            values = [records[f"scale384_{key}_s{seed}_800"][field] for seed in (17, 29, 43)]
            stats[key][field] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
    for seed in (17, 29, 43):
        group = {key: records[f"scale384_{key}_s{seed}_800"] for key in RECIPES}
        paired = {
            key: 100 * (group["blockshuffle"]["tail_nll"] / group[key]["tail_nll"] - 1)
            for key in ("full_swiglu", "full_gelu", "calibrated_narrow")
        }
        per_seed[seed] = {
            "relative_tail_nll_percent": paired,
            "gates": {
                "within_1_percent_full_swiglu": paired["full_swiglu"] <= 1,
                "within_1_percent_full_gelu": paired["full_gelu"] <= 1,
                "beats_calibrated_narrow": paired["calibrated_narrow"] < 0,
            },
        }
    result = {
        "result_sha256": sha256(folder / "result.json"),
        "per_window_file_sha256": window_hashes,
        "statistics": stats,
        "per_seed": per_seed,
        "all_tail_quality_gates_pass": all(all(r["gates"].values()) for r in per_seed.values()),
        "all_prefix_scores_reproduce_exactly": True,
        "all_target_accounting_checks_pass": True,
        "aggregate_relative_tail_nll_percent": {
            k: 100 * (stats["blockshuffle"]["tail_nll"]["mean"] / stats[k]["tail_nll"]["mean"] - 1)
            for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
        },
    }
    lines = [
        "# Previously unscored validation tail: three-seed comparison",
        "",
        "All twelve fixed checkpoints were scored on 167,936 additional targets after the replication cohort finished. Every original 32,768-target prefix score reproduced exactly. Read the [frozen plan](validation_tail_plan.md) and [raw measurements](../results/validation_tail_scale384_v1/result.json). This is additional evidence within the same corpus, not a second corpus or convergence result.",
        "",
        "| Recipe | Prefix NLL mean +/- SD | Tail NLL mean +/- SD | All complete windows NLL mean +/- SD |",
        "|---|---:|---:|---:|",
    ]
    for key, label in RECIPES.items():
        values = " | ".join(
            f"{stats[key][field]['mean']:.6f} +/- {stats[key][field]['sample_sd']:.6f}"
            for field in ("prefix_nll", "tail_nll", "aggregate_nll")
        )
        lines.append(f"| {label} | {values} |")
    lines += [
        "",
        "| Seed | Tail change/full SwiGLU | Tail change/full GELU | Tail change/calibrated narrow | All tail gates |",
        "|---|---:|---:|---:|---|",
    ]
    for seed, row in per_seed.items():
        p = row["relative_tail_nll_percent"]
        lines.append(
            f"| {seed} | {p['full_swiglu']:+.4f}% | {p['full_gelu']:+.4f}% | {p['calibrated_narrow']:+.4f}% | {'PASS' if all(row['gates'].values()) else 'FAIL'} |"
        )
    lines += [
        "",
        f"The frozen all-seed tail decision is **{'PASS' if result['all_tail_quality_gates_pass'] else 'FAIL'}**. Aggregate tail NLL changes are "
        + ", ".join(
            f"{RECIPES[k]} {v:+.4f}%"
            for k, v in result["aggregate_relative_tail_nll_percent"].items()
        )
        + ". Negative change favors BlockShuffle.",
        "",
        "![Prefix and tail paired quality differences](../results/plots/validation_tail_scale384.png)",
        "",
        "The tail uses windows 256 through 1567 at context128, with target positions 32769 through 200704 inclusive. These 1312 complete windows form 82 batches of16. Tail targets do not overlap the prefix; the boundary input token is shared. The final98 tokens lack a complete remaining window and are omitted. Every per-window loss and target range is archived and independently checked. The combined NLL weights prefix and tail by32768 and167936 targets, not by equal section weights.",
        "",
        "The same four recipes and all three seeds were retained. No strengths, windows or thresholds were changed after tail scores became visible. The result supports transfer beyond the selection prefix within these 1000 validation stories. It does not establish robustness on another corpus, equal historical tuning, convergence, autoregressive speed or architectural novelty. This tail is now scored and cannot be called fresh evidence for future model selection.",
        "",
        "Reproduce with `uv run python -m src.core.validation_tail_report`.",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.3), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("full_swiglu", "calibrated_narrow"),
        (
            "Against full SwiGLU: allowed upper change +1%",
            "Against calibrated narrow: change must be below 0%",
        ),
        strict=True,
    ):
        for field, label, color in (
            ("prefix_nll", "Selection prefix", "#505a6b"),
            ("tail_nll", "Additional tail", "#087e8b"),
        ):
            values = [
                100
                * (
                    records[f"scale384_blockshuffle_s{seed}_800"][field]
                    / records[f"scale384_{key}_s{seed}_800"][field]
                    - 1
                )
                for seed in (17, 29, 43)
            ]
            ax.plot(range(3), values, "o-", color=color, label=label)
        ax.axhline(0, color="#666666", lw=1, ls="--")
        ax.set_xticks(range(3), ["Seed 17", "Seed 29", "Seed 43"])
        ax.set(ylabel="Relative NLL change (%)", title=title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(
            root / "plots" / f"validation_tail_scale384.{suffix}", dpi=100, bbox_inches="tight"
        )
    plt.close(fig)
    write_json(root / "validation_tail_summary.json", result)
    Path("research/validation_tail_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
