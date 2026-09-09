"""Recompute joint conditioning decisions and render the measured tradeoff."""

import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.core.joint_conditioning import SEEDS, compare_seed
from src.core.reproducibility import sha256, write_json


def report(root=Path("results")):
    folder = root / "joint_conditioning_scale384_v1"
    data = json.loads((folder / "result.json").read_text())
    assert data["status"] == "complete" and len(data["records"]) == 6
    refs = json.loads((root / "validation_tail_scale384_v1/result.json").read_text())["records"]
    protocol = data["protocol"]
    with zipfile.ZipFile(folder / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == value
            for name, value in protocol["provenance"]["source_files"].items()
        )
        assert (
            hashlib.sha256(archive.read("research/joint_conditioning_plan.md")).hexdigest()
            == protocol["plan_sha256"]
        )
    summary = {
        "result_sha256": sha256(folder / "result.json"),
        "source_archive_sha256": sha256(folder / "source.zip"),
        "per_seed": {},
    }
    critical = (
        "src/core/joint_conditioning.py",
        "src/core/conditioning.py",
        "src/core/benchmark.py",
        "src/core/compile_serving.py",
        "src/core/fused_serving.py",
        "src/core/model_audit.py",
        "src/core/validation_tail.py",
        "src/core/data.py",
        "src/core/transformer.py",
        "src/core/config.py",
        "src/blockshuffle_ffn/triton_inference.py",
        "src/blockshuffle_ffn/__init__.py",
        "src/grouped_ffn/__init__.py",
        "pyproject.toml",
        "uv.lock",
    )
    for seed in SEEDS:
        original, floor = (data["records"][f"s{seed}_{kind}"] for kind in ("original", "floor"))
        controls = {
            k: refs[f"scale384_{k}_s{seed}_800"]
            for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
        }
        gates = compare_seed(original, floor, controls)
        assert gates == data["gates"][str(seed)]
        for kind, row in (("original", original), ("floor", floor)):
            case_dir = folder / f"s{seed}_{kind}"
            assert json.loads((case_dir / "result.json").read_text()) == row
            assert all(
                row["provenance"]["source_files"][name]
                == protocol["provenance"]["source_files"][name]
                for name in critical
            )
            checkpoint = root / "runs" / row["run"] / "checkpoint.pt"
            assert sha256(checkpoint) == row["checkpoint_sha256"]
            assert (
                row["other_weights_bitwise_unchanged"]
                and row["parameter_objects_unchanged"]
                and row["source_checkpoint_unchanged"]
            )
            assert row["unique_parameters"] == 9099648 and row["ffn_parameters"] == 2801664
            assert (
                row["data_hashes"]
                == json.loads((root / "validation_tail_scale384_v1/result.json").read_text())[
                    "protocol"
                ]["data_hashes"]
            )
            assert row["finite_gradients"] == all(
                r["finite"] for r in row["gradient_flow"]["records"].values()
            )
            assert row["worst_down_condition"] == max(
                r["projection"]["condition_nonzero"] for r in row["down_spectra"].values()
            )
            for execution in ("native", "compiled"):
                detail = json.loads((case_dir / f"{execution}_tail_windows.json").read_text())
                assert len(detail) == 1312 and [r["window"] for r in detail] == list(
                    range(256, 1568)
                )
                assert all(
                    r["targets"] == 128
                    and r["first_target"] == r["window"] * 128 + 1
                    and r["last_target"] == (r["window"] + 1) * 128
                    and math.isfinite(r["nll"])
                    for r in detail
                )
                q = row[execution]
                assert q["prefix_targets"] == 32768 and q["tail_targets"] == 167936
                assert abs(sum(r["nll"] * 128 for r in detail) / 167936 - q["tail_nll"]) < 1e-12
                assert (
                    abs(
                        (q["prefix_nll"] * 32768 + q["tail_nll"] * 167936) / 200704
                        - q["aggregate_nll"]
                    )
                    < 1e-12
                )
            logits = (
                row["compiled_errors_before_warmup"]
                + row["compiled_errors_after_warmup"]
                + row["fused_eager_errors"]
            )
            assert all(r["finite"] and r["max_abs"] <= 0.15 and r["rms"] <= 0.01 for r in logits)
            assert all(
                abs(row["native"][region + "_nll"] - row["compiled"][region + "_nll"]) <= 0.001
                for region in ("prefix", "tail")
            )
            if kind == "floor":
                factors = torch.load(
                    case_dir / "transformed_factors.pt", weights_only=True, map_location="cpu"
                )
                assert sha256(case_dir / "transformed_factors.pt") == row["transform_file_sha256"]
                assert set(factors) == {f"blocks.{i}.ffn.down.second.weight" for i in range(8)}
                assert all(
                    hashlib.sha256(t.numpy().tobytes()).hexdigest() == row["changes"][k]["sha256"]
                    and bool(torch.isfinite(t).all())
                    for k, t in factors.items()
                )
                assert all(
                    abs(c["frobenius_norm_ratio"] - 1) < 1e-6 for c in row["changes"].values()
                )
        summary["per_seed"][seed] = {
            "condition_before": original["worst_down_condition"],
            "condition_after": floor["worst_down_condition"],
            "condition_improvement": original["worst_down_condition"]
            / floor["worst_down_condition"],
            "native_floor_relative_nll_percent": {
                r: 100 * (floor["native"][r + "_nll"] / original["native"][r + "_nll"] - 1)
                for r in ("prefix", "tail")
            },
            "compiled_floor_relative_nll_percent": {
                r: 100 * (floor["compiled"][r + "_nll"] / original["native"][r + "_nll"] - 1)
                for r in ("prefix", "tail")
            },
            "max_compiled_rms": max(
                r["rms"]
                for r in floor["compiled_errors_before_warmup"]
                + floor["compiled_errors_after_warmup"]
            ),
            "gates": gates,
        }
    summary["all_gates_pass"] = all(all(v["gates"].values()) for v in summary["per_seed"].values())
    assert summary["all_gates_pass"] == data["all_gates_pass"]
    summary["mean_compiled_floor_nll"] = {
        r: statistics.mean(data["records"][f"s{s}_floor"]["compiled"][r + "_nll"] for s in SEEDS)
        for r in ("prefix", "tail")
    }
    summary["artifact_checks_pass"] = True
    lines = [
        "# Fixed condition floor with fused execution: three checkpoints",
        "",
        f"The frozen joint decision is **{'PASS' if summary['all_gates_pass'] else 'FAIL'}** across all three larger 800-step checkpoints. This round tests an existing post-training weight adjustment with the existing inference implementation. Read the [plan](joint_conditioning_plan.md) and [raw six-case evidence](../results/joint_conditioning_scale384_v1/result.json).",
        "",
        "## Projection condition and intervention cost",
        "",
        "The fixed strength is 0.1, selected in the earlier seed-17 study. Only the eight square down factors change. FP64 SVD flooring preserves aggregate Frobenius norm, returns FP32 weights and adds no parameters. Down projections remain rectangular; their input nullspaces remain. Condition numbers below refer to nonzero singular values of the full down projection, not a whole-network Jacobian.",
        "",
        "| Seed | Worst condition before | After | Improvement | Native prefix NLL change | Native tail NLL change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed, row in summary["per_seed"].items():
        p = row["native_floor_relative_nll_percent"]
        lines.append(
            f"| {seed} | {row['condition_before']:.2f} | {row['condition_after']:.2f} | {row['condition_improvement']:.1f}x | {p['prefix']:+.5f}% | {p['tail']:+.5f}% |"
        )
    lines += [
        "",
        "## Actual compiled model quality",
        "",
        "The four-kernel model is compiled with default Inductor, a static full graph and the unchanged tile settings. Kernel errors are measured against the native implementation of the same weights. Separately, combined quality changes below compare compiled floored weights to original native weights. Both regions must satisfy every quality gate.",
        "",
        "| Seed | Compiled floor prefix NLL | Compiled floor tail NLL | Combined prefix change | Combined tail change | All joint gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for seed, row in summary["per_seed"].items():
        q = data["records"][f"s{seed}_floor"]["compiled"]
        p = row["compiled_floor_relative_nll_percent"]
        lines.append(
            f"| {seed} | {q['prefix_nll']:.6f} | {q['tail_nll']:.6f} | {p['prefix']:+.5f}% | {p['tail']:+.5f}% | {'PASS' if all(row['gates'].values()) else 'FAIL'} |"
        )
    lines += [
        "",
        "Every native original score reproduced the prior prefix/tail result within the frozen tolerance. Evaluation uses 32,768 prefix targets and 167,936 disjoint tail targets, with per-window tail losses retained. These are the same TinyStories stories and the tail was already scored before this intervention; this is no longer a fresh holdout.",
        "",
        "All three original and all three floored cases passed finite sampled-gradient and fused numerical checks. Stored factor hashes, unchanged source checkpoint hashes, parameter objects/counts, source archives, target positions and weighted loss aggregation were verified. All comparisons use 2,801,664 unique FFN weights and 9,099,648 total weights. The original full controls have 9,437,184 FFN weights: reduction remains 70.3125%.",
        "",
        "![Conditioning and measured quality cost](../results/plots/joint_conditioning_scale384.png)",
        "",
        "## Limits and retained failure",
        "",
        "The first seed-17 floor worker terminated with Windows access violation 3221225477 during PyTorch import, before setup or measurement. Three fresh unchanged imports passed. The five unmeasured cases were then resumed once with all experimental sources/settings unchanged. The [original failure](../results/joint_conditioning_scale384_v1/failure.json), import log, probes and [continuation record](../results/joint_conditioning_scale384_v1/continuation.json) remain archived. Root cause is unresolved.",
        "",
        "This experiment supplies no paired timing, inference-memory or training-efficiency comparison. The earlier 1.206x serving result belongs to the original seed-17 checkpoint, not this combined option. The fixed-batch CPU FP32 gradients are sampled loss directions; better projection conditioning does not prove better full-network gradient propagation. No new architectural novelty, convergence, equal tuning or second-corpus result is established. Native training remains unchanged.",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    x = list(range(3))
    rows = list(summary["per_seed"].values())
    for offset, key, label, color in (
        (-0.18, "condition_before", "Original", "#505a6b"),
        (0.18, "condition_after", "Fixed floor", "#087e8b"),
    ):
        axes[0].bar(
            [i + offset for i in x], [r[key] for r in rows], width=0.34, label=label, color=color
        )
    axes[0].set(
        yscale="log",
        ylabel="Worst down-projection condition (log scale)",
        title="Same fixed floor on every checkpoint",
    )
    for offset, key, label, color in (
        (-0.18, "prefix", "Selection prefix", "#505a6b"),
        (0.18, "tail", "Already-scored tail", "#087e8b"),
    ):
        positions = [i + offset for i in x]
        values = [r["compiled_floor_relative_nll_percent"][key] for r in rows]
        axes[1].bar(positions, values, width=0.34, label=label, color=color)
        for xx, yy in zip(positions, values, strict=True):
            axes[1].annotate(
                f"{yy:+.3f}%",
                (xx, yy),
                xytext=(0, 4 if yy >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[1].axhline(0.1, color="#942e36", ls="--", lw=1, label="Maximum allowed cost")
    axes[1].axhline(0, color="#444444", lw=0.6)
    axes[1].set(
        ylabel="Compiled floor / original native NLL change (%)",
        title="Joint quality cost on both regions",
    )
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(min(-0.015, lo - 0.01), max(0.125, hi))
    for ax in axes:
        ax.set_xticks(x, [f"Seed {s}" for s in SEEDS])
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(
            root / "plots" / f"joint_conditioning_scale384.{ext}", dpi=100, bbox_inches="tight"
        )
    plt.close(fig)
    write_json(root / "joint_conditioning_summary.json", summary)
    Path("research/joint_conditioning_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
