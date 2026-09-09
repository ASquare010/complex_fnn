"""Reproduce the four-kernel trained accuracy, serving and memory result."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.reproducibility import sha256, write_json


def report(root: Path = Path("results")) -> dict:
    path = root / "fused_serving_scale384_v1/result.json"
    data = json.loads(path.read_text())
    micro = json.loads((root / "fused_probe_v1/result.json").read_text())
    assert micro["earned_full_model_audit"] and all(data["retention_gates"].values())
    workers = data["workers"]
    ratio = data["fused_throughput_ratio"]["full_reference"]
    native = data["fused_throughput_ratio"]["native_candidate"]
    labels = {
        "full_reference": "Full SwiGLU",
        "native_candidate": "Native BlockShuffle",
        "fused_candidate": "Four-kernel BlockShuffle",
    }
    ratios = {
        key: [
            r["seconds_per_forward"]["full_reference"] / r["seconds_per_forward"][key]
            for r in data["paired_rounds"]
        ]
        for key in labels
    }
    result = {
        "source_sha256": sha256(path),
        "micro_source_sha256": sha256(root / "fused_probe_v1/result.json"),
        "retention_gates": data["retention_gates"],
        "throughput_ratios_to_full": ratios,
        "fused_over_full": ratio,
        "fused_over_native": native,
        "matrix_work": micro["matrix_work"],
        "scope": "One trained checkpoint, one GPU, fixed batch16/context128 compiled full-sequence BF16 inference; not training or autoregressive speed.",
    }
    lines = [
        "# Four-kernel BlockShuffle: the measured serving gap closes",
        "",
        f"The fused implementation passes the frozen full-model accuracy and speed gates. At width384/layer8, seed17, 800 steps, it reaches **{ratio['median']:.3f}x** the throughput of equally compiled full SwiGLU ({ratio['min']:.3f}-{ratio['max']:.3f} across six rotating rounds), and **{native['median']:.3f}x** native compiled BlockShuffle. The result applies to one trained checkpoint and the stated full-sequence workload. It does not establish architectural novelty or the complete research objective.",
        "",
        "Read the [frozen plan and implementation scope](fused_execution_plan.md), [raw full-model audit](../results/fused_serving_scale384_v1/result.json), and [trained correctness/microbenchmark](../results/fused_probe_v1/result.json). All use the original checkpoint, without the later condition-floor edit.",
        "",
        "| Compiled model | Validation NLL | Median throughput/full | Range | Isolated peak MiB | First compile + forward seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        w = workers[key]
        assert (
            w["numerical_checks_pass"]
            and w["parameter_objects_unchanged"]
            and w["validation_tokens"] == 32768
        )
        samples = ratios[key]
        lines.append(
            f"| {label} | {data['paired_compiled_nll'][key]:.6f} | {statistics.median(samples):.3f} | {min(samples):.3f}-{max(samples):.3f} | {w['peak_inference_allocated_vram_bytes'] / 2**20:.2f} | {w['first_compile_and_forward_seconds']:.2f} |"
        )
    lines += [
        "",
        "![Paired compiled serving and isolated allocated memory](../results/plots/fused_serving_scale384.png)",
        "",
        "## What changed and what was counted",
        "",
        "Each FFN uses four custom launches: paired square input factors; paired rectangular expansion plus SiLU/product; rectangular down factor; square down factor. Each stage writes the permutation needed by the next stage directly. The six mathematical factors remain; paired factors execute inside shared launches. Original FP32 parameter objects are retained, and BF16 casts happen inside the kernels. No persistent packed or dense-product cache is added.",
        "",
        "The FFN still has 2801664 learned weights versus 9437184 in full SwiGLU, a 70.3125% reduction. Total model weights are 9099648 versus 15735168, a 42.17% reduction. Logical FFN matrix work is 5603328 FLOPs/token over eight layers. Padded tile work is 7864320 scalar matrix FLOPs/token, 58.33% below the full reference's 18874368. Padded arithmetic is greater than the logical factorized count; pointwise operations, memory movement and backend instruction scheduling are additional. These are arithmetic counts, not hardware counter measurements.",
        "",
        "Matrix accumulations use FP32 with BF16 factor outputs and explicit BF16 SiLU/product boundaries. All 24 trained eager FFN checks and three fresh whole-model logit checks have zero measured error. Eager validation NLL matches the native checkpoint exactly at 2.772705555. These sampled equalities are not a universal bitwise-equivalence proof. The compiled fused model has NLL2.772753686 and fresh-logit RMS error at most .008555, maximum absolute error .0625 versus native eager. All declared BF16 tolerances pass before/after warmup and after paired timing.",
        "",
        "## Protocol and limits",
        "",
        "The initial seven optional kernel/guard tests covered fresh inputs and non-tile-aligned shapes. The compiled isolated-FFN speedup was 2.597x, earning this full-model comparison. Tile32, four warps and two stages were retained; no tile search or autotuning sweep was used. All full models receive default Inductor fullgraph/static-shape compilation with no fallback.",
        "",
        "Memory is measured in a fresh process for each model after compilation and correctness. Each worker has one GPU weight set and 48 MiB of native eager reference logits on CPU; CPU storage is excluded from GPU allocation and disclosed in raw records. Peaks include compiled validation/forward tensors, with no optimizer. The parent holds all three models only for rotating paired timing. Compile time is excluded from throughput and is affected by existing disk caches; it is not a cold-install benchmark. Driver/compiler allocations outside PyTorch's allocator are not measured by allocated-VRAM peaks.",
        "",
        "The first successful serving source archive retains its original kernel argument name O. The current source renames that argument OUTPUT_WIDTH for readability without changing expressions, tiles or arithmetic; subsequent kernel checks and the diagnostic profile use the current source. Source archives distinguish these versions.",
        "",
        "This implementation resolves the measured speed gap on the selected larger checkpoint and workload. The small-model three-seed accuracy gap remains, and the larger-model native quality result now has [three-seed confirmation](scale_replication_results.md) and an [additional validation-tail check](validation_tail_results.md). Convergence, equal tuning effort and broader data remain untested. The [joint three-checkpoint audit](joint_conditioning_results.md) now passes the moderate condition floor with this path; combined serving speed remains unmeasured. Training still uses the original native FFN; no training speedup or full-network gradient lower bound is proved. Established structured projections and fusion do not constitute a verified new primitive.",
        "",
        "Reproduce this report with `uv run python -m src.core.fused_report`.",
    ]
    profile_path = root / "profiles/scale384_blockshuffle_fused_compiled.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        result["profile_sha256"] = sha256(profile_path)
        result["profiled_forwards"] = profile["profiled_forwards"]
        kernel_names = ("_paired_first", "_paired_second", "_down_factor")
        counts = {
            key: sum(row["calls"] for row in profile["rows"] if row["operator"] == key)
            / profile["profiled_forwards"]
            for key in kernel_names
        }
        assert counts == {"_paired_first": 8, "_paired_second": 8, "_down_factor": 16}
        assert not any(row["operator"] == "aten::bmm" for row in profile["rows"])
        result["profile_kernel_calls_per_forward"] = counts
        lines += [
            "",
            "The measured kernel rows contain eight paired-input, eight paired-expansion and sixteen down-factor calls per forward: 32 FFN kernels across eight layers, or four per FFN. No aten::bmm dispatch remains in this fused trace. The 17 common aten::mm calls remain. Duplicate host/kernel rows are counted once.",
        ]
        lines += [
            "",
            "The [compiled operator profile](../results/profiles/scale384_blockshuffle_fused_compiled.json) retains instrumented CPU/device events. Host ranges and CUDA kernel rows overlap; its timing must not be summed or substituted for the paired serving measurement.",
        ]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.7), constrained_layout=True)
    colors = ("#505a6b", "#cc792b", "#087e8b")
    for i, (key, label) in enumerate(labels.items()):
        samples = ratios[key]
        median = statistics.median(samples)
        axes[0].bar(i, median, color=colors[i], alpha=0.85)
        axes[0].plot(
            [i - 0.07 + 0.028 * j for j in range(len(samples))], samples, "o", color="#182333", ms=3
        )
        axes[0].text(i, max(samples) + 0.05, f"{median:.3f}x", ha="center", fontsize=9)
        memory = workers[key]["peak_inference_allocated_vram_bytes"] / 2**20
        axes[1].bar(i, memory, color=colors[i])
        axes[1].text(i, memory + 3, f"{memory:.2f}", ha="center", fontsize=9)
    ticks = ["Full\nSwiGLU", "Native\nBlockShuffle", "Four-kernel\nBlockShuffle"]
    for ax in axes:
        ax.set_xticks(range(3), ticks)
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(0.8, color="#942e36", ls="--", lw=1, label="Required .8 floor")
    axes[0].set(
        ylabel="Throughput / equally compiled full model",
        title="Six rotating rounds; dots show every sample",
        ylim=(0, 1.5),
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        ylabel="Peak allocated inference memory (MiB)",
        title="Fresh worker per model; compilation excluded",
        ylim=(0, 195),
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            root / "plots" / f"fused_serving_scale384.{suffix}", dpi=100, bbox_inches="tight"
        )
    plt.close(fig)
    write_json(root / "fused_summary.json", result)
    Path("research/fused_execution_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
