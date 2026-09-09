"""Verified trajectory comparison for the correction-only compiler repeat."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.activation_correction_repeat import CONFIG, PLAN, RUN
from src.core.activation_training_repeat import NATIVE, verify
from src.core.activation_training_repeat import RUN as PRODUCT_RUN
from src.core.reproducibility import sha256, write_json


def report():
    root = Path("results/activation_correction_repeat_v1")
    result = json.loads((root / "result.json").read_text())
    assert result == verify(RUN, CONFIG)
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["plan_sha256"] == sha256(PLAN)
    cases = {
        "Native rational": NATIVE,
        "Compiled full product": PRODUCT_RUN,
        "Compiled correction only": RUN,
    }
    rows = {k: json.loads((p / "metrics.json").read_text()) for k, p in cases.items()}
    histories = {
        k: [json.loads(line) for line in (p / "history.jsonl").read_text().splitlines()]
        for k, p in cases.items()
    }
    deltas = [
        b["validation_loss"] - a["validation_loss"]
        for a, b in zip(histories["Native rational"], histories["Compiled correction only"])
    ]
    lines = [
        "# Correction-only compilation: full training repeat",
        "",
        f"**Frozen decision: {'PASS' if result['repeat_passes'] else 'FAIL'}.** The correction-only repeat finishes at NLL {result['validation_nll']:.9f}, versus native rational {result['native_rational_nll']:.9f} ({result['relative_native_nll_percent']:+.6f}%). Allocated training peak is {result['peak_allocated_vram_bytes'] / 2**20:.3f} MiB.",
        "",
        "This is a fresh 200-step, seed-17, BF16 WikiText run from the same initialization, with 409,600 sampled training tokens and 322,688 validation targets. Only the rational correction is compiled; SiLU and gate multiplication retain native forward/backward. All learning rates, optimizer groups, parameter/data hashes and token counts are checked. The earlier full-product failure is retained separately.",
        "",
        "| Execution | Final NLL | Peak allocated MiB |",
        "|---|---:|---:|",
    ]
    for key, row in rows.items():
        lines.append(
            f"| {key} | {row['validation_loss']:.9f} | {row['peak_allocated_vram_bytes'] / 2**20:.3f} |"
        )
    lines += [
        "",
        "![Three training trajectories](../results/plots/activation_correction_repeat.png)",
        "",
        "## Frozen gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in result["gates"].items():
        lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines += [
        "",
        f"Correction-only minus native NLL at steps 1, 50, 100, 150, 200: `{deltas}`. The maximum absolute recorded trajectory difference is {max(abs(x) for x in deltas):.9f}. This is measured trajectory agreement at these checkpoints, not a mathematical long-run guarantee.",
        "",
        "The source verifier confirms identical training-loop AST, initial validation within 1e-7, unchanged warmup parameters, zero optimizer updates or sampling during compile warmup, matching optimizer groups and final sampling RNG, all finite recorded diagnostics and an intact archived checkpoint. Native diagnostics temporarily bypass compilation and restore it; validation uses the selected backend. Compilation overhead is separately recorded and excluded from steady training peaks. Serial timing is not paired speed evidence.",
        "",
        "The [local execution audit](activation_correction_execution_results.md) reduced measured gradient relative L2 from 0.00078429 to 3.95e-8 by changing the compilation boundary. The full repeat is the stronger evidence for this implementation. It does not establish compiler behavior across devices, seeds, precision modes, scales or future software versions.",
        "",
        "[Frozen repeat plan](activation_correction_repeat_plan.md), [raw verification](../results/activation_correction_repeat_v1/result.json), [native shape study](learnable_activation_results.md), and [simpler affine result](affine_activation_results.md). The original gold research goal still needs independent seeds, convergence, scale/corpus controls, throughput and credible novelty.",
    ]
    Path("research/activation_correction_repeat_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        Path("results/activation_correction_repeat_summary.json"),
        {
            "verified": result,
            "trajectory_nll_deltas": deltas,
            "max_abs_trajectory_nll_difference": max(abs(x) for x in deltas),
            "protocol_sha256": sha256(root / "protocol.json"),
        },
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.3), layout="constrained")
    for key, color, style in zip(cases, ("#246a9a", "#ad5430", "#24836a"), ("-", "-", "--")):
        h = histories[key]
        ax.plot(
            [r["step"] for r in h],
            [r["validation_loss"] for r in h],
            color=color,
            linestyle=style,
            marker="o",
            label=key,
        )
    ax.set(
        xlabel="Optimizer step",
        ylabel="Validation NLL",
        title="Rational activation: native and two compilation boundaries",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    for suffix in ("png", "svg"):
        fig.savefig(f"results/plots/activation_correction_repeat.{suffix}", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "pass": result["repeat_passes"],
                "nll": result["validation_nll"],
                "relative_native_percent": result["relative_native_nll_percent"],
                "trajectory_nll_deltas": deltas,
            }
        )
    )


if __name__ == "__main__":
    report()
