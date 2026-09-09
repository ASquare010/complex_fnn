"""Keep full-trajectory evidence separate from local compiler fidelity."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.activation_training_repeat import NATIVE, RUN, verify
from src.core.reproducibility import sha256, write_json


def report():
    root = Path("results/activation_training_repeat_v1")
    recorded = json.loads((root / "result.json").read_text())
    checked = verify()
    assert checked == recorded
    native = json.loads((NATIVE / "metrics.json").read_text())
    compiled = json.loads((RUN / "metrics.json").read_text())
    verdict = "PASS" if recorded["repeat_passes"] else "FAIL"
    lines = [
        "# Rational activation: compiled full-training repeat",
        "",
        f"**The frozen full-training repeat is {verdict}.** Compiled-product training finishes at NLL {compiled['validation_loss']:.6f}, versus native rational {native['validation_loss']:.6f} ({recorded['relative_native_nll_percent']:+.3f}%). Its memory limit passes, but the quality improvement does not survive this execution change.",
        "",
        "Read the [frozen repeat plan](activation_training_repeat_plan.md), [native activation results](learnable_activation_results.md), [isolated memory audit](activation_execution_results.md) and [raw repeat checks](../results/activation_training_repeat_v1/result.json).",
        "",
        "| Execution | Final NLL | Allocated training peak MiB | FFN weights |",
        "|---|---:|---:|---:|",
    ]
    for name, m in (("Native rational", native), ("Compiled product repeat", compiled)):
        lines.append(
            f"| {name} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {m['ffn_parameters']:,} |"
        )
    lines += ["", "## Frozen decisions", "", "| Gate | Decision |", "|---|---|"]
    for key, value in recorded["gates"].items():
        lines.append(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |")
    c = compiled["activation_compilation"]
    lines += [
        "",
        "![Native and compiled learning trajectories](../results/plots/activation_training_repeat.png)",
        "",
        "## Reproduction checks",
        "",
        "The training-step loop AST matches the archived native loop. Model dimensions, optimizer groups, sampling configuration, train/validation hashes and token budgets match. All initial parameters survive compilation warmup unchanged, initial complete-validation NLL matches, and the final sampling RNG state matches the native run. Source archives, real parameter counts, all recorded gradients/activations and the final checkpoint were checked. No source checkpoint was overwritten.",
        "",
        f"Compiler preparation uses one zero-token forward/backward and zero optimizer updates. It consumes no sampling RNG. Recorded preparation time is {c['warmup_seconds']:.2f} seconds and allocated peak through that stage is {c['allocated_peak_through_compile_warmup'] / 2**20:.2f} MiB. Training peak resets after initial validation/diagnostics, consistent with the shared protocol. Diagnostics temporarily use native pointwise functions and restore compiled execution; recorded validation uses the compiled product.",
        "",
        "## What the failure means",
        "",
        "The earlier isolated audit had exact sampled logits, global gradient relative L2 error 0.000784 and activation-gradient error 0.000169, and passed full-validation fidelity at the selected checkpoint. Those local checks did not ensure equivalent optimization over 200 updates. The full trajectory is decisive for accepting an execution backend for training.",
        "",
        "Rounding/reduction differences are a plausible contributor, but no particular compiler operation has been isolated as the cause. The limited activation derivative bounds do not bound the complete training trajectory. The native rational result remains a single-seed quality finding with excessive eager memory. The compiled product remains an optional experimental backend; it is not the recommended training recipe and earns no longer-budget promotion.",
        "",
        "## Simpler mechanism to test next",
        "",
        "A post-hoc straight-line fit to each native rational correction captures 99.768% of its squared magnitude on a uniform [-6,6] grid. Residual RMS after that fit is 0.000833 versus correction RMS 0.017315. The [shape audit](../results/verification/activation_affine_shape_audit_v1.json) is neither data-weighted nor a retrained affine control. Together with the 0.00562% reset cost, it motivates testing a simple learned affine correction and isolating compilation of only that correction before attributing the gain to flexible curvature.",
        "",
        "These experiments establish neither a new primitive nor multi-seed, convergence, broad-transfer or full-network stability claims.",
        "",
    ]
    Path("research/activation_training_repeat_results.md").write_text(
        chr(10).join(lines), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")
    for path, label, color in (
        (NATIVE, "Native rational", "#2479a3"),
        (RUN, "Compiled product repeat", "#a9561a"),
    ):
        history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
        history = [r for r in history if r["step"] >= 50]
        ax.plot(
            [r["step"] for r in history],
            [r["validation_loss"] for r in history],
            marker="o",
            label=label,
            color=color,
        )
    ax.set(
        title="Same recipe and sampling: execution changes the trajectory",
        xlabel="Optimizer steps",
        ylabel="Complete-validation NLL (lower is better)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    for extension in ("png", "svg"):
        fig.savefig(Path("results/plots") / f"activation_training_repeat.{extension}", dpi=160)
    plt.close(fig)
    write_json(
        Path("results/activation_training_repeat_summary.json"),
        {
            "verdict": verdict,
            "result": checked,
            "protocol_sha256": sha256(root / "protocol.json"),
            "native_metrics_sha256": sha256(NATIVE / "metrics.json"),
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "nll": compiled["validation_loss"],
                "peak_mib": compiled["peak_allocated_vram_bytes"] / 2**20,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    report()
