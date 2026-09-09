"""Verify H047 qualification and all three router-free validation results."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.core.headwise_screen import CRITICAL, PLAN, RECIPE
from src.core.multihead_screen import (
    RATES,
    candidate_gates,
    output_path,
    verify_archive,
    verify_trial,
)
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import RECIPES as CONTROLS
from src.core.wikitext_screen import output_path as control_path
from src.core.wikitext_screen import select_best

ROOT = Path("results/headwise_screen_v1")


def report():
    result = json.loads((ROOT / "result.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 3
    assert sha256(PLAN) == protocol["plan_sha256"] == result["plan_sha256"]
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    q = json.loads((ROOT / "qualification/result.json").read_text())
    assert q["status"] == "PASS" and q["validation_targets"] == 0
    assert q["precision_relative_l2"] <= 0.05 and len(q["history"]) == 10
    assert q["history"][-1]["loss"] < q["history"][0]["loss"]
    for h in q["history"]:
        assert all(math.isfinite(v) and v > 0 for v in h["gradient_norms_pre_clip"].values())
    assert all(
        q["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
        for n in CRITICAL
    )
    assert sha256(ROOT / "qualification/checkpoint.pt") == q["checkpoint_sha256"]
    ckpt = torch.load(ROOT / "qualification/checkpoint.pt", map_location="cpu", weights_only=True)
    assert (
        ckpt["step"] == 10
        and ckpt["model_config"] == q["model"]
        and ckpt["training_config"] == q["training"]
    )
    assert (
        ckpt["batch_sha256"]
        == q["batch_sha256"]
        == "258e6c6132c5ac9d202dd965ee64a530055aa414fd62b396b79c4d1be392362e"
    )
    assert sum(t.numel() for t in ckpt["model"].values()) == 9099648
    assert all(torch.isfinite(t).all() for t in ckpt["model"].values())
    del ckpt
    baseline = json.loads((ROOT / "baseline_initial.json").read_text())
    assert baseline["status"] == "PASS" and all(
        baseline["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
        for n in CRITICAL
    )
    controls = {}
    for r in CONTROLS:
        controls[r] = []
        for rate in RATES:
            p = control_path(r, rate)
            frozen = next(t for t in initial["controls"] if t["run"] == p.name)
            assert sha256(p / "metrics.json") == frozen["metrics_sha256"]
            assert sha256(p / "checkpoint.pt") == frozen["checkpoint_sha256"]
            assert sha256(p / "source.zip") == frozen["source_archive_sha256"]
            m = json.loads((p / "metrics.json").read_text())
            verify_archive(p, m)
            controls[r].append(m)
    rows = []
    for t in result["trials"]:
        m = verify_trial(RECIPE, t["rate"], initial)
        p = output_path(RECIPE, t["rate"])
        assert sha256(p / "metrics.json") == t["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == t["checkpoint_sha256"]
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        rows.append(m)
    selected = select_best(rows)
    references = {r: select_best(cells) for r, cells in controls.items()}
    decision = candidate_gates(selected, references)
    assert decision == result["decisions"] and selected["run"] == result["selected"]
    same_rate = {
        str(round(rate * 1e6)): {
            r: 100 * (rows[i]["validation_loss"] / cells[i]["validation_loss"] - 1)
            for r, cells in controls.items()
        }
        for i, rate in enumerate(RATES)
    }
    assert same_rate == result["same_rate_relative_nll_percent"]
    parallel = []
    for rate in RATES:
        p = output_path("multihead_calibrated", rate)
        frozen = next(t for t in initial["parallel_trials"] if t["run"] == p.name)
        assert sha256(p / "metrics.json") == frozen["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == frozen["checkpoint_sha256"]
        parallel.append(json.loads((p / "metrics.json").read_text()))
    parallel_best = select_best(parallel)
    saving = 100 * (
        1 - selected["peak_allocated_vram_bytes"] / parallel_best["peak_allocated_vram_bytes"]
    )
    differences = {
        r: 100 * (selected["validation_loss"] / m["validation_loss"] - 1)
        for r, m in references.items()
    }
    lines = [
        "# H047: Router-free headwise SwiGLU",
        "",
        "**Memory passes; quality fails.** The simpler layer uses exactly the BlockShuffle parameter budget and reduces allocated training peak, but it does not earn longer training. The best rate remains on the upper boundary of the frozen grid.",
        "",
        "[Frozen protocol](headwise_screen_plan.md), [equations and proof](../src/multihead_ffn/headwise.md), [raw decisions](../results/headwise_screen_v1/result.json), [parallel comparison](multihead_screen_results.md).",
        "",
        "## Complete frozen screen",
        "",
        "Width 384, eight layers, 24 FFN heads of width 16, private hidden width 48; seed 17, context 128, batch 16, native BF16, uniform AdamW and 200 steps. Each trial sees 409,600 sampled training tokens and all 322,688 validation targets. Three trials total 1,228,800 new screen tokens. Official test stays unscored. All twelve earlier control cells are retained with matched per-recipe three-rate tuning; historical architecture search effort is unequal.",
        "",
        "| Rate | Headwise NLL | Calibrated parallel NLL | Headwise peak MiB | Clipped steps | Training tokens/s (descriptive) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, p in zip(rows, parallel):
        lines.append(
            f"| {m['training']['learning_rate']:.4f} | {m['validation_loss']:.6f} | {p['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% | {m['training_tokens_per_second']:,.0f} |"
        )
    lines += [
        "",
        "| Same-rate NLL difference | Full SwiGLU | Full GELU | Calibrated narrow | Plain BlockShuffle |",
        "|---|---:|---:|---:|---:|",
    ]
    for rate, cells in same_rate.items():
        values = " | ".join(f"{cells[r]:+.4f}%" for r in CONTROLS)
        lines.append(f"| {int(rate) / 1e6:.4f} | {values} |")
    lines += [
        "",
        f"Selected NLL is **{selected['validation_loss']:.6f}**: {differences['full_swiglu']:+.4f}% versus full SwiGLU, {differences['full_gelu']:+.4f}% versus full GELU, {differences['calibrated_narrow']:+.4f}% versus calibrated narrow and {differences['blockshuffle']:+.4f}% versus selected plain BlockShuffle. Positive means worse. The 800-step stronger plain result is a separate budget.",
        "",
        f"FFN weights are **2,801,664**, total weights **9,099,648**, and FFN reduction **70.3125%**. Peak is **{selected['peak_allocated_vram_bytes'] / 2**20:.2f} MiB**, {saving:.2f}% below the parallel adaptation's 832.58 MiB. Both full-control memory gates pass. This allocation saving is measured in the native pipeline; no paired speed claim follows from cross-session descriptive throughput.",
        "",
        "| Frozen gate | Decision |",
        "|---|---|",
    ]
    for k, v in decision["gates"].items():
        lines.append(f"| {k.replace('_', ' ')} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        "![All rates, trajectories and allocation](../results/plots/headwise_screen.png)",
        "",
        "## Qualification and verification",
        "",
        f"All **30 earlier variants** retain exact CPU parameters, logits, loss, gradients and FLOP counts. Gaussian first-FFN output RMS is {initial['moments']['headwise']:.9f} versus full {initial['moments']['full']:.9f}, ratio **{initial['calibrated_full_rms_ratio']:.6f}**, within the frozen [0.5,2] interval. The GPU qualification's initial BF16/FP32 logit relative L2 is **{q['precision_relative_l2']:.6f}**. Its ten fixed-batch updates lower pre-update loss from **{q['history'][0]['loss']:.6f}** to **{q['history'][-1]['loss']:.6f}**, with all recorded parameter gradients finite and nonzero. All ten steps clip. Those 20,480 repeated training-token exposures contain no validation targets.",
        "",
        "Only config, factory/initialization and matrix-work counting changed among pre-existing sources. Data, optimizer, trainer, diagnostics and prior architectures remain byte-identical. A fresh GPU worker reproduced all four original control initial NLLs within 1e-7. All three new checkpoint/config counts, source archives, data/optimizer metadata, complete histories and final sampling state verify. Final sampling equals the controls. Every recorded layer diagnostic is finite. The complete qualification and screen finished without retries.",
        "",
        "## Interpretation and next requirement",
        "",
        "Eliminate this local recipe from longer training under the frozen quality rule. Preserve the measured memory benefit and the simpler implementation. It is not a proof of better learning, a new primitive, or evidence against the published architecture at larger head dimensions. The near-equal parallel/headwise NLL does not isolate routing: head width, subnetwork count and optimization geometry also differ.",
        "",
        "The accompanying proof identifies a separate structural property: in fixed head coordinates, a linear combination of head-local nonlinear maps has zero cross-head mixed second derivatives, regardless of scalar activation flexibility. Learned input mixers and stacked layers limit this statement; it is not an observed causal explanation of the NLL gap. Additional activation coefficients have not earned promotion from these results.",
        "",
        "The stronger plain BlockShuffle model remains the practical control. The next evidence requirement is equal-budget optimizer and convergence testing for it and the full/narrow controls, before broader superiority or novelty claims. The corrected affine activation benefit remains 0.101% across three seeds at the same rate.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.headwise_screen_report",
        "```",
        "",
    ]
    Path("research/headwise_screen_results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    displayed = {
        "Full SwiGLU": controls["full_swiglu"],
        "Full GELU": controls["full_gelu"],
        "Narrow": controls["calibrated_narrow"],
        "BlockShuffle": controls["blockshuffle"],
        "Parallel calibrated": parallel,
        "Headwise": rows,
    }
    for label, cells in displayed.items():
        axes[0].plot(RATES, [m["validation_loss"] for m in cells], "o-", label=label)
        m = select_best(cells)
        h = [
            json.loads(s)
            for s in (Path("results/runs") / m["run"] / "history.jsonl").read_text().splitlines()
        ]
        h = [s for s in h if s["step"] >= 50]
        axes[1].plot([s["step"] for s in h], [s["validation_loss"] for s in h], "o-", label=label)
    labels = list(displayed)
    axes[2].barh(
        labels,
        [select_best(v)["peak_allocated_vram_bytes"] / 2**20 for v in displayed.values()],
        color="#277da1",
    )
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Peak training allocation (MiB)", title="Selected recipes; native pipeline")
    axes[0].set(
        xlabel="Peak learning rate",
        ylabel="Final full-validation NLL",
        title="All frozen rates",
        xticks=RATES,
    )
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Full-validation NLL",
        title="Selected rates; steps 50 to 200",
    )
    for ax in axes[:2]:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    for ext in ("png", "svg"):
        fig.savefig(f"results/plots/headwise_screen.{ext}", dpi=150)
    plt.close(fig)
    summary = {
        "status": "QUALITY_FAIL_MEMORY_PASS",
        "selected_nll": selected["validation_loss"],
        "selected_relative_nll_percent": differences,
        "decisions": decision,
        "peak_mib": selected["peak_allocated_vram_bytes"] / 2**20,
        "parallel_peak_reduction_percent": saving,
        "old_variants_exact": 30,
        "all_artifacts_verified": True,
        "plan_sha256": sha256(PLAN),
        "result_sha256": sha256(ROOT / "result.json"),
    }
    write_json(Path("results/headwise_screen_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
