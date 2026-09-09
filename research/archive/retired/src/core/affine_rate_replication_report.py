"""Correct the activation comparison using a three-seed plain same-rate control."""

import hashlib
import json
import statistics
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.core.affine_longer import CRITICAL
from src.core.affine_longer import output_path as original_path
from src.core.affine_rate_replication import (
    PLAN,
    SEEDS,
    load_references,
    output_path,
    summarize,
    verify_plain,
)
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/affine_rate_replication_v1")


def report():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 2
    assert result["plan_sha256"] == protocol["plan_sha256"] == sha256(PLAN)
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    initial = json.loads(Path("results/affine_longer_v2/preflight.json").read_text())
    rows, rngs = load_references(initial)
    plain, hashes = {}, {}
    old17 = json.loads(Path("results/affine_longer_v2/result.json").read_text())
    old_replicated = json.loads(Path("results/affine_replication_v1/result.json").read_text())
    for seed in SEEDS:
        for recipe in rows[seed]:
            old = next(
                t
                for t in (old17 if seed == 17 else old_replicated)["trials"]
                if t["recipe"] == recipe and (seed == 17 or t["seed"] == seed)
            )
            p = original_path(recipe, seed)
            assert sha256(p / "metrics.json") == old["metrics_sha256"]
            assert sha256(p / "checkpoint.pt") == old["checkpoint_sha256"]
        m, rng = verify_plain(seed, initial, rows[seed]["blockshuffle"])
        plain[seed] = m
        assert torch.equal(rng, rngs[seed])
        trial = (
            protocol["seed17"]
            if seed == 17
            else next(t for t in result["trials"] if t["seed"] == seed)
        )
        p = output_path(seed)
        assert sha256(p / "metrics.json") == trial["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == trial["checkpoint_sha256"]
        if seed != 17:
            assert all(
                m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                for n in (*CRITICAL, "src/core/affine_rate_replication.py")
            )
        hashes[p.name] = {
            "metrics": sha256(p / "metrics.json"),
            "checkpoint": sha256(p / "checkpoint.pt"),
            "source_archive": sha256(p / "source.zip"),
        }
    summary = summarize(rows, plain)
    assert json.loads(json.dumps(summary)) == {k: result[k] for k in summary}
    write_json(
        Path("results/affine_rate_replication_summary.json"),
        {
            **summary,
            "source_hashes": hashes,
            "all_original_fifteen_reference_hashes_verified": True,
        },
    )
    lines = [
        "# Corrected three-seed activation comparison at the same rate",
        "",
        f"**Material activation benefit: {'PASS' if summary['material_activation_benefit'] else 'FAIL'}. Stronger plain quality/memory gates: {'PASS in all three seeds' if summary['plain_passes_all_seeds'] else 'FAIL'}.** At the common peak LR 0.0012, plain BlockShuffle mean NLL is {summary['plain_mean_nll']:.6f} +/- {summary['plain_sample_sd_nll']:.6f}; affine is {summary['affine_mean_nll']:.6f} +/- {summary['affine_sample_sd_nll']:.6f} (sample SD). Affine changes mean NLL by {summary['relative_mean_affine_change_percent']:+.4f}% relative to plain.",
        "",
        "This corrects the interpretation of the earlier 1.755% selected-recipe advantage, which compared plain LR 0.0006 with affine LR 0.0012. Those historical measurements remain valid, but do not isolate an activation improvement. [H043](affine_activation_rate_control_results.md) showed only 0.0193% affine benefit at the higher rate in seed 17 and 0.337% at the lower rate.",
        "",
        "[Frozen corrective replication](affine_rate_replication_plan.md), [raw results](../results/affine_rate_replication_v1/result.json), and [historical selected-recipe replication](affine_activation_replication_results.md).",
        "",
        "## Every same-rate pair",
        "",
        "| Seed | Plain NLL | Affine NLL | Affine minus plain | Relative affine change |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        b, a = plain[seed]["validation_loss"], rows[seed]["blockshuffle_affine"]["validation_loss"]
        lines.append(f"| {seed} | {b:.9f} | {a:.9f} | {a - b:+.9f} | {100 * (a / b - 1):+.4f}% |")
    paired = summary["paired_statistics"]
    lines += [
        "",
        f"Paired mean difference {paired['mean']:+.9f}; exploratory 95% t interval [{paired['student_t_95_interval'][0]:+.9f}, {paired['student_t_95_interval'][1]:+.9f}]; {paired['strict_wins']}/3 strict affine wins; exact one-sided sign p={paired['one_sided_sign_p']:.3f}. The interval assumes approximately normal independent differences, which three observations cannot establish. Seed17 motivated the corrected control. These are exploratory summaries, not confirmatory population claims.",
        "",
        "![Three-seed same-rate comparison](../results/plots/affine_rate_replication.png)",
        "",
        "## Stronger plain control against full and narrow references",
        "",
        "| Recipe | Mean NLL +/- sample SD | FFN weights | Maximum training peak MiB | Mean serial training tokens/s |",
        "|---|---:|---:|---:|---:|",
    ]
    cohorts = {
        r: [rows[s][r] for s in SEEDS] for r in ("full_swiglu", "full_gelu", "calibrated_narrow")
    }
    cohorts["plain_blockshuffle_lr1200"] = [plain[s] for s in SEEDS]
    cohorts["affine_blockshuffle_lr1200"] = [rows[s]["blockshuffle_affine"] for s in SEEDS]
    for name, values in cohorts.items():
        losses = [v["validation_loss"] for v in values]
        lines.append(
            f"| {name} | {statistics.mean(losses):.6f} +/- {statistics.stdev(losses):.6f} | {values[0]['ffn_parameters']} | {max(v['peak_allocated_vram_bytes'] for v in values) / 2**20:.2f} | {statistics.mean(v['training_tokens_per_second'] for v in values):.0f} |"
        )
    lines += [
        "",
        "Plain relative NLL change against the reference means: "
        + "; ".join(
            f"{r} {v:+.4f}%"
            for r, v in summary["plain_relative_to_reference_means_percent"].items()
        )
        + ".",
        "",
        "Plain uses 2,801,664 FFN weights, 9,099,648 total, a 70.3125% FFN reduction. Affine adds 128 weights. For EACH seed, the plain gate checks >=70% FFN reduction, <=1% NLL cost against both full controls, beating calibrated narrow and peak allocation <=1.1 times both full controls. Per-seed decisions remain in the raw result. Serial throughput is descriptive and is not a paired speed measurement.",
        "",
        "## Decision and scope",
        "",
    ]
    if summary["material_activation_benefit"]:
        lines += [
            "The predeclared mean >=0.2% and three-strict-wins requirements pass. This supports further same-execution and gain/bias controls. It still does not isolate activation learning, because the two recipes use different gate-backward implementations."
        ]
    else:
        lines += [
            "The activation does not meet the predeclared >=0.2% mean improvement plus three strict wins. Do not promote more activation variants or TinyStories transfer using the confounded historical advantage. Keep the learned families, scoped expressivity proof and all positive/negative measurements as research artifacts. The stronger plain recipe is the simpler local control."
        ]
    lines += [
        "",
        "The two new runs differ from each seed's original plain recipe only in peak LR. Actual initial NLL, optimizer groups, data hashes, critical computation, histories, diagnostics, checkpoint/config metadata and final sampling RNG are verified. All fifteen original reference checkpoint/metrics hashes match their archived H039/H041 records. Each trial trains 1,638,400 sampled tokens and scores 322,688 validation targets. Official test remains unscored.",
        "",
        "Plain still uses the custom native SiLU/product backward; affine uses checkpointed native autograd. Same-rate results therefore compare these complete recipes. The successful removal of affine corrections after training remains a valid checkpoint intervention, with no causal or deployment-speed inference. Existing fused-serving speed was measured on earlier TinyStories checkpoints, not these WikiText weights.",
        "",
        "The broad goal remains open: equal extended tuning, convergence, stronger published comparators, trained scale, new mechanisms and full-network stability. A better learning rate for an established structured layer is useful engineering, not a new activation primitive.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.affine_rate_replication_report",
        "```",
    ]
    Path("research/affine_rate_replication_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    b = [plain[s]["validation_loss"] for s in SEEDS]
    a = [rows[s]["blockshuffle_affine"]["validation_loss"] for s in SEEDS]
    axes[0].plot(SEEDS, b, "o-", color="#277da1", label="Plain; LR0.0012")
    axes[0].plot(SEEDS, a, "o-", color="#20754e", label="Affine; LR0.0012")
    axes[0].set(
        xticks=SEEDS, xlabel="Seed", ylabel="Full-validation NLL", title="Corrected comparison"
    )
    old = [
        rows[s]["blockshuffle_affine"]["validation_loss"]
        - rows[s]["blockshuffle"]["validation_loss"]
        for s in SEEDS
    ]
    new = [x - y for x, y in zip(a, b)]
    for i, (values, color) in enumerate(((old, "#888888"), (new, "#20754e"))):
        axes[1].scatter([i - 0.06, i, i + 0.06], values, color=color, s=35)
        axes[1].plot([i - 0.14, i + 0.14], [statistics.mean(values)] * 2, color="black", lw=2)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=[
            "Earlier selected rates\nplain0.0006 / affine0.0012",
            "Common rate\nboth0.0012",
        ],
        ylabel="Affine minus plain NLL",
        title="Three pairs and mean; lower favors affine",
    )
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    for extension in ("png", "svg"):
        fig.savefig(f"results/plots/affine_rate_replication.{extension}", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
