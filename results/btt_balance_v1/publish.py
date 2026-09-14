"""Publish all H164 seeds and gates; no selection from report-set outcomes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results.btt_balance_v1.model import ARMS
from results.btt_balance_v1.study import ROOT, read, sha, verify, write

verify()
for stage in ("run", "audit"):
    assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
r, s, a = [read(ROOT / name) for name in ("result.json", "summary.json", "audit.json")]
assert a["passed"] and a["independent_scores"] == 168
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
colors = plt.get_cmap("tab10").colors
rows = []
for i, task in enumerate(("teacher", "product")):
    ax = axes[0, i]
    for j, arm in enumerate(ARMS):
        vals = [v["report"] for v in r["selected"] if v["task"] == task and v["arm"] == arm]
        ax.errorbar(
            j, np.mean(vals), yerr=np.std(vals, ddof=1), fmt="o", color=colors[j], capsize=4
        )
        ax.scatter([j - 0.1, j, j + 0.1], vals, s=15, color=colors[j], alpha=0.7)
        stats = s["statistics"][task][arm]
        selected = [v for v in r["selected"] if v["task"] == task and v["arm"] == arm]
        profiles = [v for v in r["profiles"] if v["task"] == task and v["arm"] == arm]
        rows.append(
            f"| {task} | {arm} | {selected[0]['parameters']} | {stats['mean']:.5f} | {stats['median']:.5f} | {stats['variance']:.3g} | {np.mean([p['memory']['allocated'] / 2**20 for p in profiles]):.2f} | {np.mean([p['median_ms'] for p in profiles]):.3f} |"
        )
    ax.set_xticks(range(len(ARMS)), ARMS, rotation=35, ha="right")
    ax.set_ylabel("Held-out normalized MSE")
    ax.set_title(task + " — mean ± sample SD, all seeds")
    ax.grid(axis="y", alpha=0.2)
    ax = axes[1, i]
    for j, arm in enumerate(ARMS):
        selected = [v for v in r["selected"] if v["task"] == task and v["arm"] == arm]
        profiles = [v for v in r["profiles"] if v["task"] == task and v["arm"] == arm]
        ax.scatter(
            [p["memory"]["allocated"] / 2**20 for p in profiles],
            [v["report"] for v in selected],
            label=arm,
            color=colors[j],
            s=25,
        )
    ax.set_xlabel("Local training GPU peak (MiB), batch8192")
    ax.set_ylabel("Held-out normalized MSE")
    ax.grid(alpha=0.2)
    if i == 1:
        ax.legend(fontsize=8, loc="best")
fig.suptitle("H164: BTT factor choice — synthetic learning screen", fontsize=16)
figure = Path("research/figures/btt_balance.png")
fig.savefig(figure, dpi=150)
plt.close(fig)
gates = []
for task, pairs in s["decisions"].items():
    for pair in pairs:
        ratios = pair["ratios"]
        failed = ", ".join(k for k, v in pair["gates"].items() if not v) or "none"
        gates.append(
            f"| {task} | {pair['seed']} | {ratios['wide_mse']:.3f} | {ratios['narrow_mse']:.3f} | {ratios['greedy_mse']:.3f} | {ratios['memory']:.3f} | {ratios['cuda']:.3f} | {failed} |"
        )
status = "PILOT PROMOTION GATES PASS" if s["promotion"] else "NO PROMOTION"
text = f"""# H164: balanced versus greedy BTT factors

**{status}.**84 fits/50,400 learning updates,42 separate resource profiles/840
updates. All168 independently recomputed CPU FP64 scores and six regenerated
datasets verified. Task qualification:{s["qualified_tasks"]}. Balanced-factor
shape hypothesis passes all planned task conditions:{s["shape_passed"]}.

![All-seed quality and local memory](figures/btt_balance.png)

## Hypothesis and implementation

A64->152 up projection has an intermediate32 in rank1 greedy BTT, imposing a
rank<=32 linear bottleneck before GELU. Closest-factor BTT has intermediate64,
which removes this particular bottleneck; its down intermediate is152 instead
of304. Full attainable matrix rank does not guarantee arbitrary dense weights
or good nonlinear learning. BTT is established prior work; this is a shape-choice
ablation prompted by the retained official-code comparison, not a new activation.
[Qiu et al., ICML2024](https://arxiv.org/abs/2406.06248).

Two normalized cores per projection use the official Gaussian muP scale and
per-core LR multipliers; two learned gains and biases are counted. The forward
uses native batched GEMMs without constructing dense weights. Dense uses fan-in
Gaussian initialization; BlockShuffle keeps the repository's orthogonal init
and analogous per-core LR multipliers. This is not a reproduction of upstream
optimized kernels or its entire training recipe. Float64 gradcheck and explicit
dense-weight comparisons pass for the normalized BTT primitive.

Balanced has4,380 parameters versus wide19,672 (77.73% fewer). Greedy has5,340;
BlockShuffle3,672. The corresponding narrow widths33/40/27 have4,321/5,224/3,547.
Slightly lower matched-control budgets are explicit; these are no-bias-in-core
models with a bias after each projection and GELU between projections.

## Learning and resources

Two synthetic tasks: Gaussian dense GELU teacher and cyclic coordinate products.
Three independently generated teacher/data/optimization seeds431/443/457,
4,096training/1,024validation/1,024report examples. Output normalization uses only
training mean and scalar RMS. Each arm receives the same two base rates(.001,.003),
600AdamW updates per rate, batch256, clip1, no weight decay. Selection uses only
validation at200/400/600, including LR selection. FP32/TF32off, four CPU threads,
one RTX4070 Laptop GPU. Equal search steps do not mean equal FLOPs.

| Task | Arm | Parameters | Mean MSE | Median MSE | Sample variance | Mean local peak MiB | Mean median update ms |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Wide controls must reach normalized reportMSE<=.25 on teacher and<=.50 on product
for each seed, otherwise that task is inconclusive for promotion. Balanced must
then stay<=1.05wide MSE and<=.95narrow33 MSE, save>=25%parameters, save>=10%local
allocated GPU peak, and have median CUDA and wall updates<=1.15wide on every seed.
The separate shape hypothesis requires balanced/greedy meanMSE<=.95 on both
qualified tasks and no seed ratio>1.05. Neither aggregate nor a favorable task
rescues a failing promotion gate.

| Task | Seed | MSE / wide | MSE / narrow33 | MSE / greedy | GPU / wide | CUDA / wide | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(gates)}

Resource profiles restore each selected model, then execute20 completeAdamW
updates on the same fixed8,192-example random input/target batch, discarding5
warmups only for timing. Peak allocation includes resident inputs/targets,
weights, gradients, optimizer moments and eager temporaries. These are local FNN
training profiles; they do not establish Transformer whole-job savings or
inference latency. Profiling never alters selected checkpoints. The figure's
error bars are sample SD, not confidence intervals; all three seeds are visible.

## Audit, failures and decision

The independent CPU FP64 audit constructs explicit BTT dense matrices and uses
native GELU, instead of its training contraction, to verify all84 selected-rate
states' validation/report scores within1e-5 relative tolerance. BlockShuffle dense
maps are obtained from basis images. All selected and final state hashes are
checked, data are regenerated exactly, and validation-only selection is verified.
The audit performs no optimizer updates. It does not independently replay every
learning step or establish terminal convergence.

The first CPU preflight command exited1 without a captured Python diagnostic;
one unchanged-code retry with a fresh cache prefix passed before protocol freeze.
Its cause is unknown. The note and retry log are retained. GPU training and
profiling completed once under the fixed protocol; no failed seed was replaced.

{"The pilot earns a separate larger or real-data test; do not treat this synthetic result as a breakthrough." if s["promotion"] else "Do not insert balanced BTT into a language model or spend on kernel optimization based on this pilot. Preserve the failed gates and the control qualification status. A favorable factor-shape comparison alone does not establish replacing dense width."}
The broad VRAM/quality and richer-neuron objective remains open.

[Prospective plan](btt_balance_plan.md), [raw summary](../results/btt_balance_v1/summary.json),
[CPU audit](../results/btt_balance_v1/audit.json), [official comparison](btt_official_comparison.md).
"""
report = Path("research/btt_balance_results.md")
assert not report.exists()
report.write_text(text, encoding="utf-8")
for target, title in (
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
):
    old = target.read_text(encoding="utf-8")
    snapshot = ROOT / (
        "README.before.md" if target.name == "README.md" else "CURRENT_STATE.before.md"
    )
    with snapshot.open("x", encoding="utf-8") as stream:
        stream.write(old)
    assert old.startswith(title)
    link = (
        "research/btt_balance_results.md"
        if target.name == "README.md"
        else "btt_balance_results.md"
    )
    rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    target.write_text(
        title
        + f"\n\nLatest: [H164 BTT factor-shape pilot]({link}): **{status}.**\n84 fits, 42 local-resource profiles; independent CPU scores verified.\nThis is prior-art architecture screening; the broad goal remains open.\n\n"
        + rest,
        encoding="utf-8",
    )
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and not any("cache" in p for p in f.parts)
    and f.name not in ("receipt.json", "publish.log", "publish_exit.txt")
]
files += [
    report,
    figure,
    Path("research/btt_balance_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write(
    ROOT / "receipt.json",
    dict(
        status=status,
        promotion=s["promotion"],
        updates=51240,
        goal_achieved=False,
        hashes={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
