"""Write readable evidence for the selected-seed repeat diagnosis."""

import csv
import gzip
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/training_variability_v1")


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def markdown(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def run() -> None:
    summary, audit, result = [read(name) for name in ("summary.json", "audit.json", "result.json")]
    metrics = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode())
        )
    )
    pairs = markdown(
        [
            "Repetition",
            "Native FP32 NLL",
            "Chunked FP32 NLL",
            "Chunked change",
            "Allocation saved",
            "Update-time ratio",
        ],
        [
            [
                "Original H117" if p["repetition"] == 0 else f"New r{p['repetition']}",
                f"{p['native_nll']:.6f}",
                f"{p['chunk_nll']:.6f}",
                f"{100 * (p['ratio'] - 1):+.4f}%",
                f"{100 * (1 - p['memory_ratio']):.3f}%",
                f"{p['time_ratio']:.4f}",
            ]
            for p in summary["paired"]
        ],
    )
    stats = markdown(
        ["Policy", "Mean NLL", "Median", "Sample variance", "Observed range"],
        [
            [
                name.replace("fp32_default_", ""),
                f"{s['mean']:.6f}",
                f"{s['median']:.6f}",
                f"{s['sample_variance']:.8f}",
                f"{s['min']:.6f}–{s['max']:.6f}",
            ]
            for name, s in summary["groups"].items()
        ],
    )
    resources = markdown(
        ["Origin", "Repeat", "Policy", "Allocated MiB", "Reserved MiB", "Update median ms"],
        [
            [
                r["origin"],
                r["repetition"],
                r["policy"].replace("fp32_default_", ""),
                f"{float(r['peak_job_mib']):.3f}",
                f"{float(r['peak_reserved_mib']):.3f}",
                f"{float(r['wall_update_ms']):.3f}",
            ]
            for r in metrics
        ],
    )
    mapping = {r["label"]: r["policy"] for r in metrics}
    distance_rows = []
    for step in (200, 400, 800):
        for kind in ("native", "chunks", "cross"):
            selected = [
                r
                for r in audit["checkpoint_pairs"]
                if r["step"] == step
                and (
                    (not r["same_policy"])
                    if kind == "cross"
                    else r["same_policy"] and mapping[r["left"]].endswith(kind)
                )
            ]
            distance_rows.append(
                [
                    step,
                    kind,
                    len(selected),
                    *[
                        f"{st.median(r[k] for r in selected):.6f}"
                        for k in ("model", "exp_avg", "exp_avg_sq")
                    ],
                ]
            )
    distances = markdown(
        [
            "Step",
            "Pair type",
            "Pairs",
            "Model distance",
            "First-moment distance",
            "Second-moment distance",
        ],
        distance_rows,
    )
    native, chunks = [summary["groups"][p] for p in ("fp32_default_native", "fp32_default_chunks")]
    max_score = max(
        [audit["initial_native_score"]["maximum_relative_error"]]
        + [s["relative_error"] for r in audit["rows"] for s in r["checkpoint_scores"]]
    )
    max_replay = max(r["replay"]["error"]["global_relative_l2"] for r in audit["rows"])
    same_initial = [r["distance"] for r in audit["initial_gradient_pairs"] if r["same_policy"]]
    cross_initial = [r["distance"] for r in audit["initial_gradient_pairs"] if not r["same_policy"]]
    first_difference = [r["first_different_recorded_training_loss"] for r in summary["divergence"]]
    text = f"""# H118 — repeat variability on the failed WikiText fixture

**Frozen diagnostic verdict: {summary["classification"]}.** All three observed
chunked endpoints are worse than all three native endpoints, but the smallest
cross-comparison gap is **{100 * (chunks["min"] / native["max"] - 1):.3f}%**, below the
predeclared 1% material-disadvantage threshold. The two new paired gaps are
**{100 * (summary["paired"][1]["ratio"] - 1):.3f}% and {100 * (summary["paired"][2]["ratio"] - 1):.3f}%**.
The original H117 quality failure remains failed.

The ranges do not overlap. The largest within-policy range,
{summary["maximum_within_policy_range"]:.6f} NLL, is smaller than the original
pair gap of {summary["paired"][0]["gap"]:.6f}. Thus the observed repeat variability
does not span that entire gap, but these few selected repetitions cannot
uniquely establish its cause or a population-level effect.

![All repetitions and state-distance trajectories](figures/training_variability.png)

## Why this experiment was earned

[H117](fp32_training_replication_results.md) qualified a limited TinyStories
memory component, but its fixed two-corpus claim failed. WikiText seed101 lost
1.55% NLL against BF16 native and 1.35% against FP32 native after 800 updates,
despite tiny initial FP32 gradient error. H118 tests whether that latter gap
repeats beyond ordinary variation between unchanged executions.

The [plan](training_variability_plan.md) selected this failed seed deliberately.
This is **one conditional diagnostic fixture**, not three training seeds, a new
held-out task or an independent sample of model quality. The original two FP32
runs are kept as repetition0. Exactly four additional fresh executions provide
two native and two chunked repetitions, with reversed policy order in repetition2.

Every run loads the same hashed CPU step-zero weights and empty Adam state,
then consumes identical batches. The unchanged H117 common `run_case` function
is imported directly; a temporary output-directory binding is explicitly set
and restored. No copied training loop, altered optimizer, new activation or
backend change is introduced.

The model remains narrow GELU, width384/hidden456/eight layers/six heads,
vocabulary4096, batch8, context512: **9,099,648 total / 2,801,664 FFN parameters**.
All six compared trajectories use FP32 decoder/classifier arithmetic, default
SDPA, whole-block checkpointing, FP32 weights/moments, fixed LR0.0006, AdamW
betas(0.9,0.95), epsilon1e-8, existing matrix-decay groups and norm clip1.
TF32 remains disabled, with four CPU threads on the same RTX4070 Laptop GPU.
The UV-managed Python3.12.9 / PyTorch2.14.0+cu132 environment is retained.

## Every endpoint and observed variation

Lower NLL is better. All full validation scores use the same streamed BF16
scorer at0/200/400/800, independently checked with native unchunked scoring.
No best checkpoint or favorable subset is selected.

{pairs}

{stats}

Each group contains three **repeat executions of one seed**. Sample variance
uses n−1 and is descriptive. The median between-policy gap is
{summary["median_gap"]:.6f} NLL; the maximum within-policy range is
{summary["maximum_within_policy_range"]:.6f}; their ratio is
{summary["gap_over_range"]:.3f}. This ratio is not a significance test.
Native's maximum/minimum NLL differs by {100 * (native["max"] / native["min"] - 1):.3f}%,
and chunks' differs by {100 * (chunks["max"] / chunks["min"] - 1):.3f}%.

The rule fixed before repeats requires `min(chunks) > 1.01 * max(native)` for
REPEATED MATERIAL DISADVANTAGE. It is not satisfied. OVERLAPPING REPEAT VARIATION
also requires overlapping observed intervals, which do not occur. Therefore
the result is INCONCLUSIVE. A consistent ordering or favorable mean cannot
replace those predeclared criteria.

## Memory, runtime and complete accounting

{resources}

New chunked runs preserve **26.816% lower allocated job peaks** versus native
FP32. Their warmed update cost is {summary["paired"][1]["time_ratio"]:.4f}x and
{summary["paired"][2]["time_ratio"]:.4f}x the paired reference. There is no
parameter-count reduction. These are observations on the selected fixture;
they do not independently requalify quality or the full research target.

Allocation includes optimizer/data caches, probe, all updates, evaluation,
diagnostics and serialization. Reserved bytes are reported separately and
driver/context memory remains excluded. All **6,444 new memory intervals**
are recorded before resetting peaks; five study boundaries and six audit
boundaries return zero allocated/reserved tensor storage. All new losses,
gradient norms, final parameters/moments and sampled diagnostics remain finite.

Twenty updates warm up and all780 remaining updates contribute timing. The
timed region excludes batch hashing, scalar reporting and disk I/O. The complete
loop also includes intermediate scoring/checkpointing. New study elapsed time is
**{result["elapsed_seconds"]:.2f} seconds**; audit/initial setup are separate.

New budget: **3,200 updates / 13,107,200 target presentations**, plus four initial
gradient probes =3,204 main backwards. The existing H117 CPU-double qualification
is reused for unchanged code, with zero new qualification backwards. Audit adds
four backward replays: **3,208 new backwards** in total. Original H117's two
compared runs contribute1,600 historical updates and are counted separately.
No scientific run or postprocessing retry was required in H118.

## How trajectories separate

Initial gradient distances within a policy span {min(same_initial):.3e}–{max(same_initial):.3e};
across policies they span {min(cross_initial):.3e}–{max(cross_initial):.3e}.
Distances are global symmetric relative L2. The six within-policy pairs first
differ in their recorded training loss at steps {", ".join(map(str, first_difference))}
(ordered as listed in the compact summary). A recorded scalar loss can remain
equal even when gradients already differ; this is not the first differing bit
of the full state.

The table gives median distances among all unordered checkpoint pairs. Native
and chunked each contribute three within-policy pairs; across policies there
are nine. The figure also shows observed ranges, not confidence intervals.

{distances}

Model and Adam-moment distances show that repeat execution can lead to different
states. They do not identify a specific kernel, establish catastrophic gradients,
or prove why one final validation loss is worse. Initial gradients and every
saved checkpoint remain available for a future mechanism-specific experiment.

## Mathematical limit of the initial-gradient test

Let a training update act on the full state s, including parameters and Adam
moments. If its map F_t is K_t-Lipschitz in a chosen norm, and the alternative
implementation differs from it by at most e_t at each relevant state, then:

```text
delta[t+1] <= K[t] * delta[t] + e[t]
delta[T] <= sum over i=0..T-1 of e[i] * product over j=i+1..T-1 of K[j]
```

The second inequality follows by repeatedly substituting the first, starting
from identical initial state. This elementary bound requires control of every
relevant e_t and K_t. A tiny gradient difference measured only at initialization
does not supply those bounds, and this experiment estimates neither. No theorem
of training equivalence, new stability guarantee or novelty follows.

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
describe the general distinction between mathematical equivalence, seeding and
bitwise execution. They do not explain our particular NLL difference uniquely.

## Audit and decision

The independent audit {"passes" if audit["passed"] else "fails"}. It regenerates the shared initial state,
verifies all12 new trained model/Adam/sampler checkpoints, reproduces all3,200
new batches, checks six stored initial-gradient hashes (four new, two original),
and computes13 native full-validation scores. Maximum streamed/native relative
score error is {max_score:.3e}. All four new initial FP32 gradient replays pass;
maximum global relative L2 is {max_replay:.3e}. Thresholds remain1e-6 for score/loss,
1e-5 global and1e-4 per tensor for replay. Fifteen initial-gradient pairs and45
checkpoint pairs are recorded. No BF16 gradient or full training-trajectory
bitwise reproducibility is claimed.

Keep the classification **INCONCLUSIVE** and retain the consistent observed
ordering as a diagnostic finding. H117's fixed WikiText/two-corpus recipe stays
rejected, while its scoped TinyStories result and resource observations remain.
No extra unchanged800-update repeats, new default, activation or parameter
reduction are earned here. Further work should test a distinct mechanism locally
before allocating another broad language sweep. The full research goal remains open.

One concrete next question is whether Adam's coordinate-wise normalization
amplifies the small differences already present in the saved initial gradients.
Those six gradient sets allow a bounded first-update comparison without new
language training. Such a diagnostic would still need its own frozen plan and
would not, by itself, explain the full 800-update endpoint or qualify a new recipe.

## Artifacts

[Protocol](../results/training_variability_v1/protocol.json),
[audit protocol](../results/training_variability_v1/audit_protocol.json),
[summary](../results/training_variability_v1/summary.json),
[source guide](../results/training_variability_v1/source/README.md),
[verification receipt](../results/verification/training_variability_final_v1.json).
Compressed metric/curve/update tables include all six compared trajectories;
their 4,800 update rows are explicitly marked H117 or H118. New allocation is
only3,200 updates. Raw per-run histories/tensors remain local, ignored and intact.

All137 frozen source/plan hashes and61 maintained hashes are preserved. The
maintained tree stays at two model folders, five variants and eight recipes.
The prior116-test suite result is historical; this round does not rerun that
unchanged suite. H117's startup and plotting failures remain preserved, and
successful H118 execution does not establish a Windows native-runtime cure.
"""
    # Prose-only spacing, with no changes to equations, results or file paths.
    import re

    text = re.sub(
        r"\b(at|all|All|and|remaining|only|unchanged|contribute|computes|are|prior)(?=\d)",
        r"\1 ",
        text,
    )
    text = text.replace("width384/hidden456", "width 384 / hidden 456").replace(
        "vocabulary4096, batch8, context512", "vocabulary 4096, batch 8, context 512"
    )
    text = (
        text.replace("LR0.0006", "LR 0.0006")
        .replace("betas(0.9,0.95)", "betas (0.9, 0.95)")
        .replace("epsilon1e-8", "epsilon 1e-8")
        .replace("clip1", "clip 1")
    )
    text = (
        text.replace("RTX4070", "RTX 4070")
        .replace("Python3.12.9", "Python 3.12.9")
        .replace("PyTorch2.14.0", "PyTorch 2.14.0")
        .replace("remain1e-6", "remain 1e-6")
    )
    Path("research/training_variability_results.md").write_text(
        text, encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            dict(
                classification=summary["classification"],
                max_native_score_error=max_score,
                max_gradient_replay_error=max_replay,
                first_different_losses=first_difference,
                smallest_cross_gap_percent=100 * (chunks["min"] / native["max"] - 1),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
