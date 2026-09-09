"""H080 report and the user-requested progress overview from audited evidence."""

import os
from pathlib import Path

from results.blast_operator_recovery_v1.source.storage import read

ROOT = Path("results/blast_operator_recovery_v1")
a = read("results/verification/blast_operator_recovery_analysis_v2.json")
assert a["status"] == "PASS" and a["qualified_checks"] == 34
assert a["checkpoint_tensor_pairs_exact"] == 114
assert all(a["prior_readable_tensor_payload_equality"].values())
assert read(ROOT / "analysis_v2_process.json")["status"] == "PASS"


def write(path, text, new=False):
    payload = text.encode()
    with Path(path).open("xb" if new else "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    assert Path(path).read_bytes() == payload


report = f"""# H080 - BLAST operator recovery and qualification

**LOCALLY QUALIFIED; learning remains untested.** All 34 frozen checks pass in
the one explicit recovery. All 32 tensor files verify independently. No
optimizer update, corpus target or full Transformer run was allocated.

BLAST is published prior art. Our comparator uses rank-batched native group
mixing, a local orthogonal initialization and the retained BlockShuffle outer
permutations. This does not reproduce the paper's complete training recipe or
establish a new architecture. [BLAST source](https://arxiv.org/html/2410.21262v1)

| FFN | Hidden width | Projections | FFN weights, 8 layers | Total weights by count |
|---|---:|---:|---:|---:|
| BLAST GELU | 3,200 | 2 | 2,801,664 | 9,099,648 |
| BLAST SwiGLU | 1,984 | 3 | 2,801,664 | 9,099,648 |

Both have 350,208 weights per FFN: 70.3125% FFN and 42.1700% total reduction.
Transformer totals are arithmetic, not executed models. No training time,
memory or quality claim follows.

## Numerical and mathematical evidence

| Check | Verified result |
|---|---|
| Frozen functions, seeds, shapes, precision and thresholds | Unchanged |
| FP64 projection comparisons | 12 cases; 60 output/gradient comparisons |
| Maximum projection comparison error | {a['projection_max_absolute_error']:.3e} |
| Independently derived gradient maximum error | {a['analytic_vjp_max_absolute_error']:.3e} |
| Eager/checkpoint comparisons | 6 CPU FP32 + 6 CUDA BF16 cases; all 114 tensor pairs bitwise equal |
| Independent initializer Gram maximum error | {a['independent_initializer_gram_max_error']:.3e} |
| Finite differences | PASS; 10 forward calls, zero updates |
| Exact same-width BlockShuffle embeddings | Four projection shapes |
| Hadamard/rank-collapse certificates | Four projection shapes |
| Readable original tensor payloads | All 24 match the recovery exactly |

The [theory](blast_operator_theory.md) proves same-width inclusion with an
explicit Latin-square mapping, costing 3,072 extra coupling weights/projection.
Equal-budget FFNs reduce hidden width by 64, so this does not prove equal-budget
nonlinear-family inclusion. The fixed-partition rank-6 matrix approximation
has squared relative error at least 7/8 on the Hadamard witness. Neither statement
is a learning result or novelty claim.

The initial projections have controlled nonzero singular values. Down retains
a nullspace; activations, learned factors and depth can still have small
Jacobians. No full-network nonvanishing-gradient or convergence guarantee follows.

## Preserved failures and recovery scope

[H079](blast_operator_results.md) records a process pass but incomplete saved
evidence: 25 entirely zero-filled files, 18/34 parseable observation JSONs and
24/32 intact tensor archives. Cause is unknown; original bytes remain unchanged.

H080 repeats the same checks once in a new root. Only the artifact destination
and writer change: flush, fsync, ZIP/hash checks and exact payload readback.
This verifies present artifacts without diagnosing or guaranteeing removal of
the original cause. There are two qualification attempts across H079/H080.
The recovery process takes 15.680 seconds; pytest reports 14.26 seconds.

Audit v1 incorrectly required a CPU-recomputed norm to equal the recorded GPU
norm bitwise. Seventeen scalars differ by at most 6.651e-16 relatively. Audit v2
checks each norm on its original device, retaining exact equality. All numerical
thresholds remain unchanged; its 57 CUDA reductions are metadata checks with
zero model forwards. The failed audit and a pre-execution report-writer syntax
error are preserved separately; neither triggers a qualification/training repeat.

Only a separately frozen learning comparison is earned, with explicit optimizer
calibration and equal tuning against full, calibrated narrow and plain controls.
Full-model resources, language quality, long-budget seeds, convergence, scale,
broader data and published comparisons remain open. H078 stays closed. Active
source remains three model folders, six variants, nine recipes and the unchanged
previously passing 108-test suite.

[Recovery plan](blast_operator_recovery_plan.md),
[measurements](../results/blast_operator_recovery_v1/result.json),
[independent audit](../results/verification/blast_operator_recovery_analysis_v2.json),
[correction record](../results/blast_operator_recovery_v1/analysis_correction.json),
[final preservation audit](../results/verification/blast_operator_recovery_final_v1.json).
"""
write("research/blast_operator_recovery_results.md", report, True)

overview = """# Project progress - 2026-09-09

We have working parameter compression, but the complete research target remains
unmet: at least 70% fewer FFN weights, within 1% relative validation loss of BOTH
full GELU and SwiGLU, while beating calibrated narrow controls. Consistent
multi-seed, longer-training, scale and broader-data evidence is still required.

## Latest language results

Six fresh WikiText-2 runs, seed 17, 3,200 updates. NLL is validation loss:
lower is better; these values are not accuracy percentages.

| Model | Final NLL | FFN weights |
|---|---:|---:|
| Full SwiGLU | 4.1054 | 9,437,184 |
| Full GELU | 4.1279 | 9,437,184 |
| Narrow SwiGLU | 4.1530 | 2,801,664 |
| Narrow GELU | 4.2058 | 2,801,664 |
| BlockShuffle SwiGLU | 4.1443 | 2,801,664 |
| BlockShuffle GELU | 4.2420 | 2,801,664 |

Plain BlockShuffle is the best compressed form in this latest one-seed cohort.
Its earlier three-seed 3,200-step study averaged 1.256% worse NLL than full
SwiGLU, missing our 1% limit. The newer GELU passes a three-seed 800-step
comparison but fails the longer test: +3.327% versus full SwiGLU, +2.765% versus
full GELU, and worse than both narrows. Its fixed recipe is closed. Schedules,
execution and optimizer differences define separate experiments.
[Latest report](ungated_duration_results.md),
[earlier plain replication](long_duration_replication_results.md).

## Data, batches and duration

- Main corpus: WikiText-2 raw v1; train-only 4,096-token BPE vocabulary.
- Cached data: 3,083,650 training tokens and 322,802 validation tokens.
- Each training batch has 16 sequence windows of 128 input/next-token targets:
  2,048 predicted training targets per optimizer update.
- Each 3,200-update run presents 6,553,600 sampled training targets, about 2.13
  cache-token exposures. Sampling is with replacement, not ordered epochs.
- Full validation scores 322,688 targets in 158 batches; the last has nine windows.
- Earlier stages used a small TinyStories subset and synthetic function fitting.
  The cited activation fitting screen uses batch 256 and MSE, not language NLL.

Budgets progressed from 200-update screens to 800-update comparisons (including
seeds 17/29/43), then 3,200-update duration tests. The latest six-run cohort took
about 34 minutes: September 8, 10:10:41 to 10:45:06 UTC. All six models still
improved during the last 800 updates; convergence is unestablished. Reused
validation is development data, and the official test split remains unscored.

The project began September 6. At the September 9 status question, the goal
tracker recorded about 28.3 hours of accumulated agent work: reading, coding,
audits and waiting as well as experiments. This is not GPU-training time.

## Learnable activations were implemented and trained

| Tested family | Result |
|---|---|
| Shifted Bezier/quadratic | Selected BlockShuffle short-screen loss worsened 0.442%; no promotion |
| Affine correction, 128 extra weights | Only 0.101% lower mean NLL at matched LR, winning 2/3 seeds; missed material-benefit gate |
| Group-shared rational, 320 extra weights | One-seed 200-update NLL 5.8980, 1.216% below its selected base; 883.17 MiB native peak fails memory |
| Static/input-conditioned corrections | Synthetic fitting gains only 0.0585% / 0.1213% over plain; missed the frozen 2% gate |

Rational is a short-screen signal with separately selected rates, not a replicated
activation-specific advantage. Resetting its learned shape barely changes final
loss. The earlier 1.755% affine gain was rate-confounded; the corrected result is
0.101%. No tested activation has a dependable combined quality/resource win.
[Activation guide](learnable_activation_domain.md),
[rational/Bezier results](learnable_activation_results.md),
[corrected affine comparison](affine_rate_replication_results.md),
[input-conditioned fitting](token_activation_fit_results.md).

## Current work

The published BLAST factorization is locally qualified as a comparator at the
same FFN budget: 34 checks, including 114 exact eager/checkpoint tensor pairs.
The original attempt's damaged artifacts remain preserved; one explicit recovery
and independent audit pass. No BLAST language training has run. Next is a
separately frozen learning comparison with explicit optimizer calibration.
[Qualification and limits](blast_operator_recovery_results.md).
"""
write("research/PROGRESS_OVERVIEW.md", overview, True)

path = Path("README.md")
s = path.read_text(encoding="utf-8")
needle = "**The research goal remains unmet. Matched-budget BlockShuffle GELU fails\nthe fixed 3,200-step test.**"
assert needle in s
s = s.replace(needle, """**The research goal remains unmet.** The [progress overview](research/PROGRESS_OVERVIEW.md)
explains results, datasets, batch sizes, duration and learnable activations.

The [BLAST comparator](research/blast_operator_recovery_results.md) passes local
operator qualification after one documented artifact recovery. Learning quality
remains untested and the active model tree is unchanged.

**Latest language result: matched-budget BlockShuffle GELU fails the fixed
3,200-step test.**""", 1)
write(path, s)

path = Path("research/CURRENT_STATE.md")
s = path.read_text(encoding="utf-8")
s = s.replace("The latest completed experiment is matched-budget", "The latest completed language experiment is matched-budget", 1)
needle = "## Latest complete comparison"
assert needle in s
s = s.replace(needle, """## Latest operator qualification

[H080](blast_operator_recovery_results.md) qualifies a conventional BLAST comparator:
GELU h3200 and SwiGLU h1984 each have 2,801,664 FFN / 9,099,648 total weights by
count. All 34 checks pass; independent analysis verifies 32 tensor files,
114 exact checkpoint pairs, analytic derivatives, four exact same-width embeddings
and four matrix witnesses. Learning and full-model resources remain untested.

[H079](blast_operator_results.md) remains incomplete because of damaged artifacts.
One explicit recovery preserves numerical tests and adds durable readback; all
24 readable original tensor payloads match. A separate scalar-norm audit mismatch
is corrected on the original devices without changing thresholds. There are zero
optimizer updates, corpus targets or Transformer runs. Only a separately frozen
learning comparison is earned; no active model is added.

For data, batches, duration and learned-activation results, see the
[plain-language progress overview](PROGRESS_OVERVIEW.md).

## Latest complete language comparison""", 1)
s = s.replace("an unallocated count-compatible BLAST budget", "the original count-compatible BLAST budget")
s += """
The [H080 final audit](../results/verification/blast_operator_recovery_final_v1.json)
checks 118 scientific sources, 69 frozen plans and the unchanged original H079
damaged/intact files alongside earlier language, fitting and resource evidence.
Its 34 isolated checks do not replace the unchanged 108-test active suite.
"""
write(path, s)

path = Path("research/idea_bank.md")
s = path.read_text(encoding="utf-8")
assert "## H079 -" not in s and "## H080 -" not in s
s += """
## H079 - BLAST qualification (NEEDS_INVESTIGATION; saved-artifact integrity failed)

Original process reports PASS in 13.826 seconds, but 25 files are zero-filled;
only 18/34 observation JSONs parse and 24/32 tensor archives pass integrity.
Source/protocol/archive and H078 evidence remain intact. Cause is unknown; original
bytes are preserved and qualification remains incomplete. No optimizer update,
corpus target or Transformer was allocated. [Integrity report](blast_operator_results.md).

## H080 - Explicit BLAST recovery (LOCALLY QUALIFIED; learning untested)

One frozen recovery repeats unchanged operator/test bodies, seeds, shapes,
precision and thresholds. Only artifact location and durable writing change.
All 34 checks and 32 saved tensor files pass; 114 eager/checkpoint tensor pairs
are exact, and 60 FP64 projection comparisons meet the original tolerances.
Independent analytic-gradient error is at most 8.882e-15; Gram error 1.998e-15.
Finite differences use ten forwards, zero updates.

GELU h3200 and SwiGLU h1984 each have 350,208 weights/FFN, 2,801,664 across eight
layers and 9,099,648 total by count. Four same-width embeddings and four
Hadamard/rank-collapse certificates pass. The local rank-6 matrix squared-error
bound is 7/8; same-width inclusion costs 3,072 extra weights/projection.
Reducing hidden width by 64 pays the budget but prevents an equal-budget nonlinear
containment inference. Initial partial isometry is not a trained-network guarantee.

All 24 readable original tensor payloads match exactly. H079's damaged evidence
remains unchanged. A failed CPU/GPU scalar-norm audit is corrected by checking on
the original device, keeping exact equality. Its failure and a pre-execution
report-writer syntax error are preserved. Neither changes a numerical threshold
or repeats qualification/training. Across both stages: two qualification attempts,
one explicit repetition, zero optimizer updates/corpus targets/Transformers.

BLAST is prior art with local permutation/initialization adaptations. Only a
separately frozen learning comparison with explicit optimizer calibration and
equal tuning budgets is earned. Active code remains three folders, six variants,
nine recipes. [Complete report](blast_operator_recovery_results.md).
"""
write(path, s)
print("PASS: report, overview and three navigation documents updated.")

