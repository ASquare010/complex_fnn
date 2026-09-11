"""Publish audited H118 navigation, preserving exact previous document snapshots."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/training_variability_v1")
protocol = json.loads((ROOT / "protocol.json").read_text())
summary = json.loads((ROOT / "summary.json").read_text())
audit = json.loads((ROOT / "audit.json").read_text())
assert summary["classification"] == "INCONCLUSIVE" and audit["passed"]
assert (ROOT / "plot_exit.txt").read_text().strip() == "0"
assert Path("research/training_variability_results.md").exists()
before = {}
for name, info in protocol["before_documents"].items():
    data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == info["sha256"], name
    assert data == Path(info["preserved_path"]).read_bytes(), name
    before[name] = data.decode("utf-8").replace("\r\n", "\n")

overview = """## Latest: repeated execution exposes material trajectory variation

[H118](training_variability_results.md) adds four 800-update executions of the
failed WikiText seed101, using the same initial state, batches and unchanged
H117 runner. New chunk/native NLL gaps are **+1.143% and +0.940%**, versus the
original +1.352%. All three chunked endpoints are worse than all three native
endpoints, but the smallest cross-comparison gap is only0.630%, below the fixed
1% rule. The preset diagnostic verdict is **INCONCLUSIVE**. These are repeats
of one deliberately selected seed, not three independent training seeds.

The native NLL range is0.035284, smaller than the original0.066509 pair gap.
Observed ranges do not overlap, so repeat variability does not span the entire
original gap. It still affects interpretation, and no unique cause is proved.
Within-policy initial gradients differ by at most5.904e-8 symmetric relative L2;
median model distances reach36–37% by step800. First recorded scalar-loss
differences appear at steps14–18. This does not establish catastrophic gradients.

All12 new trained checkpoints,13 native scores, four FP32 backward replays and
all3,200 new sampled batches verify. The new runs preserve26.816% lower allocated
memory versus native FP32, costing12.65–17.65% more warmed update time. New
budget is3,200 updates /13,107,200 targets /3,208 backwards, with no failed stage
or scientific retry. All137 frozen sources and61 maintained hashes are retained.

H117's failed WikiText/two-corpus gate stays failed; its scoped TinyStories
component remains separate. No new default, layer or parameter reduction is
claimed. Do not allocate more identical long repeats automatically. A bounded
first-Adam-update comparison on the six saved initial gradients is the next
specific diagnostic question and needs a separate plan. The broad goal remains
open; the prior116-test result is historical, with no maintained-suite rerun.
[Plan](training_variability_plan.md),
[receipt](../results/verification/training_variability_final_v1.json).

"""
# Improve prose spacing while retaining every numerical value and path.
for old, new in (
    ("only0.630", "only 0.630"),
    ("is0.035284", "is 0.035284"),
    ("original0.066509", "original 0.066509"),
    ("most5.904", "most 5.904"),
    ("reach36", "reach 36"),
    ("step800", "step 800"),
    ("steps14", "steps 14"),
    ("All12", "All 12"),
    ("checkpoints,13", "checkpoints, 13"),
    ("all3,200", "all 3,200"),
    ("preserve26", "preserve 26"),
    ("costing12", "costing 12"),
    ("is3,200", "is 3,200"),
    ("/13,107", "/ 13,107"),
    ("/3,208", "/ 3,208"),
    ("All137", "All 137"),
    ("and61", "and 61"),
    ("prior116", "prior 116"),
):
    overview = overview.replace(old, new)
output = {}
title, body = before["README.md"].split("\n", 1)
body = body.lstrip("\n").replace(
    "The latest [fresh three-seed replication]", "The preceding [fresh three-seed replication]", 1
)
output["README.md"] = (
    title
    + "\n\n"
    + """The latest [repeat-variability diagnosis](research/training_variability_results.md)
adds four matched 800-update runs on the failed WikiText seed. Chunking remains
worse in both new pairs (+1.143% / +0.940% NLL), while unchanged native runs also
vary. The fixed diagnostic is **INCONCLUSIVE**; the prior two-corpus gate stays
failed. Initial gradient agreement does not establish training-trajectory
agreement. All checkpoint, native-score, replay and batch checks pass. No model
default changes and no new FFN breakthrough is established.

"""
    + body
)
for name in ("research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md"):
    title, body = before[name].split("\n", 1)
    body = body.lstrip("\n").replace(
        "## Latest: fresh replication", "## Previous: fresh replication", 1
    )
    output[name] = title + "\n\n" + overview + body
output["research/idea_bank.md"] = (
    before["research/idea_bank.md"].rstrip()
    + """
| H118 | Does the failed WikiText seed's FP32 chunk/native gap exceed unchanged repeat variation? | INCONCLUSIVE by fixed rules: four additional 800-update executions give +1.143% / +0.940% paired NLL gaps; all observed chunked endpoints are worse, but minimum cross-gap is only0.630%. All audits pass, memory saving persists. One selected seed; H117 failure unchanged. No further unchanged long repeats allocated; inspect first-update Adam sensitivity using saved gradients under a new plan. |
""".replace("only0.630", "only 0.630")
)
output["research/ARTIFACTS.md"] = (
    before["research/ARTIFACTS.md"].rstrip()
    + """

## H118 conditional training-variability diagnosis

[Results](training_variability_results.md), [plan](training_variability_plan.md),
[source guide](../results/training_variability_v1/source/README.md).
The protocol preserves137 source/plan hashes,61 maintained hashes and verifies
all78 H117 tensor artifacts before allocation. Four fresh repeated executions
add3,200 updates; the original pair's1,600 updates are retained separately.
Compact exports contain six metric rows,24 convergence rows and4,800 update
rows with explicit H117/H118 origins. Summary, protocols, compressed raw
result/audit/logs, environment and exit records remain available.

Local `runs/` holds16 new tensors:12 trained model/Adam/sampler states and four
initial-gradient files. The audit additionally uses nine original tensors
(one initial state, six trained states and two gradient sets), for25 compared
artifacts. It verifies13 native scores, four gradient replays, all new batches,
15 initial-gradient pairs and45 checkpoint-pair diagnostics. Prior tensors,
failures and navigation snapshots remain local and unchanged. A compact clone
cannot reproduce tensor/data verification without those local artifacts.
The maintained test suite is unchanged from its earlier116-test pass.
"""
)
output["research/literature.md"] = (
    before["research/literature.md"].rstrip()
    + """

## H118: compare repeat variability before attributing a training gap

[H118](training_variability_results.md) measures four additional executions of
the selected failed WikiText fixture. Initial gradients differ by tiny amounts,
but trained states and final NLLs differ between unchanged-policy runs. All
observed chunked endpoints remain worse; the predeclared1% separation criterion
is not established. This selected-seed result neither proves equivalence nor
isolates a kernel or optimizer cause.

The report gives the elementary update-map perturbation recurrence: controlling
initial gradient error alone cannot bound final-state error without assumptions
about every update's sensitivity. No new theorem is claimed. Primary context:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
These sources establish general numerical limitations, not a causal explanation
of our specific result. Adam's response to saved near-zero gradient coordinates
is a separate, unallocated diagnostic hypothesis.
"""
)
output[".gitignore"] = (
    before[".gitignore"].rstrip()
    + """

# H118: retain compact evidence; keep raw repeats and snapshots local.
/results/training_variability_v1/runs/
/results/training_variability_v1/before_documents/
/results/training_variability_v1/result.json
/results/training_variability_v1/audit.json
/results/training_variability_v1/progress.json
/results/training_variability_v1/*.log
!/results/training_variability_v1/*.json.gz
!/results/training_variability_v1/*.csv.gz
!/results/training_variability_v1/*.log.gz
!/results/training_variability_v1/*_exit.txt
!/results/training_variability_v1/*_protocol.json
!/results/training_variability_v1/environment.json
"""
)
assert output.keys() == before.keys()
for name, content in output.items():
    Path(name).write_text(content, encoding="utf-8", newline="\n")
print("Published H118 navigation and compact-artifact rules after checking seven snapshots.")
