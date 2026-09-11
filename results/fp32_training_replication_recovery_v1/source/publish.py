"""Publish H117 navigation after verifying all original document snapshots."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")


def run():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    audit = json.loads((ROOT / "audit.json").read_text())
    assert summary["status"] == "COMPLETE" and len(summary["decisions"]) == 2
    assert (ROOT / "plot_exit.txt").read_text().strip() == "3221225477"
    assert (ROOT / "plot_recovery_exit.txt").read_text().strip() == "0"
    assert Path("research/fp32_training_replication_results.md").exists()
    before = {}
    for name, info in protocol["before_documents"].items():
        data = Path(name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], name
        assert data == Path(info["preserved_path"]).read_bytes(), name
        before[name] = data.decode("utf-8").replace("\r\n", "\n")
    rows = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode())
        )
    )
    candidates = [r for r in rows if r["policy"] == "fp32_default_chunks"]
    lo, hi = [
        fn(100 * (1 - float(r["memory_ratio_bf16"])) for r in candidates) for fn in (min, max)
    ]
    verdicts = "; ".join(
        d["scope"]
        + (
            " qualifies in this scope"
            if d["qualifies_scoped_800_update_component"]
            else " fails the fixed gates"
        )
        for d in summary["decisions"]
    )
    failed = (
        ", ".join(sorted({k for d in summary["decisions"] for k, v in d["gates"].items() if not v}))
        or "none"
    )
    brief = f"""The latest [fresh three-seed replication](research/fp32_training_replication_results.md)
completes **18 runs / 14,400 updates**. FP32 decoder/classifier chunking saves
**{lo:.2f}–{hi:.2f}% full-job tensor allocation** versus BF16 native. {verdicts}.
The fixed two-corpus claim {"passes only in this measured scope" if summary["two_corpus_component_qualified"] else "fails"}.
Every seed uses 800 updates from matched random initialization; a passing mean
cannot override a failed seed. The independent audit {"passes" if audit["passed"] else "does not pass all numerical gates"}:
six regenerated initial states, 54 trained checkpoints, 60 native scores,
12 FP32 gradient replays and all 14,400 sampled batches. No new architecture,
parameter reduction or default change is established. The broad goal stays open.

"""
    old_readme = before["README.md"]
    title, body = old_readme.split("\n", 1)
    body = body.lstrip("\r\n").replace(
        "The latest [full-job FP32 resource screen]",
        "The preceding [full-job FP32 resource screen]",
        1,
    )
    output = {"README.md": title + "\n\n" + brief + body}
    current = f"""## Latest: fresh replication retains memory savings but requires every seed's quality

[H117](fp32_training_replication_results.md) completes 18 fresh 800-update runs
on seeds 101/113/127 for both WikiText-2 and TinyStories. Full-job tensor allocation
falls {lo:.2f}–{hi:.2f}% versus BF16 native, with practical warmed update time.
{verdicts}. Failed gate names: {failed}.
The two-corpus recipe {"qualifies in this measured scope" if summary["two_corpus_component_qualified"] else "does not qualify"}.

The candidate and both references have identical 9,099,648 parameters. Initial
FP32 gradient fidelity does not guarantee the endpoint after 800 updates.
Independent verification covers six exact initial-state regenerations, 54
trained model/Adam/sampler checkpoints, 60 complete native validation scores,
12 FP32 initial-gradient replays and all 14,400 sampled batches. Three seeds,
reused development validation and no complete-trajectory repeats limit the claim.

The original startup failed after 24 CPU qualification backwards and before
any training update because environment metadata initialized CUDA too early.
Its frozen recovery reorders CPU initialization before metadata; no training,
scientific recipe or tolerance is repeated or changed. Total backwards across
attempts: 14,478; recorded memory intervals: 28,998. Original failure is retained.
The initial figure process later hit a Matplotlib import access violation;
one frozen-input CPU recovery ran the unchanged plot successfully. No training
or audit was repeated and no runtime cure is claimed.

No default or maintained model is promoted. All 61 maintained files remain
unchanged from the earlier 116-test pass; that suite is not rerun. Keep the
scoped memory evidence and close failed fixed quality claims. Paired versus
within-policy trajectory variability is the next bounded diagnostic question,
before any further broad training sweep. The broad goal remains unmet.
[Plan](fp32_training_replication_plan.md),
[recovery](fp32_training_replication_recovery_plan.md),
[receipt](../results/verification/fp32_training_replication_final_v1.json).

"""
    for name in ("research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md"):
        title, body = before[name].split("\n", 1)
        body = body.lstrip("\r\n")
        body = body.replace("## Latest: default FP32", "## Previous: default FP32", 1)
        body = body.replace("## Newest result:", "## Previous result:", 1)
        output[name] = title + "\n\n" + current + body
    output["research/idea_bank.md"] = (
        before["research/idea_bank.md"].rstrip()
        + f"""
| H117 | Do default FP32 chunks preserve quality across fresh seeds and 800 updates? | {verdicts}. Allocation falls {lo:.2f}–{hi:.2f}% versus BF16 native; fixed all-seed gates, 18 fresh runs / 14,400 updates. Audit verifies 54 trained states, 60 native scores, 12 FP32 replays and all batches. Two-corpus claim {"qualifies only in scope" if summary["two_corpus_component_qualified"] else "REJECTED"}. Original zero-training startup failure retained; no new layer, default change or further sweep allocated. |
"""
    )
    output["research/ARTIFACTS.md"] = (
        before["research/ARTIFACTS.md"].rstrip()
        + """

## H117 fresh three-seed replication

[Result](fp32_training_replication_results.md),
[original plan](fp32_training_replication_plan.md),
[recovery plan](fp32_training_replication_recovery_plan.md),
[source guide](../results/fp32_training_replication_recovery_v1/source/README.md).
The original failed attempt and recovery are both retained. Recovery freezes
129 source/plan files; the audit freezes 85 files including 78 tensor artifacts.
Compact evidence includes protocols, environment/qualification, initialization
manifest, summary, compressed result/audit/logs, 18 metric rows, 14,400 update
rows and 72 validation rows. The final receipt checks all fixed-gate arithmetic,
prior evidence and 61 unchanged maintained files.

Six initial checkpoints, 54 trained model/Adam/sampler checkpoints and 18 saved
initial-gradient files remain local under recovery `initial/` and `runs/`.
Raw histories, aggregate JSON/logs and original pre-study navigation snapshots
are ignored, not deleted. These tensors and dataset caches are required for
independent replay; a compact clone alone cannot verify their contents. The
original attempt has zero training updates, a preserved traceback/exit1 and
24 CPU qualification backwards. Total across attempts is 14,478 backwards.
No maintained default changes and no historical test count is inflated.
The failed Matplotlib import and successful unchanged-plot CPU recovery are
retained with `plot_recovery_protocol.json`, both logs and exit records.
"""
    )
    output["research/literature.md"] = (
        before["research/literature.md"].rstrip()
        + """

## H117: local gradient agreement does not establish training equivalence

[H117](fp32_training_replication_results.md) extends H116 to 18 fresh runs,
three seeds per corpus and 800 updates. It retains memory savings while the
fixed all-seed quality requirement is assessed separately. No new activation,
FFN or classifier-memory algorithm is claimed. Native checkpoint scoring and
initial FP32 gradient replays are verified independently; full training
trajectory reproducibility is not tested. A failed endpoint cannot be uniquely
attributed to chunking without within-policy trajectory controls.

Previously reviewed primary sources remain applicable:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) and
[Cut Your Losses](https://arxiv.org/abs/2411.09009).
They explain general numerical constraints and established classifier-memory
work; they do not prove the cause of this study's specific NLL differences.
This is an empirical replication with scoped decisions, not a new literature
priority or theoretical convergence claim.
"""
    )
    output[".gitignore"] = (
        before[".gitignore"].rstrip()
        + """

# H117 raw states and logs remain local; compact evidence/source are retained.
results/fp32_training_replication_v1/before_documents/
results/fp32_training_replication_v1/initial/
results/fp32_training_replication_v1/runs/
results/fp32_training_replication_v1/*.log
results/fp32_training_replication_recovery_v1/before_documents/
results/fp32_training_replication_recovery_v1/initial/
results/fp32_training_replication_recovery_v1/runs/
results/fp32_training_replication_recovery_v1/*.log
results/fp32_training_replication_recovery_v1/result.json
results/fp32_training_replication_recovery_v1/audit.json
results/fp32_training_replication_recovery_v1/progress.json
!/results/fp32_training_replication_v1/*.log.gz
!/results/fp32_training_replication_v1/*_exit.txt
!/results/fp32_training_replication_v1/failure.json
!/results/fp32_training_replication_v1/qualification.json
!/results/fp32_training_replication_v1/environment.json
!/results/fp32_training_replication_recovery_v1/*.json.gz
!/results/fp32_training_replication_recovery_v1/*.csv.gz
!/results/fp32_training_replication_recovery_v1/*.log.gz
!/results/fp32_training_replication_recovery_v1/*_exit.txt
!/results/fp32_training_replication_recovery_v1/*_protocol.json
!/results/fp32_training_replication_recovery_v1/qualification.json
!/results/fp32_training_replication_recovery_v1/environment.json
!/results/fp32_training_replication_recovery_v1/initializations.json
"""
    )
    assert output.keys() == before.keys()
    for name, content in output.items():
        Path(name).write_text(content, encoding="utf-8", newline="\n")
    print("Published six navigation documents and artifact ignore rules after verifying snapshots.")


if __name__ == "__main__":
    run()
