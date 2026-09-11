"""Update the research entry points after preserving the prior navigation snapshot."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")
protocol = json.loads((ROOT / "protocol.json").read_text())
summary = json.loads((ROOT / "summary.json").read_text())
assert summary["qualified_scopes"] == []
for name, record in protocol["before_documents"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == record["sha256"], name


def prepend(name, content):
    path = Path(name)
    title, remaining = path.read_text(encoding="utf-8").split("\n", 1)
    if name == "research/CURRENT_STATE.md":
        remaining = remaining.replace("## Latest:", "## Previous:", 1)
    path.write_text(title + "\n\n" + content.strip() + "\n" + remaining, encoding="utf-8")


prepend(
    "README.md",
    """
The latest [complete-model precision screen](research/fp32_classifier_profile_results.md)
finishes 56 matched checkpoint continuations / 2,800 updates. FP32 classifier
chunks save **32-33% whole-job allocation** on narrow context-512 models, with
about **13% slower updates**, but fail the original global-gradient fidelity
gate. Full context-128 GELU/SwiGLU save only 0.53-1.08%. Two gradient replay
audits also fail. Native scores and all batches verify; the fixed qualification
claim is closed and no fresh language allocation is earned. These are scoped
memory observations, with no parameter reduction or breakthrough claim.
""",
)
prepend(
    "research/CURRENT_STATE.md",
    """
## Latest: full-model precision gate fails despite useful narrow-model memory savings

H114 completes **56 checkpoint continuations / 2,800 updates** in 502.12 seconds,
using all twelve H112 final states plus full GELU/SwiGLU H078 controls. Each
case receives 20 warmup and 30 measured updates from the same paired weights
and Adam state. This is not fresh LM training. FP32 classifier chunks save
**32.36% / 33.06%** whole-job allocation on WikiText/TinyStories narrow T512
models, costing **12.84% / 12.64%** more median update time. Native validation
NLL differs by at most 0.0051% after 50 updates; this does not prove long-run quality.

Every scope fails the original <=0.002 global-gradient relative-L2 gate versus
the native FP32 classifier. Median candidate error is 0.00225 / 0.00337 in the
two narrow corpora. Full T128 GELU/SwiGLU additionally fail memory: only 0.53%
/ 1.08% whole-job saving. Final diagnostics set both full-GELU peaks and the
FP32-chunk SwiGLU peak. All 5,992 memory intervals are logged; warmed timing
block ratios remain <=1.0479, without outlier removal.

Independent exact gradient replay fails, and an explicit 1e-5 global / 1e-4
per-tensor numerical recovery also fails on a different fixture (0.0005106
global error). The cause is unresolved; no further tolerance relaxation or
training retry occurs. A narrower completion audit verifies **70 native scores,
56 saved gradient hashes, final model/Adam states and all 2,800 batches** with
zero backwards. It does not qualify independent gradient replay.

Close the fixed full-model precision-fidelity claim. Retain the scoped memory
observation; hold fresh LM allocation pending a new matched-state backward
diagnosis. Two model folders, five variants, eight recipes and all 61 maintained
files remain unchanged from the prior 116-test pass. The broad goal remains open.
[Report](fp32_classifier_profile_results.md), [plan](fp32_classifier_profile_plan.md),
[verification and explicit limitations](../results/verification/fp32_classifier_profile_final_v1.json).
""",
)
prepend(
    "research/PROGRESS_OVERVIEW.md",
    """
## Newest result: 32-33% lower memory, but full-gradient qualification fails

[H114](fp32_classifier_profile_results.md) completes 56 matched 50-update
continuations in 502 seconds. A BF16 decoder with FP32 classifier chunks keeps
about one-third lower whole-job allocation on narrow T512 models and costs
roughly 13% more update time. All short native validation changes are tiny.
Full T128 GELU/SwiGLU achieve only 0.53-1.08% job saving, below the 15% gate.

The stronger test rejects the fixed precision recipe: every scope fails the
original full-model gradient threshold. Exact and numerical gradient replays
also fail; both are preserved and the cause remains unresolved. The successful
completion audit covers native scores, saved states/gradients and batches, not
complete backward reproducibility. No fresh training or new architecture is
earned. The next useful question is why the matched BF16 decoder gradients
change, while preserving the demonstrated memory component and every failed gate.
""",
)
with Path("research/idea_bank.md").open("a", encoding="utf-8") as stream:
    stream.write("""

## H114 — FP32 classifier chunks in complete models: fixed fidelity claim rejected

**Status: scoped memory component; full-model qualification fails.** 14 saved
models × four policies × 50 updates = 2,800 updates, 10,649,600 targets, 502.12s.
Narrow T512 job allocation improves 32.36% / 33.06%, with 12.84% / 12.64% slower
updates and tiny short-NLL changes. Full T128 models save just 0.53-1.08%.
Every scope fails the original global-gradient gate versus native FP32.
Two independent gradient replay audits also fail; a narrower score/state/stream
audit passes. Do not relax those failed gates or claim a fresh LM result.
Retain memory accounting, reject the fixed fidelity/full-width saving claims,
and investigate backward reproducibility under a separately frozen diagnosis.
[Report](fp32_classifier_profile_results.md), [plan](fp32_classifier_profile_plan.md).
""")
with Path("research/ARTIFACTS.md").open("a", encoding="utf-8") as stream:
    stream.write("""

## H114: complete-model classifier precision and explicit audit failures

`results/fp32_classifier_profile_v1` freezes 104 sources and 14 starting states.
Its 56 cases perform 2,800 updates and retain 112 local gradient/final-state
tensor artifacts. `result.json.gz` preserves every update and all 5,992 memory
phases. `metrics.csv.gz` and `updates.csv.gz` expose all 56 / 2,800 rows.
`completion_audit.json.gz` verifies 70 native scores, saved-gradient integrity,
final Adam states and all batches; it explicitly does not certify gradient replay.

`audit_failure.json`, `audit_recovery_failure.json`, both frozen audit protocols,
the six-pass diagnostic and compressed original traces preserve the two failed
gradient audits. `postprocessing_record.json` records the interrupted analyzer
and premature plotting attempt; unchanged postprocessing then succeeds without
scientific reruns. `before_documents/` and raw `runs/` remain local, not deleted.
The [H114 final receipt](../results/verification/fp32_classifier_profile_final_v1.json)
separates evidence-integrity PASS from failed candidate qualification and an
unresolved gradient replay. The [report](fp32_classifier_profile_results.md)
includes full training-versus-job peaks, all controls and the scoped decision.
""")
prepend(
    "research/literature.md",
    """
## H114 numerical accuracy and CUDA timing

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
warns that equivalent floating-point computations need not agree bitwise.
[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) and the
[event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)
support separate synchronized wall/event timing and allocator accounting.
[H114](fp32_classifier_profile_results.md) measures stable warmed full-model timing
but fails its full-gradient gate and two replay checks. The documentation does
not identify the cause of these particular discrepancies. All original gates
and failures remain explicit; no novelty or deterministic-backward claim follows.
""",
)
with Path(".gitignore").open("a", encoding="utf-8") as stream:
    stream.write("""

# H114: full compact evidence and failures; large checkpoints/histories stay local.
/results/fp32_classifier_profile_v1/runs/
/results/fp32_classifier_profile_v1/before_documents/
/results/fp32_classifier_profile_v1/result.json
!/results/fp32_classifier_profile_v1/*.json.gz
!/results/fp32_classifier_profile_v1/*.csv.gz
!/results/fp32_classifier_profile_v1/*.log.gz
!/results/fp32_classifier_profile_v1/*_exit.txt
!/results/fp32_classifier_profile_v1/*_protocol.json
!/results/fp32_classifier_profile_v1/*failure*.json
!/results/fp32_classifier_profile_v1/qualification.json
!/results/fp32_classifier_profile_v1/environment.json
!/results/fp32_classifier_profile_v1/gradient_replay_diagnosis.json
!/results/fp32_classifier_profile_v1/postprocessing_record.json
""")
record = dict(
    scientific_training_repeated=False,
    analyzer_stall=dict(
        pid=16516,
        observed_cpu_seconds=95.72,
        observed_working_set_bytes=24576000,
        action="terminated only the confirmed H114 analyzer process",
        child_exit_code_unsigned=4294967295,
        launcher_tool_exit_code_signed=-1,
        cause="unknown",
        no_summary_or_csv_existed_at_termination=True,
    ),
    premature_report_attempt="FileNotFoundError for missing summary; no report was written",
    premature_plot_attempt="FileNotFoundError for missing CSV; original plot.log and exit 1 preserved",
    failed_process_inspection="optional psutil unavailable; used read-only PowerShell process metadata",
    recovery="same analyzer with traceback watchdog finished; plot rerun after numeric exports existed",
    source_hashes={
        str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            ROOT / "source/analyze.py",
            ROOT / "source/analyze_recovery.py",
            ROOT / "source/plot.py",
            ROOT / "source/plot_recovery.py",
        )
    },
)
(ROOT / "postprocessing_record.json").write_text(json.dumps(record, indent=2) + "\n")
print("Updated six research entry points and compact artifact policy; previous versions preserved")
