"""Update navigation after verifying exact pre-study document snapshots."""

from pathlib import Path

from results.optimizer_memory_v1.source.prepare import ROOT, read, sha

p = read(ROOT / "protocol.json")
for name, info in p["before_documents"].items():
    assert sha(name) == info["sha256"] == sha(info["preserved_path"])


def original(path):
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def save(path, value):
    Path(path).write_text(value, encoding="utf-8", newline="\n")


intro = """The latest [optimizer memory profile](research/optimizer_memory_results.md)
finds that fused AdamW saves **35–36 MiB during the optimizer step**, but saves
**no complete-job VRAM**: backward sets every peak. Per-tensor AdamW also fails
the memory gate and is slower. All 12 matched continuations, 360 updates and
independent numerical/data/scoring audits complete. Both alternatives are
eliminated for this memory objective; model defaults remain unchanged.

"""
section = """## Latest: optimizer temporary savings do not reduce the complete-job peak

[H120](optimizer_memory_results.md) compares default, per-tensor and fused
native AdamW on the same update800 states for WikiText/TinyStories, using both
native and chunked FP32 classifier losses. Twelve cases × 30 updates complete.
Backward sets all 12 job peaks. Fused saves 35–36 MiB within the optimizer phase,
but complete-job allocation rises 25 KiB; per-tensor mode leaves it identical.
**Both are ELIMINATED for the fixed 10% whole-job memory gate.**

Fused event-sum timing is 2.7–3.6% lower in the instrumented screen; this is not
an uninstrumented throughput claim. Per-tensor mode fails several timing gates.
All short NLL, stability, numerical, batch and scoring gates pass. These are
correlated continuations of one seed per corpus, not a broad quality result.
First-step clipping is inactive (norm<1), so its zero error does not resolve
H119's active CPU clipping failure.

The audit independently verifies first-step AdamW with NumPy FP64, every
first/final model/moment state, all 360 batches/samplers and 12 native final
scores. Total work is 360 updates/backwards / 1,474,560 targets, plus 24 study
and 12 audit scores. All 1,884 memory intervals and 26 zero CUDA boundaries
are retained. No scientific stage or case was repeated. A later read-only
PowerShell display failure is recorded separately.

All 152 frozen sources and 61 maintained files stay intact. No optimizer,
layer, parameter count or default is promoted; the historical 116-test result
is not represented as a new test run. H117's old quality gate remains failed.
Move the memory hypothesis to backward storage: inspect which checkpoint
inputs remain resident and test selective host storage only under a separate
budget with gradient/lifetime/transfer-cost checks. The eight input tensors
contain at most 48 MiB of payload; that is an estimate, not a measured saving.
The broad VRAM/quality and architectural goal remains open.
[Plan](optimizer_memory_plan.md),
[receipt](../results/verification/optimizer_memory_final_v1.json).

"""
head, rest = original("README.md").split("\n\n", 1)
save(
    "README.md", head + "\n\n" + intro + rest.replace("The latest [Adam", "The preceding [Adam", 1)
)
for path in ("research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md"):
    head, rest = original(path).split("\n\n", 1)
    save(path, head + "\n\n" + section + rest.replace("## Latest:", "## Previous:", 1))
save(
    "research/idea_bank.md",
    original("research/idea_bank.md")
    + """
| H120 | Can existing per-tensor/fused AdamW reduce complete-job allocation by lowering optimizer temporaries? | ELIMINATED FOR THIS MEMORY GATE: optimizer-only saving does not lower any of four fixture job peaks; backward dominates. Fused is 25 KiB higher overall, per-tensor identical/slower. All numerical/data/scoring checks pass; 12 continuations / 360 updates. No broad quality or new architecture claim. Move to checkpoint-input lifetime/residency with a separately frozen test. |
""",
)
save(
    "research/ARTIFACTS.md",
    original("research/ARTIFACTS.md")
    + """
## H120: complete-update optimizer memory profile

[Report](optimizer_memory_results.md) and [plan](optimizer_memory_plan.md) retain
all 12 matched continuations and failed complete-job memory gates. Source,
protocols, summary, figure and compressed result/audit/metrics/phase/update/log
files belong in Git. The 36 raw gradient/model/moment artifacts and navigation
snapshots stay local. All 360 updates and 1,884 intervals are accounted for;
the read-only CLR display failure is a separate note, not a scientific retry.
""",
)
save(
    "research/literature.md",
    original("research/literature.md")
    + """
## H120: optimizer temporaries are distinct from complete-job peaks

[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
documents foreach temporary storage and its native per-tensor/fused alternatives.
The documentation was checked on 2026-09-10 and installed optimizer sources
were pinned before profiling. [H120](optimizer_memory_results.md) measures a
35–36 MiB fused optimizer-phase reduction but no complete-job reduction:
backward is larger in every fixture. This is a scoped systems finding, not a
new optimization algorithm. Independent NumPy updates and native scoring pass.
Further memory work must measure backward tensor lifetimes and residency;
a temporary-storage reduction in a smaller phase cannot lower an unchanged
larger phase. Existing long-run quality failures remain failed.
""",
)
save(
    ".gitignore",
    original(".gitignore")
    + """
# H120: compact complete-update profiles; raw tensors and snapshots stay local.
/results/optimizer_memory_v1/runs/
/results/optimizer_memory_v1/before_documents/
/results/optimizer_memory_v1/result.json
/results/optimizer_memory_v1/audit.json
/results/optimizer_memory_v1/*.log
!/results/optimizer_memory_v1/*.json.gz
!/results/optimizer_memory_v1/*.csv.gz
!/results/optimizer_memory_v1/*.log.gz
!/results/optimizer_memory_v1/*_exit.txt
!/results/optimizer_memory_v1/*_protocol.json
!/results/optimizer_memory_v1/environment.json
!/results/optimizer_memory_v1/observation_failure.txt
""",
)
print("Seven navigation documents updated; snapshots preserved.")
