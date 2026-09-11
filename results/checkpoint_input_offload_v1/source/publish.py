"""Publish bounded findings after checking preserved navigation snapshots."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import ROOT, read, sha

p = read(ROOT / "protocol.json")
for name, info in p["before_documents"].items():
    assert sha(name) == info["sha256"] == sha(info["preserved_path"])


def original(path):
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def save(path, value):
    Path(path).write_text(value, encoding="utf-8", newline="\n")


intro = """The latest [checkpoint-input offload diagnostic](research/checkpoint_input_offload_results.md)
saves **48 MiB (about 12%)** with native classifier loss and about **15 MiB
(5%)** with chunked loss. All gradient and timing checks pass, but the chunked
cases fail the preset 10% memory gate. These are 262 fixed-state backward
checks with zero optimizer updates, not a training-quality or parameter-saving
result. No default changed; the broader research goal remains open.

"""
section = """## Latest: checkpoint-input offload preserves gradients but misses the combined gate

[H121](checkpoint_input_offload_results.md) temporarily places eight block
checkpoint inputs in pinned CPU memory using an existing PyTorch API. With
native FP32 classifier loss, complete diagnostic-case peak falls by exactly
48 MiB (11.81–11.99%) on WikiText and TinyStories, with 1.21–2.34% event-time
overhead. With chunked loss it falls by only 14.47–15.40 MiB (4.98–5.19%).
**The all-four-fixture 10% memory gate fails.** No training extension is
earned under this protocol; no maintained default changes.

All traces confirm eight pinned [8,512,384] FP32 inputs, exact copy hashes,
one unpack each and zero logical payload after cleanup. Payload is 48 MiB;
rounded pinned-host allocation is about 64 MiB and its cache persists. FP64
finite differences and 24 NumPy gradient comparisons pass; maximum global
gradient relative L2 is 9.344e-8 and checked losses match exactly. GPU allocator
boundaries are zero and model/moment states remain unchanged.

The first audit process crashed during SymPy import before PyTorch/replay.
Its failure is preserved; a prospective recovery runs only the eight remaining
replays through the unchanged audit function. Total work: 262 backwards,
zero optimizer/training updates, 1,048,576 diagnostic target evaluations,
24 gradient artifacts and 824 full-model memory intervals. Repeated fixed-state
targets are not training exposure. The native import issue remains undiagnosed.

All 162 frozen sources and 61 maintained files stay intact. The previous
116-test pass is historical, not a new test execution. No parameter reduction,
endpoint quality improvement or novelty is claimed. H117's quality and H119's
active-clipping failures remain. A separately budgeted allocation timeline
could identify the remaining chunked backward peak before another change;
phase-boundary inventories alone do not uniquely explain it.
[Plan](checkpoint_input_offload_plan.md),
[recovery](checkpoint_input_offload_audit_recovery.md),
[receipt](../results/verification/checkpoint_input_offload_final_v1.json).

"""
head, rest = original("README.md").split("\n\n", 1)
save(
    "README.md",
    head + "\n\n" + intro + rest.replace("The latest [optimizer", "The preceding [optimizer", 1),
)
for path in ("research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md"):
    head, rest = original(path).split("\n\n", 1)
    save(path, head + "\n\n" + section + rest.replace("## Latest:", "## Previous:", 1))
save(
    "research/idea_bank.md",
    original("research/idea_bank.md")
    + """
| H121 | Can pinned host storage of checkpoint inputs lower diagnostic peak by at least 10% with at most 15% time overhead? | ELIMINATED FOR THE ALL-FOUR-FIXTURE GATE: native classifier saves 48 MiB (about 12%); chunked classifier about 15 MiB (5%), failing its memory gate. All numerical/trace/timing checks pass. 262 backwards, zero updates; roughly 64 MiB pinned host allocation. Existing PyTorch API; no novelty or quality claim. Inspect remaining backward allocations under a new protocol. |
""",
)
save(
    "research/ARTIFACTS.md",
    original("research/ARTIFACTS.md")
    + """
## H121: selective checkpoint-input host storage

[Report](checkpoint_input_offload_results.md), [plan](checkpoint_input_offload_plan.md)
and [audit recovery](checkpoint_input_offload_audit_recovery.md) preserve the
failed combined gate and bounded native-classifier saving. Keep source,
protocols, summary, figure and compressed JSON/CSV/log evidence in Git. The
24 raw gradients, per-case JSON and navigation snapshots remain local, with
hashes in the receipt. All 262 backwards and 824 full-model intervals are
accounted for; the failed SymPy import completed no scientific case.
""",
)
save(
    "research/literature.md",
    original("research/literature.md")
    + """
## H121: existing saved-tensor hooks for checkpoint-input offload

[PyTorch save_on_cpu](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.graph.save_on_cpu),
[checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html) and the official
[checkpoint/offload wrapper](https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py)
were checked on 2026-09-10. Installed checkpoint, autograd, CUDA-memory and
wrapper source hashes are frozen. This runtime saves checkpoint inputs before
entering its internal hook, so an outer CPU hook captures the eight boundary
inputs. H121 verifies their values and lifetimes experimentally. The technique
is established; the measured saving is about 12% with native classifier loss
and 5% with chunked loss, failing the preset all-four-fixture memory gate.
No novel algorithm is claimed.
""",
)
save(
    ".gitignore",
    original(".gitignore")
    + """
# H121: compact diagnostics; raw gradients and snapshots remain local.
/results/checkpoint_input_offload_v1/runs/
/results/checkpoint_input_offload_v1/before_documents/
/results/checkpoint_input_offload_v1/result.json
/results/checkpoint_input_offload_v1/audit.json
/results/checkpoint_input_offload_v1/qualification_progress.json
/results/checkpoint_input_offload_v1/*.log
!/results/checkpoint_input_offload_v1/*.json.gz
!/results/checkpoint_input_offload_v1/*.csv.gz
!/results/checkpoint_input_offload_v1/*.log.gz
!/results/checkpoint_input_offload_v1/*_exit.txt
!/results/checkpoint_input_offload_v1/*_protocol.json
!/results/checkpoint_input_offload_v1/environment.json
""",
)
print("Seven navigation documents updated; snapshots preserved.")
