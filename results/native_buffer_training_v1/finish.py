"""Seal complete-update evidence and update the research decision."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/native_buffer_training_v1")
p, r, a, s = [
    read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json", "summary.json")
]
assert not (ROOT / "receipt.json").exists()
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/attention_reproducibility_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for dataset in p["datasets"].values():
    folder = Path(dataset["path"])
    assert sha(folder / "manifest.json") == dataset["manifest_sha256"]
    hashes(
        {
            (folder / name).as_posix(): digest
            for name, digest in dataset["manifest"]["files"].items()
        }
    )
for stage in ("prepare", "ordinary", "deterministic", "prepare_audit", "audit", "analyze", "plot"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
status = (
    "PASS short complete-training gates" if s["passed"] else "FAIL combined short-training gate"
)
next_step = (
    "The candidate earns a separately frozen longer fresh-initialization study, then multiple seeds and scale. Include the ordinary execution control and matched deterministic controls; do not infer universal quality or resource savings from this continuation."
    if s["passed"]
    else "Do not expand the training matrix yet. The deterministic native arm adds exactly 47.75 MiB over ordinary native on both corpora. Isolate the deterministic policy bundle, especially cuBLAS workspace allocation, before more training: test a lower-workspace deterministic configuration with matched controls, exactness checks, and full resource accounting. This is a causal hypothesis, not an attribution established by the present experiment. Preserve the current failures and thresholds."
)
rows = [
    f"| {x['dataset']} | {x['arm']} | {x['nll']:.8f} | {x['peak_mib']:.3f} | {x['event_ms']:.3f} | {x['wall_ms']:.3f} | {x['stability']:.3f} |"
    for x in s["metrics"]
]
comparisons = []
for x in s["comparisons"]:
    q = x["ratios"]
    failed = ", ".join(k for k, v in x["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {x['dataset']} | {x['control']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f}x | {q['wall_ms']:.4f}x | {100 * (q['nll'] - 1):+.5f}% | {failed} |"
    )
exact_rows = [
    f"| {x['dataset']} | {x['arm']} | {'pass' if x['passed'] else 'fail'} |" for x in a["exact"]
]
host = max(x["host_mib"] for x in s["metrics"])
score_error = max(x["check"]["relative_error"] for x in a["scores"])
report = f"""# H130: native-buffer classifier through complete optimizer updates

**{status}.** Eight 30-update continuations compare ordinary native execution
with three deterministic arms on two corpora. These are fixed seed-101
continuations from trained step-800 states, not fresh training or multiseed proof.
Independent numerical/data/score and deterministic-equality audit: **{a["passed"]}**.

| Corpus | Arm | Final validation NLL | Job peak MiB | Median event ms | Median wall ms | Timing stability |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Candidate (`reuse`) comparisons; positive NLL change is worse:

| Corpus | Control | GPU allocation saved | Event ratio | Wall ratio | NLL change | Failed gates |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

![Complete-job memory and update timing](figures/native_buffer_training.png)

`ordinary` uses native classifier loss and resident checkpoint inputs with the
original nondeterministic policy. `native` changes only to H129's reproducible
deterministic-default policy. `offload` additionally offloads checkpoint inputs
to CPU. `reuse` also uses H128's layout-aware native-buffer classifier.
All use ordinary RMSNorm, resident gradients and unchanged global clipping and
default AdamW. No compact RMSNorm or gradient staging. Parameter count is
9,099,648 in every arm. Pinned host allocation peaks at {host:.3f} MiB and is
additional RAM; the cache can persist across arms within each process.

The candidate saves 12.85%/13.05% complete-job GPU allocation versus ordinary\nexecution on WikiText/TinyStories, while final validation NLL stays within\n0.001% of the ordinary control. CUDA-event update overhead is 20.46%/17.06%;\nwall overhead is 21.22%/16.24%. These exceed the fixed 15% runtime limit, so\nthe combined gate fails despite passing memory, quality, state and stability\nchecks. All comparisons against the two deterministic controls pass.\n\n## Equality and independent checks

The predeclared deterministic comparison requires equal first raw/clipped
gradients, all 30 loss/gradient-norm pairs, initial/final validation NLL, and
model/optimizer/sampler hashes at saved steps 801 and 830:

| Corpus | Compared with deterministic native | All exactness checks |
|---|---|---|
{chr(10).join(exact_rows)}

The independent inherited audit recomputes clipping and the first AdamW update
and moments in NumPy, verifies all saved model/optimizer hashes and finiteness,
reconstructs all 240 batches/sampler endpoints, and scores every final model
through the native evaluator. Maximum native-score relative discrepancy is
{score_error:.3e}. The strict equality checks concern the saved checkpoints and
recorded losses/norms; unrecorded intermediate model states were not directly
compared. Ordinary-versus-deterministic trajectories are not required to match
bitwise and use the predeclared +1% final-NLL limit.

## Resource accounting and scope

**240 updates/backwards, 983,040 training targets**, 16 full study scores and
eight independent native scores. Twenty-four tensor artifacts hold first
raw/clipped gradients and model/optimizer checkpoints at 801/830. All 1,256
memory intervals and 19 zero GPU allocator boundaries verify. Source/dataset
hashes and all 61 maintained file hashes verify; no completed case was retried.

The existing H120 update loop is reused unchanged. Its peak includes construction,
initial/final evaluation, warmup, all training/transfers, clipping, optimizer,
diagnostics and serialization. Independent native replay is separate audit
work. CUDA allocation excludes driver/context memory and other processes.
Timing uses updates 11–30, retaining mean, median, variance and every raw step;
there is no profiler in timed regions. Original controls run in a separate
process before deterministic arms, so order/thermal effects remain a limitation.
Within deterministic execution arm order reverses across corpora. Each arm's
half-window timing stability must be within 1.15; that check does not eliminate
all runtime uncertainty.

The classifier implementation retains the native full-batch matrix products
and loss kernels while reusing private buffers, with H128's layout correction.
This storage experiment introduces no new activation or FFN. See
[H128](native_buffer_layout_results.md) for equations, qualification and prior
art, and [H129](attention_reproducibility_results.md) for the execution policy.
Earlier failed gates remain unchanged. No maintained defaults change.

## Next decision

{next_step}
The broader VRAM/quality and parameter-efficient FFN goal remains open.

[Plan](native_buffer_training_plan.md),
[summary](../results/native_buffer_training_v1/summary.json),
[audit](../results/native_buffer_training_v1/audit.json),
[evidence receipt](../results/native_buffer_training_v1/receipt.json).
"""
report_path = Path("research/native_buffer_training_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
current.write_text(
    head
    + f"\n\n## Latest: complete native-buffer training comparison\n\n[H130](native_buffer_training_results.md): **{status}.**\nEight 30-update continuations, 240 updates, 24 tensor artifacts. Independent\nnumerical/data/score and deterministic-equality audit: {a['passed']}. All 61\nmaintained files unchanged; no parameter reduction or default change.\n\n{next_step}\nThe full research goal remains open.\n\n"
    + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
readme.write_text(
    head
    + f"\n\nLatest: [complete native-buffer training](research/native_buffer_training_results.md).\n**{status}.** Eight matched short continuations; see the report for exact\nresource/quality comparisons and limitations. The broader goal remains open.\n\n"
    + rest.replace("Latest:", "Earlier reproducibility study:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
for path in ROOT.glob("*/runs/*/result.json"):
    path.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    *ROOT.glob("*.before.md"),
    ROOT / ".gitignore",
    *ROOT.glob("*/runs/*/*"),
    current,
    readme,
    report_path,
    Path("research/native_buffer_training_plan.md"),
    Path("research/figures/native_buffer_training.png"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_status=status,
    short_training_passed=s["passed"],
    independent_audit_passed=a["passed"],
    training_updates=240,
    backwards=240,
    training_targets=983040,
    tensor_artifacts=24,
    maintained_hashes_verified=61,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
