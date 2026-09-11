"""Publish the failed fixed gate and seal the completed evidence."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/gradient_staging_v1")
p, r, s = [read(ROOT / n) for n in ("protocol.json", "result.json", "summary.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
prior = read("results/backward_allocation_v1/receipt.json")
for path, digest in prior["files"].items():
    target = ROOT / "CURRENT_STATE.before.md" if path == "research/CURRENT_STATE.md" else path
    assert sha(target) == digest
assert not s["passed"] and not r["broad_goal_achieved"]
assert all(not c["gates"]["memory"] for c in s["comparisons"])
assert all(all(v for k, v in c["gates"].items() if k != "memory") for c in s["comparisons"])
for stage in ("prepare", "study", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
rows = []
for c in s["comparisons"]:
    rows.append(
        f"| {c['dataset']} | {c['resident_mib']:.3f} | {c['staged_mib']:.3f} | {100 * (1 - c['memory_ratio']):.4f}% | {c['event_ratio']:.4f}x | {c['wall_ratio']:.4f}x |"
    )
ge = max(c["global_error"] for c in r["cases"])
te = max(c["tensor_error"] for c in r["cases"])
le = max(c["loss_error"] for c in r["cases"])
report = f"""# H123: completed-gradient host staging

**The fixed 10% memory gate fails on both corpora.** Savings of 9.4605% and
9.9778% must not be rounded into passes. Gradient and transfer-cost gates pass.

| Corpus | Resident peak MiB | Staged peak MiB | Saved | Event-time ratio | Wall-time ratio |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Both arms already offload checkpoint inputs and use chunked FP32 classifier
loss. The staged arm additionally moves each completed parameter gradient to
pinned CPU storage and clears its CUDA .grad field. It restores **every**
gradient to CUDA after backward. Restoration is included in both memory and
timing, so this is not a one-way transfer comparison. No optimizer step runs.

Peak CUDA allocation falls by 26.60 MiB on WikiText and 27.53 MiB on TinyStories.
Additional pinned gradient payload is 34.71 MiB. Combined pinned-host allocator
peak reaches 112.034 MiB, including the checkpoint-input buffers and rounded
cached allocations. Active and allocated host statistics are recorded separately.
These are process allocator figures, not system-wide VRAM or total RAM use.

The post-accumulation hook runs exactly once for each of 50 parameters per
backward, including tied embedding weights. Staged CUDA .grad payload is zero
at the synchronized backward boundary; all gradients are subsequently restored.
Every final restored gradient is bitwise equal to its staged CPU buffer.
Against H121's independently audited reference, maximum global symmetric
relative L2 is {ge:.3e}, maximum tensor relative L2 {te:.3e}, and maximum
relative loss difference {le:.3e}. The shared-weight FP64 toy passes 1e-12
checks. All saved gradients are finite; source weights, Adam moments/counters
and sampler state remain unchanged.

Budget: two toy backwards plus four cases with ten backwards each = **42
backwards, zero updates, 163,840 diagnostic target evaluations**. The last seven
repetitions are timed after three warmups. Mean, median, sample variance and
all repetitions are saved. This is a short transfer-cost screen, not sustained
training throughput or language-quality evidence. No case was repeated.
All six GPU boundaries are zero; all 61 maintained file hashes are unchanged.

The prototype only handles one backward per batch. It intentionally rejects
multiple post-accumulation callbacks; gradient accumulation, distributed
training and concurrent streams are not qualified. The original AdamW/global
clipping recipe would still need complete-training validation. That follow-up
was not earned under this study's fixed gate. H121/H122 failures remain intact.

**Next hypothesis:** reduce a remaining temporary allocation instead of making
this transfer path more complex. For example, a memory-efficient normalization
backward could be tested against the maintained RMSNorm under explicit gradient
and end-to-end memory gates. This is a new hypothesis, not a rescue of H123.

[Plan](gradient_staging_plan.md), [source](../results/gradient_staging_v1/staging.py),
[evidence receipt](../results/gradient_staging_v1/receipt.json).
The existing [PyTorch post-accumulation hook](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_post_accumulate_grad_hook.html)
supports changing a leaf parameter's .grad field. Related gradient-lifetime
work appears in the official [optimizer-in-backward tutorial](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html).
No algorithmic novelty, parameter reduction or breakthrough is claimed.
"""
Path("research/gradient_staging_results.md").write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
note = """## Latest: gradient staging narrowly misses the memory gate

[H123](gradient_staging_results.md) completes 42 backwards, zero updates.
Completed gradients move to pinned CPU buffers and are restored to CUDA;
round-trip cost is included. Peak allocation falls 26.60–27.53 MiB, but
9.4605%/9.9778% savings both fail the fixed 10% requirement. Gradient checks
pass; event overhead is 11.3–13.0%, within the 15% limit. Combined pinned-host
allocation is 112.034 MiB. No training extension, default change or repeated
case. Next hypothesis: reduce remaining temporary storage, such as RMSNorm
backward intermediates, under a separate correctness/resource protocol.
The broader research goal remains open; H121/H122 gates remain failed.

"""
current.write_text(
    head + "\n\n" + note + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    ROOT / ".gitignore",
    ROOT / "CURRENT_STATE.before.md",
    *ROOT.glob("runs/*/*"),
    current,
    Path("research/gradient_staging_plan.md"),
    Path("research/gradient_staging_results.md"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_gate="FAIL_MEMORY_BOTH_CORPORA",
    gradient_checks_passed=True,
    backwards=42,
    optimizer_updates=0,
    diagnostic_targets=163840,
    gradient_artifacts=4,
    maintained_hashes_verified=61,
    repeated_cases=0,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
