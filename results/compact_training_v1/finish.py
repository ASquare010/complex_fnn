"""Seal the short complete-update result; broader training remains unproven."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_training_v1")
p, r, a, s, q = [
    read(ROOT / n)
    for n in ("protocol.json", "result.json", "audit.json", "summary.json", "qualification.json")
]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
previous = read("results/compact_rmsnorm_v1/receipt.json")
for path, digest in previous["files"].items():
    assert (
        sha(ROOT / "CURRENT_STATE.before.md" if path == "research/CURRENT_STATE.md" else path)
        == digest
    )
assert s["passed"] and a["passed"] and q["passed"]
for stage in ("prepare", "study", "prepare_audit", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
rows = []
for c in s["comparisons"]:
    rows.append(
        f"| {c['dataset']} | {c['baseline_mib']:.3f} | {c['combined_mib']:.3f} | {100 * (1 - c['memory_ratio']):.2f}% | {c['time_ratios']['event_sum_ms']:.4f}x | {c['time_ratios']['wall_ms']:.4f}x | {c['baseline_nll']:.6f} / {c['combined_nll']:.6f} |"
    )
pe = max(v["check"]["parameter_error"]["distance"] for v in a["numerical"])
me = max(e["distance"] for v in a["numerical"] for e in v["check"]["moment_errors"])
ge = max(v["global_relative"] for v in a["gradients"])
clip = max(v["clip_error"] for v in q["rows"])
score = max(v["check"]["relative_error"] for v in a["scores"])
report = f"""# H125: complete-step compact training screen

**Both corpora pass the predeclared memory, timing, quality and numerical gates.**
Compact RMSNorm plus gradient staging preserves the memory saving through
AdamW, global clipping, evaluation and serialization. This is a short
continuation result, not a fresh-run or multiseed breakthrough.

| Corpus | Baseline peak MiB | Candidate peak MiB | Saved | Event ratio | Wall ratio | Final NLL baseline / candidate |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Both arms use chunked FP32 loss and checkpoint-input offload. The candidate
adds analytical RMSNorm backward and pinned gradient staging, then restores
all CUDA gradients before the unchanged global clipping and default AdamW
step. Evaluation uses the ordinary RMSNorm path, including BF16. Parameter
count remains 9,099,648; there is no parameter reduction or new activation.

Complete-job peaks include initialization, ten warmups, all training phases,
evaluation, transfer/restoration, clipping, updates, serialization and checks.
Combined pinned-host allocated peak is 112.100 MiB; active/cached allocation is
recorded separately. CUDA allocated/reserved memory excludes driver/context
and other processes. Timing uses the last 20 of 30 complete updates; means,
medians, sample variances, split-half stability and individual rows are saved.
This instrumented continuation does not prove sustained production throughput.

## Evidence

- Four runs continue the same two trained checkpoints from step 800 to 830:
  **120 updates, 491,520 training targets, 122 backwards including qualification**.
- Two tiny-model backwards activate global clipping. Independent FP64 clipping
  error is at most {clip:.3e}. FP32/BF16 evaluation outputs match exactly.
- Independent NumPy AdamW checks pass: maximum first-update parameter relative
  error {pe:.3e}, moment error {me:.3e}; first raw gradient discrepancy
  {ge:.3e}. First/final steps, states, moments and finiteness are checked.
- All 120 sampled batches and sampler states reconstruct exactly. Four fresh
  native validation passes agree with the recorded streamed scores within
  {score:.3e} relative error. Study evaluation contributes eight full scores.
- Candidate restoration runs once per backward: all 50 parameters and
  36,398,592 gradient bytes restored before clipping. Qualification verifies
  bitwise staged round trips. No completed scientific case was repeated.
- Twelve tensor artifacts, 628 complete-job memory intervals and twelve zero
  GPU boundaries are preserved. All 61 maintained file hashes remain unchanged.

The unchanged H120 training and audit functions are reused through scoped
factory/loss/backward adapters. The adapter restores patched methods on exit.
It supports one backward per batch; accumulation, distributed execution and
concurrent streams remain outside the tested scope. No maintained default
was changed.

## Next required evidence

Run longer matched training from the original initialization, then replicate
seeds and scale. Include the conventional native-loss control: matching the
chunked baseline alone cannot resolve H117's earlier native-versus-chunked
quality failure. Check endpoint quality and actual complete-job memory in every
run. The successful short gate does not rescue H117/H119 or establish the
original architectural/parameter-efficiency goal.

[Plan](compact_training_plan.md), [adapter](../results/compact_training_v1/adapter.py),
[audit](../results/compact_training_v1/audit.py), [receipt](../results/compact_training_v1/receipt.json).
"""
Path("research/compact_training_results.md").write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
note = """## Latest: compact storage passes short complete-training validation

[H125](compact_training_results.md): four 30-update continuations pass all
fixed gates. Complete-job VRAM falls 13.82–14.42%, with 4.67–5.57% event-time
overhead and about 0.0006% final NLL difference. Original AdamW/global clipping,
BF16 evaluation fallback and all transfers are included. Independent update,
batch and native-score audits pass. Total120 updates/122 backwards; no retries.

Next: longer training from original initialization, conventional native-loss
controls, multiple seeds and scale. No parameter reduction or default change;
H117/H119 failures and the broader research goal remain unresolved.

"""
current.write_text(
    head + "\n\n" + note + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
(ROOT / "README.before.md").write_bytes(readme.read_bytes())
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
intro = """Latest evidence: [short complete-training validation](research/compact_training_results.md)
measures 13.82–14.42% lower whole-job VRAM with passing numerical and validation
checks. Longer runs and multiple seeds remain required. See
[current research state](research/CURRENT_STATE.md) for the next experiment.

"""
readme.write_text(
    head
    + "\n\n"
    + intro
    + rest.replace("The latest [checkpoint-input", "The earlier [checkpoint-input", 1),
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
    ROOT / "README.before.md",
    *ROOT.glob("*/runs/*/*"),
    current,
    readme,
    Path("research/compact_training_plan.md"),
    Path("research/compact_training_results.md"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_gate="PASS_SHORT_COMPLETE_TRAINING",
    long_training_qualified=False,
    independent_audit_passed=True,
    training_updates=120,
    training_targets=491520,
    backwards=122,
    native_audit_scores=4,
    tensor_artifacts=12,
    maintained_hashes_verified=61,
    repeated_cases=0,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
