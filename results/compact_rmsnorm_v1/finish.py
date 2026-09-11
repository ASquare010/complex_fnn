"""Publish the qualified component result without claiming training success."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_rmsnorm_v1")
p, r, s = [read(ROOT / n) for n in ("protocol.json", "result.json", "summary.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
previous = read("results/gradient_staging_v1/receipt.json")
for path, digest in previous["files"].items():
    assert (
        sha(ROOT / "CURRENT_STATE.before.md" if path == "research/CURRENT_STATE.md" else path)
        == digest
    )
for stage in ("prepare", "study", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert s["combined_qualified"]
rows = []
for c in s["comparisons"]:
    baseline = c["normalization"] == "ordinary" and c["gradients"] == "resident"
    status = "baseline" if baseline else "PASS" if c["passed"] else "FAIL: memory"
    rows.append(
        f"| {c['dataset']} | {c['normalization']} | {c['gradients']} | {c['peak_mib']:.3f} | {100 * (1 - c['memory_ratio']):.2f}% | {c['event_ratio']:.4f}x | {c['wall_ratio']:.4f}x | {status} |"
    )
for dataset in ("wikitext2", "tinystories"):
    for gradients in ("resident", "staged"):
        pair = [
            next(
                c
                for c in s["comparisons"]
                if c["dataset"] == dataset
                and c["gradients"] == gradients
                and c["normalization"] == kind
            )
            for kind in ("ordinary", "compact")
        ]
        assert pair[0]["peak_mib"] - pair[1]["peak_mib"] == 12
measurements = [c["measurement"] for c in r["cases"]]
ge = max(c["global_error"] for c in measurements)
te = max(c["tensor_error"] for c in measurements)
le = max(c["loss_error"] for c in measurements)
fd = max(d["absolute_error"] for c in r["qualification"]["records"] for d in c["directions"])
report = f"""# H124: compact RMSNorm backward plus gradient staging

**The combined storage candidate passes the two-corpus diagnostic gate.**
It saves 13.73–14.33% peak allocated VRAM relative to ordinary RMSNorm with
resident gradients, at 4.36–6.41% measured event-time overhead. Both arms
already offload checkpoint inputs and use chunked FP32 classifier loss.
This qualifies a separate complete-training test; it does not establish
language quality, parameter reduction, production throughput or a breakthrough.

| Corpus | RMSNorm | Gradients | Peak MiB | Saved | Event ratio | Wall ratio | Gate |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Compact RMSNorm removes exactly 12 MiB from the peak in each matched gradient
policy on each corpus. Alone it saves only 4.27–4.35%, failing the fixed 10%
requirement. Gradient staging alone still fails. Their combination passes
all memory, time and host-memory gates. Old H123 failures remain failed.

## Equation and implementation

For r=(mean(x²)+epsilon)^(-1/2), y=w*x*r, incoming gradient g:

    dw = sum_over_tokens(g*x*r)
    dx = r*(g*w) - x*r^3*mean(g*w*x)

The custom autograd function saves x, w and r. Its backward computes dw first,
then uses an owned dx buffer and addcmul_ for the correction. Saved inputs and
upstream gradients are never overwritten. This is the standard RMSNorm
function with an analytical first derivative; changed floating-point operation
order can still affect training. FP32 is the execution scope, FP64 the
mathematical qualification. BF16/FP16 and higher-order derivatives are not
supported by this prototype.

In exact arithmetic, unweighted RMS normalization has tangential Jacobian
eigenvalue r and radial eigenvalue epsilon*r³. Thus its spectral norm is at
most 1/sqrt(epsilon), and multiplication by w gives the bound
max(abs(w))/sqrt(epsilon). This local bound does not prevent exploding or
vanishing gradients through a whole network. No new expressivity is claimed.

## Verification and scope

Three FP64 shapes, including zero inputs, pass reference-autograd comparisons
at 1e-10 tolerances. Six joint finite-difference directions have maximum
absolute error {fd:.3e}. Input/upstream hashes are unchanged by backward.
Full-model gradients match H121's independently audited references with maximum
global symmetric relative L2 {ge:.3e}, maximum tensor relative L2 {te:.3e},
and relative loss difference {le:.3e}. Saved gradients are finite. Staged
round trips are bitwise exact and every parameter hook runs once per backward.

All source weights, optimizer moments/counters and sampler states remain
unchanged. All ten GPU allocator boundaries are zero. Complete diagnostic
peaks include construction, warmup, forward/backward, gradient restoration,
serialization and state checks. Combined pinned-host allocation remains about
112.034 MiB. CUDA allocated/reserved and pinned active/cached statistics are
recorded separately. Driver/context memory is excluded from allocator figures.

The unchanged H123 run_case is reused through a temporary constructor adapter
and separate output roots; only normalization construction differs. The
maintained repository code and all 61 protected file hashes remain unchanged.

Budget: **86 backwards** (six qualification plus 80 full-model), twelve extra
finite-difference loss forwards, zero optimizer updates, 327,680 diagnostic
target evaluations and eight gradient artifacts. Three warmups and seven timed
repetitions per case; mean, median, sample variance and every repetition are
saved. Arm order reverses across corpora. No scientific case was repeated.
These are two fixed trained states, not independent training seeds.

## Next required evidence

Run a separately frozen complete-training comparison including the unchanged
AdamW update and global gradient clipping, all round-trip transfers, validation
and complete-job VRAM. Verify first updates and endpoint quality before
extending seeds or scale. The FP32-only wrapper needs an explicit ordinary
RMSNorm path for existing BF16 evaluation. Earlier H117 quality and H119
numerical failures remain unresolved; this short result does not erase them.

[Plan](compact_rmsnorm_plan.md), [implementation](../results/compact_rmsnorm_v1/norm.py),
[receipt](../results/compact_rmsnorm_v1/receipt.json).
Prior art includes [RMSNorm](https://arxiv.org/abs/1910.07467) and established
[normalization kernel optimization](https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/).
No algorithmic novelty or parameter reduction is claimed.
"""
Path("research/compact_rmsnorm_results.md").write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
note = """## Latest: compact RMSNorm plus gradient staging qualifies a training test

[H124](compact_rmsnorm_results.md) passes the two-corpus diagnostic gate:
13.73–14.33% lower peak VRAM, with 4.36–6.41% event-time overhead, relative
to ordinary RMSNorm/resident gradients with checkpoint-input offload. Compact
RMSNorm alone saves 12 MiB but fails 10%; gradient staging alone still fails.
The combined arm passes memory, time and host-memory gates. Gradient checks,
finite differences and bitwise staging restoration pass. No updates yet.

86 backwards, eight gradient artifacts, zero repeated cases; 61 maintained
file hashes unchanged. This is an FP32 first-derivative storage implementation,
not a new activation or parameter reduction. Next: frozen complete-training
validation with original AdamW/clipping, BF16 evaluation fallback, actual job
VRAM and endpoint quality. Earlier quality/numerical failures remain failed.
The broader research goal is still open.

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
    *ROOT.glob("*/runs/*/*"),
    current,
    Path("research/compact_rmsnorm_plan.md"),
    Path("research/compact_rmsnorm_results.md"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_gate="PASS_COMBINED_DIAGNOSTIC",
    complete_training_qualified=False,
    gradient_checks_passed=True,
    backwards=86,
    optimizer_updates=0,
    diagnostic_targets=327680,
    gradient_artifacts=8,
    maintained_hashes_verified=61,
    repeated_cases=0,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
