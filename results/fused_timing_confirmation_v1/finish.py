"""Publish the longer mirrored-order confirmation without revising H149."""

import ast
import textwrap
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/fused_timing_confirmation_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert a["status"] == "VERIFIED" and (ROOT / "audit_exit.txt").read_text().strip() == "0"
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/fused_reconstruction_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/fused_reconstruction_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest


def timed_body(path):
    text = Path(path).read_text()
    start = text.index("        optimizer.zero_grad(set_to_none=True)")
    end = text.index("        history.append(", start)
    return ast.dump(ast.parse(textwrap.dedent(text[start:end])))


assert timed_body(ROOT / "worker.py") == timed_body("results/fused_reconstruction_v1/worker.py")
gate = (
    "PASS scoped mirrored-order resource confirmation"
    if a["passed"]
    else "FAIL mirrored-order resource confirmation"
)
rows = []
for c in a["aggregates"]:
    rows.append(
        f"| {c['batch']} | {c['seed']} | {c['arm']} | {c['peak_bytes'] / 2**20:.3f} | {c['timing']['event_ms']['median']:.3f} | {c['timing']['wall_ms']['median']:.3f} | {max(c['repeat_ratio'].values()):.3f} | {c['stable']} |"
    )
comps = []
for c in a["comparisons"]:
    v = c["ratios"]
    failures = ", ".join(k for k, value in c["gates"].items() if not value) or "none"
    comps.append(
        f"| {c['batch']} | {c['seed']} | {c['control']} | {v['peak']:.4f} | {v['event_ms']:.3f} | {v['wall_ms']:.3f} | {failures} |"
    )
unstable = [
    dict(
        index=c["index"],
        arm=c["arm"],
        batch=c["batch"],
        seed=c["seed"],
        repeat=c["repeat"],
        stability=c["stability"],
    )
    for c in r["cases"]
    if max(c["stability"].values()) > 1.15
]
write_json(ROOT / "unstable_segments.json", unstable)
report = f"""# H150: longer, mirrored-order resource confirmation

**{gate}.** This is an independent timing protocol using H149's fixed kernel
and saved states. H149's rejected timing screen is preserved. The confirmation
uses longer windows, mirrored execution order and an additional across-repeat
drift gate. It changes no architecture, mathematical operation or threshold.
Resource confirmation does not establish learning quality or a breakthrough.

## Frozen design

For each batch128/2048 and seed223/239/251, restore H149's step12 model and Adam
state for full GELU checkpoint, full SwiGLU checkpoint and learned fused reversible
stack. Rotate the three-arm order by fixture, then reverse it: ABC CBA.
Every segment restores its source state; these are two independent30-update
segments, not60 continuous updates. The exact initial model and optimizer tensors
are checked against saved source tensors before training. Source hashes are frozen.

Discard10 warmups and pool the remaining20 updates from each repeat for40 timing
observations per arm/fixture. Gradient reset is outside timing; forward, backward,
clipping and AdamW are inside. The timed-body AST matches H149 exactly. Input
gradients and all fixed matrix buffers are included. Final no-grad scoring enters
peak but not timing. Model/optimizer loading and QR setup are outside timing;
setup cost is retained per segment ({min(c["setup_ms"] for c in r["cases"]):.1f}–
{max(c["setup_ms"] for c in r["cases"]):.1f}ms). Cached kernel code is unchanged.

## All fixture measurements

Peak is the maximum across both repeats. Timing is the pooled40-update median.
Repeat drift is the larger CUDA/wall ratio between repeat medians. Stable requires
both repeat drift and both segments' half-window median ratios<=1.15.

| Batch | Seed | Arm | Peak MiB | CUDA ms | Wall ms | Repeat drift | Stable |
|---:|---:|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Every candidate comparison

Ratios are fused/control. Every fixture must satisfy parameter ratio<=.8,
peak<=.9, CUDA/wall<=1.15 and all candidate/control stability checks. Parameter
ratio is about.0026; the fused model's4.5MiB of fixed matrices is still counted
in allocated GPU peak. This is allocator memory, not total driver VRAM.

| Batch | Seed | Checkpointed control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---:|---:|---:|---|
{chr(10).join(comps)}

{len(unstable)} individual segments fail the15% half-window stability limit;
identities and values are in unstable_segments.json. Across-repeat failures, if
any, appear above. No selected reruns, clock changes, new thresholds or extensions
were used. Means, medians, sample variances and all raw per-update times are saved.
Forty observations per arm are repeated updates, not40 independent model seeds.

## Independent audit and limits

All36 final saved states independently score on CPU in FP64 using explicit forward
formulas and batch257, without the custom backward. Maximum relative score error
is{max(c["relative_error"] for c in a["checks"]):.3e}, below1e-5. Six datasets
regenerate bitwise, immutable matrices match regeneration, parameters/moments and
first/final gradient diagnostics are finite. Optimizer counters equal42 and exact
parameter/buffer/optimizer byte counts verify. Source restoration is checked by
the worker, and source hashes/protocol bindings by the audit; no independent replay
of each Adam update is claimed.

Maximum relative final parameter difference between matched repeats is
{max(c["parameter_relative_error"] for c in a["repeat_states"]):.3e}; maximum
relative diagnostic-loss difference is
{max(c["loss_relative_error"] for c in a["repeat_states"]):.3e}. This describes
these30-update windows, not long-trajectory equivalence or convergence.

There were1,080 optimizer/backward calls,37 clean allocated/reserved CUDA
boundaries,36 final fixed-batch scores plus36 CPU audit scores. Audit initialized
no CUDA and performed no backward. FP32/TF32off, four CPU threads, ordinary
8.125MiB cuBLAS workspace, AdamW LR.003 betas(.9,.95), decay0,clip1. Hardware and
kernel are the same local RTX4070 Laptop GPU setup used in H149. No telemetry,
clock/power change or model/default modification. The initial PowerShell process
suffered a CLR crash after file writes and before native formatting; inspection
confirmed no protocol or GPU run existed. Formatting/checks completed through
cmd, then every experiment stage ran once. All frozen and prior receipt hashes verify.

This reuses six previously tested fixtures and18 saved states deliberately; it
is not fresh-seed or new-task generalization. Fixed-shape and affine controls
remain in H149; this confirmation isolates the three primary resource arms.
No loss-based model selection or held-out quality evaluation occurred.

A pass qualifies this implementation for a separately frozen learning study with
unrelated nonlinear targets, matched parameter/compute controls, fixed-shape and
affine ablations, ordinary activations, independent seeds and held-out metrics.
A fail leaves timing/resource eligibility unresolved under these thresholds.
The full research goal remains open in either case.

[Plan](fused_timing_confirmation_plan.md),
[raw segments](../results/fused_timing_confirmation_v1/result.json),
[audit](../results/fused_timing_confirmation_v1/audit.json),
[receipt](../results/fused_timing_confirmation_v1/receipt.json).
"""
reportpath = Path("research/fused_timing_confirmation_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: mirrored-order fused resource confirmation\n\n[H150](fused_timing_confirmation_results.md): **{gate}**.\n36 saved-state segments/1,080 updates, longer windows and repeat-stability gates,\nindependent saved-state scoring verified. H149 remains rejected. Resource\nevidence does not establish quality; the broad research goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H150 mirrored-order confirmation](research/fused_timing_confirmation_results.md):\n{gate}. Learning quality remains unproven."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and not {"unused_cache", "triton_cache"}.intersection(f.parts)
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/fused_timing_confirmation_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H150",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=1080,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
