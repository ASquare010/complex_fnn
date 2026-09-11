"""Publish the audited approximate-gradient/backward-memory preflight."""

import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/checkpoint_fp16_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0" and a["status"] == "EVIDENCE_VERIFIED"
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
hashes(a["inputs"])
assert sha("results/batch_scale_long_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/batch_scale_long_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
status = (
    "PASS approximate-gradient and backward-memory preflight"
    if a["passed"]
    else "FAIL pinned-host gate; numerical and GPU-memory checks pass"
)
# Diagnose the failed frozen gate without changing its definition or raw evidence.
assert not a["passed"] and all(c["passed"] for c in a["checks"])
assert all(
    c["memory_native_ratio"] <= 0.9 and c["memory_offload_ratio"] <= 1.02 for c in a["comparisons"]
)
for fixture in p["fixtures"]:
    cs = {
        c["arm"]: c
        for c in r["cases"]
        if (c["dataset"], c["seed"]) == (fixture["dataset"], fixture["seed"])
    }
    for arm in ("offload4", "fp16", "native1"):
        assert cs[arm]["host"]["allocated_bytes.current"] == 67108869
    for arm in ("fp16", "native1"):
        assert cs[arm]["host"]["active_bytes.current"] == 0
        assert cs[arm]["host"]["active_bytes.peak"] == 5
rows = []
for c in a["comparisons"]:
    g = next(
        v
        for v in a["checks"]
        if (v["dataset"], v["seed"], v["arm"]) == (c["dataset"], c["seed"], "fp16")
    )
    rows.append(
        f"| {c['dataset']} | {c['seed']} | {100 * g['global_relative_l2']:.5f}% | {100 * g['max_tensor_relative_l2']:.5f}% | {g['global_cosine']:.9f} | {100 * (1 - c['memory_native_ratio']):.2f}% | {c['memory_offload_ratio']:.4f} | {c['passed']} |"
    )
resources = []
for arm in p["arms"]:
    cs = [c for c in r["cases"] if c["arm"] == arm]
    resources.append(
        f"| {arm} | {min(c['peak_bytes'] for c in cs) / 2**20:.2f}-{max(c['peak_bytes'] for c in cs) / 2**20:.2f} | {max(c['host']['allocated_bytes.peak'] for c in cs) / 2**20:.2f} | {st.mean(c['peak_bytes'] for c in cs) / 2**20:.2f} |"
    )
quant = []
for block in range(8):
    cs = [c for c in a["diagnostics"] if c["block"] == block]
    quant.append(
        f"| {block} | {max(c['max_abs'] for c in cs):.5f} | {100 * max(c['relative_l2'] for c in cs):.5f}% | {sum(c['underflow_count'] for c in cs):,} |"
    )
exact = [c for c in a["checks"] if c["arm"] != "fp16"]
checks = read(ROOT / "checks.json")
report = f"""# H157: FP16 checkpoint storage as an alternative to CPU offload

**{status}.** This is a numerical/storage result: 36 GPU backwards and six
separate diagnostic forwards, with zero optimizer updates. It does not establish
faster complete updates or preserved training quality. H156's qualified exact-path
memory helper remains unchanged; the broader research objective stays open.

## Mechanism and mathematical meaning

Each of the eight whole-block checkpoints saves only its FP32 input as FP16 on
GPU, then restores FP32 for recomputation. The original forward uses FP32 inputs;
weights, optimizer moments and classifier/log-softmax buffers stay FP32. Identity
packing and native repeats control for hook semantics and nondeterministic noise.

For a block and fixed upstream vector, the observed backward matches the native
block VJP at Q(x), where Q is the FP16 roundtrip, while the output remains F(x).
The CPU input-and-parameter check passes with maximum relative error
{max(checks["local_vjp_relative_errors"]):.3e}. Across a stack, independently rounded
saved inputs generally do not describe one coherent original forward trajectory.
This is approximate gradient computation, not exact differentiation of the FP32
forward, and no unbiasedness claim follows from rounding.

The codec matches pointer, shape, stride and storage offset within each block's
checkpoint scope; unexpected nonempty saved tensors raise an error. Empty
checkpoint sentinels pass through. It closes over primitive metadata, not the
original input tensor, so retaining a closure cannot silently retain FP32 storage.
All temporary forward overrides are restored. Exactly eight inputs are saved and
unpacked in each identity/FP16 backward. Diagnostic CPU input copies are taken in
separate forwards and excluded from the clean memory probes.

[PyTorch saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
provide the mechanism. [GACT](https://proceedings.mlr.press/v162/liu22v.html) establishes
generic activation-compressed training as prior art. This prototype tests a limited
checkpoint-storage tradeoff; it is not claimed as a novel activation or architecture.

## Every source fixture

All six H156 ordinary step800 checkpoints are used, with their original Adam
moments and sampler positions: two corpora, seeds 101/113/127, batch16/context512,
9,099,648 parameters. Each arm samples the same next batch and preserves its model
and optimizer state hashes. All native/control gradients use FP32 autograd.

| Corpus | Seed | Global gradient error | Max tensor error | Gradient cosine | Allocation saved vs native | Allocation / offload4 | All gates |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

The FP16 thresholds were fixed before execution: global relative L2 <=0.002,
maximum tensor relative L2 <=0.02, global cosine >=0.99999 and scalar loss error
<=1e-6. These are approximate-gradient guardrails, not H156's exact-path limits
and not a guarantee of long-run quality. Per-tensor values remain in audit.json.
For native repeats, buffered loss, identity hooks and offload4, the original
1e-5 global / 1e-4 tensor / 1e-6 loss limits apply. Their maximum observed global
relative gradient error is {max(c["global_relative_l2"] for c in exact):.3e} and
maximum tensor error is {max(c["max_tensor_relative_l2"] for c in exact):.3e}.

## Measured allocation and host memory

| Arm | GPU peak range MiB | Maximum pinned host MiB | Mean GPU peak MiB |
|---|---:|---:|---:|
{chr(10).join(resources)}

The gate requires FP16 peak <=1.02 times offload4, <=0.90 times native0, and zero
pinned-host peak, on every fixture. Loaded model, Adam moments and datasets remain
resident. Peaks include construction, forward, backward and gradient-copy/check
phases; the larger construction or later peak is retained. There is **no optimizer
step**, so these are not complete-update training peaks, and no cold timing is
reported as throughput. They are not isolated inference memory. Driver/context
and other processes are excluded. Pinned host memory is not total CPU RSS.

**The host gate fails on all six fixtures.** Each FP16 probe follows offload4 in
one process. Offload4 leaves 67,108,869 pinned bytes owned by the allocator, and
both FP16 and the subsequent native repeat retain that same allocation. Their
active pinned allocation returns to zero, with a five-byte active peak. Even the
first native probe reports five allocated bytes. The boundary helper resets host
peaks but does not release the pinned cache. Consequently this design cannot
establish the promised zero-pinned-host result for FP16 in isolation.

The [PyTorch host allocator documentation](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.host_memory_stats.html)
distinguishes allocated bytes (active plus cached) from active bytes. The cache
carryover diagnosis follows from that definition and the measured arm sequence;
the source of the five-byte activity is not established. We neither subtract the
cache retrospectively nor replace the frozen gate with an active-byte threshold.
The original audit remains failed. This is a host-measurement qualification
failure, not evidence that FP16 checkpoint storage needs 64 MiB of pinned buffers.
A follow-up must preregister fresh-process controls and distinguish incidental
host transfers from checkpoint offload before claiming a host-memory advantage.
No additional GPU runs or optimizer updates were spent on this diagnosis.

Eight FP16 inputs and four GPU-resident FP32 inputs both nominally occupy 48 MiB
at this shape; casts, unpacking temporaries and actual tensor lifetimes still need
measurement. The buffer-only and identity controls expose the cost of keeping all
eight FP32 inputs. Native loss additionally uses the ordinary classifier path.

## Activation rounding and finite-range scope

The separate diagnostic forward saves every full original checkpoint input to CPU.
Independent NumPy FP16 conversion verifies all 48 tensors, including the following
maxima over source fixtures. Underflows count nonzero FP32 inputs rounded to zero;
counts are summed over six input tensors per block.

| Block | Maximum input magnitude | Maximum relative rounding L2 | Underflow count |
|---:|---:|---:|---:|
{chr(10).join(quant)}

All actual rounded inputs and all gradients are finite. A separate CPU cast of
100,000 to FP16 produces infinity, deliberately demonstrating the range limitation.
The prototype is not an arbitrary-range production API; it does not hide overflows
with clipping. Tensor gradient errors and approximate long-run effects must be
requalified at other model sizes, scales and precisions.

## Audit, startup repair and decision

An independent NumPy FP64 audit recomputes every gradient error and cosine from
saved full gradient arrays; it also recomputes every diagnostic rounding error.
There are 43 clean zero-allocation/reservation GPU boundaries. All sources,
checkpoints, batch identities, unchanged model/moment hashes and hook counts verify.
The full-model reference is measured native FP32 autograd twice, not CPU FP64.
The local CPU semantic check contributes two additional backwards, no updates.

The first launch failed before any stage log, protocol or numerical work because
operator.py shadowed Python's standard-library operator. Renaming it to codec.py
and changing two imports repaired startup; original contents and the error record
are preserved in operator.before.txt and startup_failure.md. No recipe or numerical
threshold changed. The single retry passed local preflight before GPU execution.

{"This earns a warmed, temporally paired complete-update timing comparison against offload4 and ordinary checkpointing. Only surviving that comparison warrants a fresh multi-seed quality test; this preflight alone earns neither claim." if a["passed"] else "The numerical and GPU-allocation subchecks pass, but the frozen host gate fails. The candidate remains unqualified: resolve host-memory measurement with fresh-process controls before complete-update timing or long-training allocation. Preserve this failure; do not interpret it as a numerical rejection."}

[Prospective plan](checkpoint_fp16_plan.md),
[raw measurements](../results/checkpoint_fp16_v1/result.json),
[NumPy audit](../results/checkpoint_fp16_v1/audit.json).
"""
report_path = Path("research/checkpoint_fp16_results.md")
report_path.write_text(report, encoding="utf-8")
for path, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = path.read_text(encoding="utf-8")
    assert old.startswith(title)
    if path.name == "README.md":
        rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H157 FP16 checkpoint preflight](research/checkpoint_fp16_results.md):\n**{status}.** No optimizer updates; complete-update speed and quality remain untested.\n\n"
    else:
        rest = old[len(title) :].lstrip().replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: FP16 checkpoint-storage preflight\n\n[H157](checkpoint_fp16_results.md): **{status}**.\n36 GPU backwards, six diagnostic forwards, full-gradient and activation-rounding\nNumPy audit. Approximate gradients use separately frozen caps. Zero optimizer\nupdates; startup module-name collision and pre-protocol repair retained.\nHost-cache carryover prevents qualification of the zero-host-memory gate.\nComplete-update timing and fresh quality validation remain necessary. Goal open.\n\n"
    path.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.iterdir()
    if f.is_file() and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/checkpoint_fp16_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
receipt = dict(
    study="H157",
    status="EVIDENCE_VERIFIED",
    gate=status,
    training_updates=0,
    gpu_backwards=36,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
print({k: v for k, v in receipt.items() if k != "files"})
