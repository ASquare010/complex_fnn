"""Publish H158 without changing H157's failed zero-host-memory gate."""

import statistics as st
from pathlib import Path

from results.checkpoint_host_isolation_v1.study import (
    PRIOR,
    ROOT,
    check_protocol,
    read,
    sha,
    verify,
    write,
)

p, old = check_protocol()
recovery = read(ROOT / "recovery_protocol.json")
verify(recovery["inputs"])
assert sha(ROOT / "protocol.json") == recovery["original_protocol"]
for stage in ("recovery", "audit_recovery"):
    assert (ROOT / (stage + "_exit.txt")).read_text() == "0"
verify(read(PRIOR / "receipt.json")["files"])
a = read(ROOT / "audit.json")
assert a["status"] == "EVIDENCE_VERIFIED" and len(a["fixtures"]) == 6 and len(a["cases"]) == 18
assert not (ROOT / "receipt.json").exists()
rows = []
for f in a["fixtures"]:
    g = next(v for v in f["numeric"] if v["arm"] == "fp16")
    h = f["host"]
    rows.append(
        f"| {f['dataset']} | {f['seed']} | {100 * g['global_relative_l2']:.5f}% | {100 * g['max_tensor_relative_l2']:.5f}% | {100 * (1 - f['gpu_native_ratio']):.2f}% | {f['gpu_offload_ratio']:.4f} | {h['native0']['allocated_bytes.peak']:,} / {h['offload4']['allocated_bytes.peak']:,} / {h['fp16']['allocated_bytes.peak']:,} | {f['passed']} |"
    )
status = (
    "PASS fresh-process resource/numerical qualification"
    if a["passed"]
    else "FAIL fresh-process qualification"
)
resources = []
for arm in ("native0", "offload4", "fp16"):
    values = [c["peak_bytes"] / 2**20 for c in a["cases"] if c["arm"] == arm]
    resources.append(f"| {arm} | {min(values):.3f} | {max(values):.3f} | {st.mean(values):.3f} |")
summary = []
for dataset in ("wikitext2", "tinystories"):
    values = [
        next(v for v in f["numeric"] if v["arm"] == "fp16")["global_relative_l2"]
        for f in a["fixtures"]
        if f["dataset"] == dataset
    ]
    summary.append(
        f"| {dataset} | {st.mean(values):.7g} | {st.median(values):.7g} | {st.variance(values):.7g} |"
    )
report = f"""# H158: Isolated host-allocation qualification

**{status}.** Eighteen fresh-process GPU backwards, six source fixtures,
zero optimizer updates. This resolves a resource-measurement question; it does
not establish training quality, throughput, parameter savings or novelty.

## Why this follows H157

H157 passed numerical and GPU-allocation subchecks but failed its absolute-zero
pinned-host gate. Its single process retained about 64 MiB from an earlier offload
probe. H158 preserves that failed result and preregisters separate fresh processes
for native0, offload4 and FP16 on every fixture. Each process exits before the next
starts; arm order alternates by fixture. The maintained API and H157 codec/probe
are reused unchanged. No cached bytes are retrospectively subtracted.

The new claim is **no additional pinned allocator footprint relative to native**,
not zero pinned bytes. FP16 must have allocated and active peaks no greater than
native, and allocated peak at most 1% of offload4. All three processes must start
with zero allocated host bytes. This is a new H158 gate, not a relaxed H157 pass.
PyTorch allocated-byte statistics include active and cached allocations; active
bytes are reported separately. See the sources and discussion in
[H157](checkpoint_fp16_results.md). The five-byte incidental allocation's source
has not been traced, so no stronger claim about host transfers is made.

## Results: every seed

| Corpus | Seed | Global gradient error | Max tensor error | GPU allocation saved vs native | GPU / offload4 | Pinned allocated peak bytes: native / offload4 / FP16 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Absolute allocator peaks across all six fixtures (MiB):

| Arm | Minimum | Maximum | Mean |
|---|---:|---:|---:|
{chr(10).join(resources)}

Across the three seeds, gradient relative-L2 statistics (dimensionless; sample variance):

| Corpus | Mean | Median | Variance |
|---|---:|---:|---:|
{chr(10).join(summary)}

The GPU gates are FP16/native <=0.90 and FP16/offload4 <=1.02. FP16 numerical
limits remain global relative L2 <=0.002, maximum tensor <=0.02, cosine >=0.99999,
relative loss error <=1e-6. Offload retains exact-path limits 1e-5 global,
1e-4 maximum tensor and 1e-6 loss. Every fixture must pass; no seed is discarded.
Full tensor errors, cosines, active and allocated host peaks remain in audit.json.

## Scope and verification

Two corpora, seeds 101/113/127, ordinary H156 step800 checkpoints, batch16,
context512, 9,099,648 parameters, eight blocks, FP32, TF32 off, four CPU threads,
ordinary attention and default cuBLAS workspace. Each arm reloads identical model,
Adam moments, dataset and sampler state, then uses the identical next batch.
The candidate stores eight block-checkpoint inputs as FP16 on GPU and recomputes
with FP32 restored inputs. Original forward evaluation remains FP32; the backward
is approximate, as established by H157's local semantic check and rounding audit.

Loaded model, moments and data remain resident. GPU peaks retain the larger of
construction and later forward/backward/gradient-copy/check peaks. There is no
Adam step, and these are not whole-update or isolated-inference memory measurements.
Driver/context, unrelated processes and total CPU RSS are outside these counters.
No cold-process timing is reported as throughput. No extra diagnostic GPU runs.

Independent NumPy FP64 calculations recompute every saved gradient error and cosine;
all 18 gradients and 36 GPU boundaries are checked. Batch, model and optimizer
hashes match both new controls and original H157. The FP16 save/unpack count and
stored bytes are checked for all eight inputs. Original sources, data and checkpoint
hashes verify. Each child has its own exclusive log and exit record. No completed case was rerun.

## Preserved startup failure

The original worker1 crashed during PyTorch import with Windows access violation
3221225477, before model construction or result files. Case00 completed and was
retained. A prospectively documented single retry added the prior H157 launcher's
unused bytecode-cache prefix; it then continued cases2-17. The original failed
logs and exit files remain. The cause of the import crash is not established.
The recovery audit differs only in which exit record it accepts for worker1.
No numerical recipe, model code or gate changed; successful backward count is 18.
[Recovery record](checkpoint_host_isolation_recovery.md).

## Decision

{"Proceed to a warmed, temporally paired complete-update timing comparison of FP16, offload4 and native. Only a timing survivor earns fresh multi-seed quality training. This result establishes neither of those outcomes." if a["passed"] else "Do not allocate long training. Preserve the failing criterion and diagnose it before any further promotion."}
The broader goal remains open. Activation-compressed training is established
prior art; no novel activation or breakthrough is claimed.

[Prospective plan](checkpoint_host_isolation_plan.md),
[full audit](../results/checkpoint_host_isolation_v1/audit.json),
[original failed H157 audit](../results/checkpoint_fp16_v1/audit.json).
"""
report_path = Path("research/checkpoint_host_isolation_results.md")
assert not report_path.exists()
report_path.write_text(report, encoding="utf-8")
for path, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith(title)
    rest = text[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    link = (
        "research/checkpoint_host_isolation_results.md"
        if path.name == "README.md"
        else "checkpoint_host_isolation_results.md"
    )
    intro = f"Latest: [H158 fresh-process qualification]({link}): **{status}.**\n18 GPU backwards; zero optimizer updates. H157's failed zero-host gate is preserved.\nComplete-update speed and fresh quality validation remain outstanding. Goal open.\n\n"
    path.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.iterdir()
    if f.is_file() and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/checkpoint_host_isolation_plan.md"),
    Path("research/checkpoint_host_isolation_recovery.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write(
    ROOT / "receipt.json",
    dict(
        study="H158",
        status="EVIDENCE_VERIFIED",
        gate=status,
        gpu_backwards=18,
        training_updates=0,
        goal_achieved=False,
        files={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
