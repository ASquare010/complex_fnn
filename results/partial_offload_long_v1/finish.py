"""Publish the selected variant's fresh-seed evidence and provenance."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/partial_offload_long_v1")
p, s, a = [read(ROOT / name) for name in ("protocol.json", "summary.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/partial_offload_training_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "prepare_audit", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
status = (
    "PASS four-block fresh-training gate" if s["passed"] else "FAIL four-block fresh-training gate"
)
comparisons = []
for c in s["comparisons"]:
    q = c["ratios"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {c['dataset']} | {c['seed']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f} | {q['wall_ms']:.4f} | {q['nll']:.6f} | {failed} |"
    )
metrics = [
    f"| {m['dataset']} | {m['seed']} | {m['arm']} | {m['peak_mib']:.3f} | {m['host_mib']:.3f} | {m['event_ms']:.3f} | {m['wall_ms']:.3f} | {m['nll']:.6f} | {m['stability']:.4f} |"
    for m in s["metrics"]
]
aggregates = []
for group in s["aggregates"]:
    v = group["stats"]
    aggregates.append(
        f"| {group['dataset']} | {group['arm']} | {v['nll']['mean']:.6f} | {v['nll']['median']:.6f} | {v['nll']['variance']:.3e} | {v['peak_mib']['mean']:.3f} | {v['event_ms']['mean']:.3f} |"
    )
next_step = (
    "Add an explicit opt-in maintained memory helper for this qualified scope, keeping ordinary defaults and historical sources intact. Verify it against these saved artifacts with numerical and resource tests, then test another workload scale. This establishes a scoped memory implementation, not novel activation geometry or parameter-efficient FFN superiority; the broader objective remains open."
    if s["passed"]
    else "Do not promote the configuration. Retain every seed and inspect the failed quality/runtime/resource gate. Choose a narrower falsifiable follow-up without relaxing limits, reordering completed runs or using aggregate means to rescue failures."
)
figure = Path("research/figures/partial_offload_long.png")
figure_note = (
    "![Fresh-seed convergence and resource evidence](figures/partial_offload_long.png)"
    if figure.exists()
    else "All numerical evidence is retained in the tables and raw artifacts."
)
report = f"""# H138: fresh validation of four-block offload

**{status}.** Independent initialization/native-gradient/checkpoint audit:
**{a["passed"]}**. All six seed/corpus comparisons must pass individually.
The candidate was selected prospectively by H137; H136's eight-block long-run
failure remains unchanged.

| Corpus | Seed | GPU allocation saved | Complete CUDA ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

{figure_note}

Candidate `buffer4` combines H128 buffer classifier with H137's unchanged CPU
checkpoint-input adapter on blocks 4-7. `ordinary` uses native loss and retains
checkpoint inputs on GPU. All other layer/optimizer/data operations remain
identical. There are 9,099,648 trainable parameters in both arms. This is a
memory optimization, not a novel activation or parameter-count reduction.

## Three independent seeds

Seeds 101, 113 and 127 start from six original step-zero states with empty Adam
moments and train 800 updates per arm/corpus. The audit regenerates every initial
state exactly. Arm order alternates by fixture: three ordinary-first and three
candidate-first pairs. No completed run is reordered, dropped or selected as
representative. Aggregates describe every seed and cannot rescue a failed gate.

| Corpus | Arm | Mean final NLL | Median final NLL | Sample NLL variance | Mean GPU peak MiB | Mean complete CUDA ms |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(aggregates)}

## Per-run evidence

| Corpus | Seed | Arm | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing block ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Per-seed gates remain allocation ratio <=0.90, complete CUDA/wall <=1.15, final
NLL <=1.01, candidate pinned host peak <=128 MiB and timing stability <=1.15.
The latter compares three 260-update mean-wall-time blocks after 20 warmups.
All phase records, per-step histories and seed mean/median/sample variance are
retained. Complete timing spans forward/backward/clipping/Adam, recomputation
and transfers; sampling, gradient reset, validation, serialization and inventory
are excluded from timing but included in whole-job CUDA peak accounting.
Driver/context/other processes are excluded; pinned peak is not total CPU RSS.

## Independent audit and reproducibility

9,600 updates and 39,321,600 training targets. Twelve initial gradient probes
plus twelve independent native replays give 9,624 backwards. There are 48 full
study validation scores and 42 independent native scores, covering initial and
200/400/800 checkpoints. Forty-eight tensor artifacts, 19,332 memory intervals,
43 zero allocator boundaries; all 61 maintained files and frozen source/input/
library hashes verify. Recorded wrapped block indices must equal [4,5,6,7] for
the candidate and [] for ordinary training.

The unchanged H136 loop and H117 independent audit verify regenerated initial
states, every batch, saved model/optimizer/sampler hashes, finite states and native
validation at every checkpoint. Initial gradients are recomputed without custom
loss/offload under existing caps: 1e-5 global and 1e-4 tensor relative L2, with
loss/score relative error <=1e-6. Ordinary attention is nondeterministic; this is
bounded numerical equivalence rather than bitwise trajectory identity. H128's
operator proof and H137's short noise calibration remain supporting evidence.

FP32, TF32 off, ordinary attention, default 8.125-MiB workspace, four CPU threads,
UV-managed Python/PYTHONMALLOC=pymalloc, original AdamW/LR/clipping, batch 8 and
context 512 are shared. Raw-case aggregation preserves original numerical JSON
text. Passive GPU telemetry samples every 200 ms and stops in finally; all runs
require measured-window coverage. Missing sensors remain missing. No hardware
power or clock settings change, and sensor correlation cannot establish causation.

These are two text corpora at one model scale and 800 updates, not evidence of
cross-domain generality, final convergence, SOTA superiority or a new FFN.
All earlier failed gates remain recorded. No maintained default is changed by
this experiment.

## Next decision

{next_step}

[Prospective plan](partial_offload_long_plan.md),
[summary](../results/partial_offload_long_v1/summary.json),
[receipt](../results/partial_offload_long_v1/receipt.json).
"""
report_path = Path("research/partial_offload_long_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + f"\n\n## Latest: four-block fresh-training validation\n\n[H138](partial_offload_long_results.md): **{status}.**\n12 fresh 800-update runs, three seeds and two corpora; independent audit: {a['passed']}.\n\n{next_step}\nHistorical failures and the full research goal remain open.\n\n"
    + rest,
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("Latest:", "Earlier:", 1)
readme.write_text(
    head
    + f"\n\nLatest: [four-block fresh validation](research/partial_offload_long_results.md).\n**{status}.** Three seeds, two corpora and independent checkpoint audit.\nThe broader FFN research goal remains open.\n\n"
    + rest,
    encoding="utf-8",
    newline="\n",
)
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [report_path, Path("research/partial_offload_long_plan.md"), current, readme]
if figure.exists():
    files.append(figure)
receipt = dict(
    study="H138",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=9600,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
