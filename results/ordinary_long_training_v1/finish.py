"""Publish three-seed evidence without conflating it with broad FFN success."""

import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/ordinary_long_training_v1")
p, s, a = [read(ROOT / name) for name in ("protocol.json", "summary.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/ordinary_complete_training_v1/receipt.json")["files"].items():
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
    "PASS three-seed long-training gate" if s["passed"] else "FAIL three-seed long-training gate"
)
comparisons = []
for c in s["comparisons"]:
    q = c["ratios"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {c['dataset']} | {c['seed']} | {c['control']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f} | {q['wall_ms']:.4f} | {q['nll']:.6f} | {failed} |"
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
paired = []
for dataset in ("wikitext2", "tinystories"):
    for control in ("ordinary", "native"):
        rows = [c for c in s["comparisons"] if c["dataset"] == dataset and c["control"] == control]
        stats = {
            k: dict(mean=st.mean(v), median=st.median(v), variance=st.variance(v))
            for k in ("peak_mib", "event_ms", "wall_ms", "nll")
            if (v := [r["ratios"][k] for r in rows])
        }
        paired.append(dict(dataset=dataset, control=control, stats=stats))
write_json(ROOT / "paired_seed_statistics.json", paired)
next_step = (
    "Integrate the qualified ordinary-policy memory path as an explicit opt-in research configuration, preserving native defaults and scope checks. Verify maintained integration against these artifacts, then test another workload scale and task before broader claims. The activation/FFN parameter-efficiency objective remains separate and unresolved."
    if s["passed"]
    else "Do not promote the configuration. Identify the failing seed/gate from the table, retain all three seeds and choose one narrower follow-up. Do not change limits or substitute aggregate means for a failed individual run."
)
figure = Path("research/figures/ordinary_long_training.png")
figure_note = (
    "![Three-seed convergence and resource evidence](figures/ordinary_long_training.png)"
    if figure.exists()
    else "All numerical evidence is retained in the tables and machine-readable artifacts."
)
report = f"""# H136: fresh three-seed ordinary-policy training

**{status}.** Independent initialization/native-gradient/checkpoint audit:
**{a["passed"]}**. All twelve corpus/seed/control comparisons must pass.
This tests an architecture-independent memory implementation on two text corpora;
it does not prove a novel activation, parameter reduction or SOTA superiority.

| Corpus | Seed | Control | GPU allocation saved | Complete CUDA ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

Candidate `reuse` combines H128 buffer classifier with checkpoint-input CPU
offload. `native` uses native loss with the same offload; `ordinary` keeps native
loss and GPU checkpoint inputs. All share ordinary attention, default 8.125-MiB
workspace, FP32/TF32off, four threads, original AdamW/clipping and 9,099,648
parameters. These are fresh step-zero states, with empty optimizer moments,
not continuations from the favorable short-training fixtures.

{figure_note}

## Independent seeds and aggregate results

Each of seeds 101, 113, 127 trains 800 updates per arm and corpus. Latin-square
arm order balances position across seeds and reverses for TinyStories. All
initial states are regenerated exactly by the independent audit. Means below
summarize all seeds; they never override an individual failed gate.

| Corpus | Arm | Mean final NLL | Median final NLL | Sample NLL variance | Mean GPU peak MiB | Mean complete CUDA ms |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(aggregates)}

[Paired seed statistics](../results/ordinary_long_training_v1/paired_seed_statistics.json)
retain mean, median and sample variance of all memory/runtime/NLL ratios.
Three seeds are a limited replication, not a universal guarantee.

## Per-run resource and quality evidence

| Corpus | Seed | Arm | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing block ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Gates remain allocation ratio<=0.90, complete CUDA/wall<=1.15, final NLL<=1.01,
candidate pinned host peak<=128MiB, and timing stability<=1.15 for every run.
The latter uses three 260-update mean-wall-time blocks after 20 warmups. Timing
spans forward/backward/clipping/Adam including offload transfers and checkpoint
recomputation. It excludes sampling, gradient clearing, validation, serialization
and inventory. Whole-job allocated peak includes those phases. CUDA allocation
excludes driver/context and other processes; pinned host peak is not total RSS.
Reserved memory, complete histories, moments/grad norms and layer diagnostics
are retained. Native validation at0/200/400/800 tracks convergence throughout.

## Verification and limits

14,400 updates and 58,982,400 training targets. 18 initial gradient probes plus 18
independent native replays give14,436 total backwards. 72 full study scores and 60
independent native scores cover initial and200/400/800 checkpoints. 72 tensor
artifacts, 28,998 memory intervals, 61 zero allocator boundaries. All 61 maintained
files and frozen fixture/source/library hashes verify.

The independent audit regenerates all six step-zero models and empty Adam
states, replays every batch, checks model/optimizer/sampler hashes and finite
states at each saved endpoint, and scores through native code. Initial gradients
are recomputed without custom loss/offload under the existing 1e-5 global and 1e-4
tensor relative-L2 caps; loss/score relative error remains<=1e-6. Ordinary
attention is nondeterministic, so this is bounded numerical equivalence rather
than a claim of bitwise trajectory identity. H135's shorter numerical audit and
H128's exact operator qualification remain complementary evidence.

The H117 loop was reused with only timestamp fields added; its syntax tree was
checked against declared edits before freezing. Compact atomic JSON output
preserves numerical content and reduces exposure to previously observed Python
encoder failures. The UV-managed runtime and allocator choice are recorded.

Passive GPU telemetry runs every 200 ms per worker and stops in finally. Coverage
is required during measured updates; raw clocks, temperature and power are
retained with unavailable fields marked missing. No power/clock settings change.
Sensor correlation and cross-study timing differences cannot diagnose a kernel
or thermal cause. Two text corpora, this scale and 800 updates do not establish
cross-domain generality or final-convergence behavior. Historical failures stay
unchanged, and no maintained training default is changed by this experiment.

## Next decision

{next_step}
The broad research goal remains open.

[Prospective plan](ordinary_long_training_plan.md),
[summary](../results/ordinary_long_training_v1/summary.json),
[receipt](../results/ordinary_long_training_v1/receipt.json).
"""
report_path = Path("research/ordinary_long_training_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + f"\n\n## Latest: fresh three-seed ordinary-policy training\n\n[H136](ordinary_long_training_results.md): **{status}.**\n18 fresh 800-update runs, three seeds, two corpora; independent audit:{a['passed']}.\n\n{next_step}\nHistorical failures and the full research goal remain open.\n\n"
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
    + f"\n\nLatest: [fresh three-seed memory validation](research/ordinary_long_training_results.md).\n**{status}.** 18 fresh runs with native controls and independent checkpoint audit.\nThe broader FFN research goal remains open.\n\n"
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
files += [report_path, Path("research/ordinary_long_training_plan.md"), current, readme]
if figure.exists():
    files.append(figure)
receipt = dict(
    study="H136",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=14400,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
