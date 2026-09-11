"""Report partial-offload tradeoffs and the prospectively eligible variant."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/partial_offload_training_v1")
p, s, a = [read(ROOT / name) for name in ("protocol.json", "summary.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
for path, digest in read("results/ordinary_long_training_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "seal", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
selected = s["selected_arm"]
status = (
    f"PASS short qualification: {selected}" if selected else "FAIL new-variant short qualification"
)
comparisons = []
for c in s["comparisons"]:
    q = c["ratios"]
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {c['dataset']} | {c['repeat']} | {c['arm']} | {100 * (1 - q['peak_mib']):.2f}% | {q['event_ms']:.4f} | {q['wall_ms']:.4f} | {q['nll']:.8f} | {failed} |"
    )
metrics = [
    f"| {m['index']} | {m['dataset']} | {m['arm']} | {m['peak_mib']:.3f} | {m['host_mib']:.3f} | {m['event_ms']:.3f} | {m['wall_ms']:.3f} | {m['nll']:.6f} | {m['stability']:.4f} |"
    for m in s["metrics"]
]
calibration = []
for group in a["calibrated"]:
    for c in group["checks"]:
        calibration.append(
            f"| {group['dataset']} | {group['arm']} | {c['kind']} | {c['repeat']} | {c['error']['global_relative']:.3e} | {c['limits']['global']:.3e} | {c['error']['tensor_relative']:.3e} | {c['limits']['tensor']:.3e} | {c['passed']} |"
        )
next_step = (
    f"Validate {selected} from fresh initialization against ordinary native training on both corpora and three seeds, using complete-update timing and native scoring. Preserve H136's failed full-offload result; this short continuation does not prove fresh convergence or robust long-run efficiency. The broader FFN/activation parameter-efficiency objective remains open."
    if selected
    else "Do not extend long training or change defaults. Inspect which limits failed for buffer0 and buffer4; choose the next falsifiable modification without dropping unfavorable repeats or relaxing thresholds."
)
report = f"""# H137: fewer checkpoint-input transfers

**{status}.** Independent clipping/Adam/native-score/gradient-noise audit:
**{a["passed"]}**. Eligible new variants: **{", ".join(s["eligible"]) or "none"}**.
Every corpus and mirrored repeat must pass for a new variant to qualify.
The eight-block arm is a control; it cannot override H136's long-run failure.

| Corpus | Repeat | Variant | GPU allocation saved vs ordinary | Complete CUDA ratio | Wall ratio | NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

`ordinary` uses native loss without input offload. All buffer variants use the
same H128 loss. The suffix gives the number of offloaded blocks: zero, last four,
or all eight. The original native CPU-save adapter is scoped through a view of
selected blocks; model registration, parameter identities and numerical layer
operations remain unchanged. There are still 9,099,648 trainable parameters.

Each FP32 checkpoint input has nominal payload 8 x 512 x 384 x 4 bytes = 6 MiB.
Selecting four blocks therefore transfers/retains half the nominal checkpoint
input payload of eight. This is not a formula for whole-job peak reduction:
the peak can move, and the table reports measured allocation including all phases.
No activation or offload-algorithm novelty is claimed.

## Measurements

| Run | Corpus | Variant | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing halves ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Both corpora use H136 ordinary seed101 step800 checkpoints, including optimizer
and sampler states. Mirrored order is ordinary, buffer0, buffer4, buffer8,
buffer8, buffer4, buffer0, ordinary. Repeats share one starting state; they are
not independent seeds. The frozen selection rule prefers greater worst-case
memory savings among buffer0/4 passing every comparison, then runtime. Buffer8
remains in the table even when it passes this short test.

Limits are memory ratio <=0.90, complete CUDA/wall <=1.15, NLL <=1.01, candidate
pinned host peak <=128 MiB and timing halves <=1.15, plus numerical/source/telemetry
checks. No limit changed and no failed repeat is averaged away. Raw records retain
means, medians and sample variance. Candidate timings include forward, backward,
clipping, Adam, recomputation and transfers; they exclude sampling, gradient reset,
validation, serialization and inventory. Whole-job GPU peak includes all phases;
driver/context and other processes are excluded. Pinned peak is not total CPU RSS.

## Independent numerical evidence

Inherited NumPy checks verify original clipping and first Adam updates/moments.
Every final checkpoint is independently scored through native code, and all batches
are replayed. Parameter/optimizer/sampler hashes and finite states are checked.
Ordinary attention is nondeterministic, so H135's bounded native-repeat calibration
is retained rather than claiming bitwise model trajectories. Native-repeat noise,
bitwise flags and final weight differences are retained in the audit JSON.

| Corpus | Variant | Gradient | Repeat | Global error | Global limit | Max tensor error | Tensor limit | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(calibration)}

Global limit=min(1e-5,max(1e-6,10*native noise)); tensor limit=min(1e-4,max(1e-5,
10*native noise)). Native noise must itself satisfy the caps. This two-repeat
noise diagnostic is not a statistical confidence interval. The existing exact
operator qualification remains complementary evidence.

## Reproducibility and limits

480 updates/backwards and 1,966,080 training targets. Thirty-two full study scores
plus sixteen independent native scores; 48 tensor artifacts, 1,072 memory
intervals and 49 zero allocator boundaries. All 61 maintained files and frozen
source/input/library hashes verify. The actual wrapped block indices are recorded
and checked for every run. Original FP32, default attention/workspace, TF32 off,
four threads, UV-managed Python and PYTHONMALLOC=pymalloc are shared by all arms.

Read-only GPU telemetry is sampled every 200 ms, coverage is required during
measured updates, and monitors stop in finally. Raw samples and sensor summaries
remain available; unavailable readings stay missing. No hardware power/clock
settings change. These data do not establish a causal explanation for H136's
runtime outlier. H136's failed gate remains unchanged. No maintained default is
modified, and this test establishes neither fresh training nor parameter reduction.

The initial CPU sealing process crashed in the compact JSON encoder before
writing an aggregate or manifest. One bounded CPU recovery streamed the existing
case JSON text verbatim into the aggregate, preserving numerical values. Original
source, failure log and exit are retained. No scientific work was repeated.

## Next decision

{next_step}

[Prospective plan](partial_offload_training_plan.md),
[summary](../results/partial_offload_training_v1/summary.json),
[receipt](../results/partial_offload_training_v1/receipt.json).
"""
report_path = Path("research/partial_offload_training_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
rest = rest.replace("## Latest:", "## Previous:", 1)
current.write_text(
    head
    + f"\n\n## Latest: partial checkpoint-input offload\n\n[H137](partial_offload_training_results.md): **{status}.**\n480 updates; zero/four/eight-block transfer comparison and independent audit.\n\n{next_step}\nHistorical failures and the full research goal remain open.\n\n"
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
    + f"\n\nLatest: [partial checkpoint offload](research/partial_offload_training_results.md).\n**{status}.** Explicit memory/transfer tradeoff with matched controls.\nFresh validation and the broader research goal remain open.\n\n"
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
files += [report_path, Path("research/partial_offload_training_plan.md"), current, readme]
receipt = dict(
    study="H137",
    status="EVIDENCE_VERIFIED",
    gate=status,
    selected_arm=selected,
    goal_achieved=False,
    training_updates=480,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
