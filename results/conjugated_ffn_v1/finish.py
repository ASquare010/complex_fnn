"""Publish the structural falsification result and preserve all earlier evidence."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/conjugated_ffn_v1")
p, r, s, a = [
    read(ROOT / n) for n in ("protocol.json", "result.json", "summary.json", "audit.json")
]
assert not (ROOT / "receipt.json").exists()
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert sha("results/batch_scale_training_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/batch_scale_training_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "worker", "prepare_audit", "audit", "analyze"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
status = "PASS fixed structural screen" if s["passed"] else "REJECT fixed shared-permuted recipe"
rows = []
for c in s["checks"]:
    rows.append(
        f"| {c['task']} | {c['seed']} | {c['candidate_mse']:.6f} | {c['wide_gelu_ratio']:.3f} | {c['narrow_ratio']:.3f} | {c['memory_ratio']:.3f} | {c['time_ratio']:.3f} | {', '.join(k for k, v in c['gates'].items() if not v) or 'none'} |"
    )
means = [
    f"| {g['task']} | {g['arm']} | {g['mean']:.6f} | {g['median']:.6f} | {g['sample_variance']:.3e} | {g['mean_peak_mib']:.3f} | {g['mean_update_ms']:.3f} | {g['ranks']} |"
    for g in s["aggregates"]
]
counts = [f"| {arm} | {v['parameters']:,} | {v['macs']:,} |" for arm, v in p["counts"].items()]
nextstep = (
    "Only a stronger controlled study is earned; first add the remaining activation baselines and evaluate actual FFN captures before language insertion."
    if s["passed"]
    else "Do not insert this recipe into language models, tune kernels or claim a better FFN. The rank witness survived; the fixed learning/resource recipe did not. Use the linear and teacher-task controls to distinguish optimization or tying restrictions from the rank obstruction. Cubic nonlearning is inconclusive when wide controls also fail. A future change must identify a distinct mechanism or a specific failure diagnosis, not silently extend this failed budget."
)
report = f"""# H143: shared residual FFN with permutation conjugation

**{status}.** Independent audit passed: **{a["passed"]}**.
All 54 fits and 16,200 updates are retained. No active model/default changed.

## The hypothesis and what was proved

Use one narrow GELU branch twice inside a residual transformation, permuting the
coordinates before its second use and undoing that permutation on its output.
The two calls share every weight and bias. Compare same-basis sharing, independent
second-step weights, narrow GELU, wide GELU and wide SwiGLU.

For hidden width h, centered outputs of same-basis reuse stay in one h-dimensional
readout subspace. Permutation conjugation permits the sum of two readout subspaces,
with rank at most 2h. A coordinate-half-swap construction attains full rank at
h=d/2. Independent explicit-matrix evaluation and FP64 finite differences of both
inputs and tied weights pass. This elementary result removes one obstruction;
it proves neither efficient learnability nor a superset of every baseline family.
FP32 fitted-output ranks below are numerical diagnostics, not proofs.

A conditional residual-state Jacobian bound requires a sufficiently small branch
Jacobian norm. No such constraint is enforced here, and subtracting the input to
produce the FFN output removes the residual state's lower-bound guarantee. There
is no claim of eliminating vanishing or exploding gradients.

| Recipe | Trainable scalars | Matrix MACs per token |
|---|---:|---:|
{chr(10).join(counts)}

The candidate uses about half the wide controls' parameters at equal matrix MACs.
Both fixed permutation buffers are included in resource accounting (6,144 bytes
per model, including controls). Extra operations, memory traffic and kernel
launches are not represented by matrix MACs.

## Every task and seed

Ratios compare candidate to the named control; lower is better. Memory and time
ratios use wide GELU. Every candidate-to-SwiGLU comparison and both sharing
controls are included in the machine-readable gates.

| Task | Seed | Candidate reporting MSE | / Wide GELU | / Narrow GELU | GPU peak ratio | Update-time ratio | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

The positive-control gate requires both wide recipes to improve over the zero
predictor by at least 20%. A task failing this condition is an inconclusive
learning probe, never a candidate success. Gates additionally require at least
40% parameter savings, error within 5% of both wide controls, strict gains over
narrow GELU and same-basis sharing, GPU peak at most 90% and update time at most
115% of wide GELU. No averages rescue failed seeds.

## All recipes, three independent seeds

| Task | Recipe | Mean reporting MSE | Median | Sample variance | Mean CUDA peak MiB | Mean update ms | Numerical output ranks |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(means)}

Each of nine independent datasets has 4,096 training, 2,048 validation and 2,048
reporting input vectors of dimension 384. Targets are an independent orthogonal
linear map, a random wide GELU teacher, or mixed cyclic cubic products. Input
means/SDs and output centering/global scale use training rows only. No teacher
weights or held-out targets initialize students. Every arm uses identical
minibatch indices, FP32/TF32 off, AdamW at 0.001, zero decay and clipping at 1.
All 300 updates count; initial validation and final validation/reporting scores
are recorded. The final model is used, with no rate search or checkpoint selection.
This single-rate experiment closes or advances a recipe, not an entire function
family. Synthetic generalization is not language-model quality or convergence.

Update timing includes CPU gathering, transfer, gradient clearing and complete
synchronized optimization; the first 20 timings are excluded. Whole-job CUDA
peaks include scoring and diagnostics but exclude driver/context/other processes.
Datasets and their normalization remain CPU-resident. This is local FFN resource
accounting, not a full model's VRAM measurement or a cross-machine latency claim.

## Reproducibility and prior art

The independent audit regenerates all nine datasets bitwise and replays 108
validation/reporting scores using explicit linear/nonlinear operations without
Model.forward or the training scorer, a different batch size (257), and FP64
error accumulation. Every relative score discrepancy is at most 1e-5. All 54
exported states, 300-step histories, batch-stream hashes, parameter counts,
finite flags and 110 zero allocator boundaries verify. There are 162 study scores
and 108 audit scores, 2,073,600 sampled training vectors and no audit backwards.
Preflight finite-difference evaluations are separate CPU correctness work.

Weight sharing and shuffling are established: [Universal Transformers](https://arxiv.org/abs/1807.03819),
[ShaResNet](https://arxiv.org/abs/1702.08782),
[Shuffling RNNs](https://arxiv.org/abs/2007.07324) and
[permuted-feature compression](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cdt2.12060).
The precise combination's novelty is unresolved. Nothing here is a novelty,
SOTA, general approximation or broad-goal completion claim. H006, H100/H104 and
H106/H107 remain recorded; this result does not reopen their rejected recipes.
The remaining activation baselines are required before any broader claim.

## Next decision

{nextstep}

[Prospective plan and equations](conjugated_ffn_plan.md),
[summary](../results/conjugated_ffn_v1/summary.json),
[evidence receipt](../results/conjugated_ffn_v1/receipt.json).
"""
reportpath = Path("research/conjugated_ffn_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: conjugated shared-FFN falsification\n\n[H143](conjugated_ffn_results.md): **{status}.**\n54 fits, three tasks/seeds, 16,200 updates; rank/gradient preflight and independent\ndata/score audit pass. No active model/default changes.\n\n{nextstep}\nThe broader VRAM/parameter-efficient FFN research goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H143 structural FFN screen](research/conjugated_ffn_results.md): **{status}.** All recipes and failed gates are retained; no default changes."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/conjugated_ffn_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H143",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=16200,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
