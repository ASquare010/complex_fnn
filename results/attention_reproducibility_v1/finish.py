"""Publish the reproducibility decision and preserve all previous failed gates."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/attention_reproducibility_v1")
p, a = read(ROOT / "protocol.json"), read(ROOT / "audit.json")
assert not (ROOT / "receipt.json").exists()
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(p[key])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert a["verification_passed"] and a["backwards"] == a["training_updates"] == 0
for path, digest in read("results/native_buffer_layout_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", *p["policies"], "prepare_audit", "audit"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
results = [read(ROOT / mode / "result.json") for mode in p["policies"]]
backwards = sum(r["counts"]["backwards"] for r in results)
attempts = sum(r["counts"]["backward_attempts"] for r in results)
assert backwards == a["total_study_backwards"] <= attempts <= 36
artifacts = sum("artifact" in x for r in results for x in r["cases"])
boundaries = sum(len(r["boundaries"]) for r in results)
assert all(b["gpu"] == dict(allocated=0, reserved=0) for r in results for b in r["boundaries"])
qualified = [
    r["policy"] for r in results if r["passed"] and r["policy"].startswith("deterministic")
]
status = (
    "PASS scoped reproducibility: " + ", ".join(qualified)
    if qualified
    else "FAIL: no deterministic policy qualified"
)
rows = []
operators = []
for policy in a["policies"]:
    for group in policy["groups"]:
        if group["complete"]:
            rows.append(
                f"| {policy['policy']} | {group['dataset']} | {group['exact_to_native']}/6 | {group['same_arm_repeat_exact']}/3 | {group['max_relative_l2']:.3e} | {'pass' if group['passed'] else 'fail'} |"
            )
        else:
            rows.append(
                f"| {policy['policy']} | {group['dataset']} | incomplete | incomplete | — | fail |"
            )
    names = sorted({k for g in policy["groups"] if g["complete"] for k in g["operators"]})
    operators.append(f"- `{policy['policy']}`: " + ", ".join(f"`{k}`" for k in names))
next_step = (
    "Run a separately frozen complete-update and resource comparison under `"
    + qualified[0]
    + "`, including all transfers, original clipping/Adam and native validation. Match the policy in both arms. Compare its cost with ordinary execution separately before claiming practical efficiency."
    if qualified
    else "Investigate the recorded policy errors or residual mismatches before spending more training compute. Do not loosen the original gates."
)
failures = [dict(policy=r["policy"], failure=r["failure"]) for r in results if r["failure"]]
report = f"""# H129: whole-model reproducibility under explicit attention policies

**{status}.** This test addresses the unreliable native reference in H128.
It measures equality at two fixed model states, not training quality or speed.
All recorded scientific outcomes pass independent CPU evidence verification.

| Execution policy | Corpus | Exact versus native anchor | Exact same-arm repeats | Maximum relative L2 versus anchor | Gate |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

Each corpus/policy has three arms: native/resident checkpoint inputs,
native/CPU-offloaded checkpoint inputs, and H128 classifier/CPU-offloaded inputs.
Each arm is constructed twice from the same step-800 state and sampled batch.
The six anchor comparisons include the native anchor's trivial self-comparison;
the three repeat comparisons each compare two independently constructed probes.
Loss and all 50 parameter gradients must match bitwise. L2 differences are
reported for context and never substitute for the exactness gate.

Ordinary execution matches none of the three same-arm repeat pairs on either\ncorpus. Both deterministic policies match every comparison, with zero gradient\ndifference. The default deterministic policy retains the efficient-attention\noperator family; forcing the math backend is unnecessary for these fixed-state\nchecks. H128 memory savings cannot yet be transferred to this changed policy\nwithout a fresh resource measurement.\n\n## What the policy changes

`original_default` disables deterministic algorithms and unsets the cuBLAS
workspace environment variable. `deterministic_default` enables strict
deterministic algorithms and cuDNN deterministic execution, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, while retaining default SDPA selection.
`deterministic_math` uses that same deterministic configuration and forces
SDPA MATH around both forward and backward, including recomputation.
Four CPU threads, FP32 and disabled TF32 apply to all policies. Optimizer and
model states are unchanged; no compact RMSNorm or gradient staging is used.

The policy changes a bundle of settings. The result does not isolate one
flag as the cause. These are fresh sequential processes on the same RTX 4070
Laptop GPU and UV-managed Python/PyTorch installation. CPU profiler events in
every probe identify the dispatched attention operators:

{chr(10).join(operators)}

Math attention is decomposed and need not expose a fused SDPA backward entry.\nThe recorded operators identify dispatch, not every internal CUDA reduction
choice. Profiling may affect scheduling; no throughput or memory-saving claim
is drawn from this instrumented experiment. A passing policy demonstrates
repeatability for these probes, not a guarantee across hardware or versions.
See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## Evidence and limits

Actual execution: **{backwards} successful backwards**, {attempts} backward
attempts, {backwards * 4096:,} diagnostic targets, **zero optimizer updates**.
There are {artifacts} first-repetition gradient artifacts and {boundaries}
zero GPU allocator boundaries. Every probe records gradient hashes, exact
loss, model/optimizer/input/sampler provenance, finiteness and actual attention
operators. All 61 maintained file hashes remain unchanged.

An independent CPU verifier rehashes the saved gradients, recomputes their
per-tensor and whole-gradient distances using NumPy, checks all repeat-hash
comparisons and verifies budgets/provenance. Second-repetition tensors are
represented by recorded hashes and comparison metrics rather than extra
full tensor files; numeric distances for those repetitions cannot be
independently recomputed from retained tensors. Their bitwise comparisons
are checked against first-repetition hashes. No additional GPU audit was run.

Policy failures: `{json.dumps(failures)}`. No completed probe was retried.
H127's operator failure and H128's full-model replay failure remain recorded;
this separate policy comparison does not retroactively change either gate.
Parameter count remains 9,099,648. No activation, FFN or algorithmic novelty
is claimed. No maintained default changes.

## Next decision

{next_step}
The broad VRAM, quality and parameter-efficient FFN research goal remains open.

[Prospective plan](attention_reproducibility_plan.md),
[independent verification](../results/attention_reproducibility_v1/audit.json),
[evidence receipt](../results/attention_reproducibility_v1/receipt.json).
"""
report_path = Path("research/attention_reproducibility_results.md")
report_path.write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
current.write_text(
    head
    + f"\n\n## Latest: explicit-policy reproducibility test\n\n[H129](attention_reproducibility_results.md): **{status}.**\n{backwards} backwards, zero updates; {artifacts} gradient artifacts, CPU verification\ncomplete, all 61 maintained files unchanged. No training/resource claim.\n\n{next_step}\nPrior failed gates remain unchanged; the broad research goal remains open.\n\n"
    + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
readme.write_text(
    head
    + f"\n\nLatest: [reference reproducibility](research/attention_reproducibility_results.md).\n**{status}.** Fixed-state evidence only; training and resource validation remain.\n\n"
    + rest.replace("Latest diagnostic:", "Earlier diagnostic:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
for path in ROOT.glob("*/result.json"):
    path.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    *ROOT.glob("*.before.md"),
    ROOT / ".gitignore",
    *ROOT.glob("*/*.pt"),
    *ROOT.glob("*/*.json"),
    *ROOT.glob("*/*.gz"),
    current,
    readme,
    report_path,
    Path("research/attention_reproducibility_plan.md"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    qualified_policies=qualified,
    scientific_status=status,
    backwards=backwards,
    backward_attempts=attempts,
    training_updates=0,
    artifacts=artifacts,
    maintained_hashes_verified=61,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
