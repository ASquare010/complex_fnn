"""Independently check saved CPU replay outputs and publish integration evidence."""

import math
from pathlib import Path

import numpy as np
import torch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/training_memory_integration_v1")
p, r = [read(ROOT / name) for name in ("protocol.json", "result.json")]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files"):
    hashes(p[field])
for stage in ("test", "prepare", "compiler_tests", "worker"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert (ROOT / "suite_exit.txt").read_text().strip() == "1"
assert "6 failed, 126 passed" in (ROOT / "suite.log").read_text()
assert "6 passed, 2 deselected" in (ROOT / "compiler_tests.log").read_text()
assert p["classifier_ast_equal"] and r["training_updates"] == r["backwards"] == 12
assert len(r["boundaries"]) == 13 and all(
    b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"]
)
compiler = read(ROOT / "compiler.json")
assert sha(compiler["path"]) == compiler["sha256"]
checks = []
for f in p["fixtures"]:
    peers = {
        c["arm"]: c for c in r["cases"] if c["dataset"] == f["dataset"] and c["seed"] == f["seed"]
    }
    assert len(peers) == 2
    for c in peers.values():
        assert c["restored"] and sha(c["artifact"]["path"]) == c["artifact"]["sha256"]
    old, new = [
        torch.load(peers[k]["artifact"]["path"], map_location="cpu", weights_only=True)
        for k in ("frozen", "maintained")
    ]
    errors = {}
    for kind in old:
        numerator = denominator = 0.0
        worst = 0.0
        assert old[kind].keys() == new[kind].keys()
        for name in old[kind]:
            x = old[kind][name].numpy().astype(np.float64)
            y = new[kind][name].numpy().astype(np.float64)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            delta = float(np.square(x - y).sum())
            scale = float(np.square(x).sum())
            numerator += delta
            denominator += scale
            worst = max(worst, math.sqrt(delta / max(scale, 1e-60)))
        errors[kind] = dict(
            global_relative=math.sqrt(numerator / max(denominator, 1e-60)), tensor_relative=worst
        )
    assert all(
        errors[k]["global_relative"] <= 1e-5 and errors[k]["tensor_relative"] <= 1e-4
        for k in ("raw", "clipped")
    )
    assert all(errors[k]["global_relative"] <= 1e-6 for k in ("parameters", "moments"))
    assert peers["maintained"]["peak_bytes"] <= peers["frozen"]["peak_bytes"] + 2**20
    checks.append(
        dict(
            dataset=f["dataset"],
            seed=f["seed"],
            errors=errors,
            frozen_peak_mib=peers["frozen"]["peak_bytes"] / 2**20,
            maintained_peak_mib=peers["maintained"]["peak_bytes"] / 2**20,
        )
    )
assert r["passed"] and len(checks) == 6
write_json(
    ROOT / "independent_check.json",
    dict(passed=True, checks=checks, gpu_backwards=0, training_updates=0),
)
rows = [
    f"| {c['dataset']} | {c['seed']} | {c['frozen_peak_mib']:.3f} | {c['maintained_peak_mib']:.3f} | {c['errors']['raw']['global_relative']:.3e} | {c['errors']['raw']['tensor_relative']:.3e} | {c['errors']['parameters']['global_relative']:.3e} | {c['errors']['moments']['global_relative']:.3e} |"
    for c in checks
]
report = f"""# H141: maintained opt-in training-memory helper

**Integration verification passed.** The qualified classifier and final-block
input offload are available through explicit APIs in
[src/core/training_memory.py](../src/core/training_memory.py).
[Usage and limitations](training_memory_usage.md) include a complete invocation.
No existing maintained source or training default changed.

## Maintained versus frozen GPU replay

| Corpus | Source seed | Frozen peak MiB | Maintained peak MiB | Raw gradient relative L2 | Worst tensor gradient relative L2 | Parameter relative L2 | Moment relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

All six H140 source model/Adam/sampler states were replayed once under each path,
with alternating arm order: 12 complete updates/backwards and 49,152 targets.
The two paths start from identical states and batches. Correct suffix wrapping,
parameter object identity, state paths, finite states, Adam step 801 and complete
wrapper restoration are checked. Thirteen zero allocated/reserved boundaries
verify GPU cleanup. Saved raw/clipped gradients, parameters and moments were
independently compared on CPU with NumPy FP64. Every prospective numerical and
memory limit passed. Per-pair bitwise results remain in the raw result and are
not a gate under ordinary nondeterministic attention.

The classifier's forward arithmetic has identical AST to H127's qualified
native-buffer forward; backward arithmetic has identical AST to H128's layout
repair. Public argument validation happens before that unchanged core. The CUDA
adapter uses the same native save_on_cpu hook around whole-block checkpointing.
The maintained context additionally preflights the selected blocks, rejects
unsupported overrides/nesting and bypasses transfer on CPU. It restores methods
on exceptions. It does not change recomputation settings implicitly.

## Test coverage and environment recovery

Sixteen new tests passed: native loss/input/weight derivatives, column-major and
strided layouts, masked labels, all-ignored semantics, finite-difference checking,
caller tensor ownership, tied model gradients, zero/invalid suffix sizes, atomic
preflight and restoration after nesting/errors.

The full suite initially had **126 passes and six failures**. All six were in
existing optional Triton inference tests and failed compiler discovery before
kernel execution. The UV interpreter's sysconfig directory differs from the
repository package directory. A bounded rerun set process-local CC to the already
installed Triton TinyCC executable: **all six passed**. The original failure log,
compiler path/hash, recovery plan and focused rerun remain recorded. Thus all
132 collected tests have passing coverage across those runs; this was not a
single green full-suite invocation. No compiler was installed and no old test
or source was changed to suppress a failure.

## Scope and next experiment

These checks establish integration equivalence at the previously qualified scale.
They do not provide new validation scores or meaningful throughput measurements:
one-step peaks include gradient/state diagnostics and cannot replace whole-job
training benchmarks. The earlier H140 short-segment savings remain supporting
prior evidence; H138's long-run failure remains unchanged. Parameters are unchanged.
This is an experimental opt-in helper, not a new activation, SOTA result or proof
of the broad research goal.

Next test a different workload scale with the maintained API, ordinary controls,
complete memory/compute accounting and explicit quality checks. Keep the separate
parameter-efficient FFN research objective open; do not mistake memory-implementation
integration for novel neuron geometry or proof of sustained deployment throughput.

[Prospective plan](training_memory_integration_plan.md),
[GPU replay](../results/training_memory_integration_v1/result.json),
[independent CPU comparison](../results/training_memory_integration_v1/independent_check.json),
[evidence receipt](../results/training_memory_integration_v1/receipt.json).
"""
reportpath = Path("research/training_memory_integration_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = """## Latest: maintained opt-in memory helper

[H141](training_memory_integration_results.md): **integration verification passed**.
Explicit helper, unchanged classifier arithmetic, 16 new tests and 12 GPU replay
updates across all six source states. All 132 tests have passing coverage after
a documented compiler-path rerun of six existing Triton tests. Old maintained
sources/defaults remain unchanged. [Usage](training_memory_usage.md).

Next: qualify another workload scale through the maintained API with ordinary
controls and memory/compute/quality checks. H138 remains failed; sustained
throughput and the broader parameter-efficient FFN objective remain open."""
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = """Latest: [H141 opt-in memory helper](research/training_memory_integration_results.md)
passed source-equivalence, numerical and lifecycle checks. [Usage](research/training_memory_usage.md).
Ordinary defaults are unchanged; broader performance/generalization remains open."""
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
    Path("research/training_memory_usage.md"),
    Path("research/training_memory_integration_plan.md"),
    Path("src/core/training_memory.py"),
    Path("tests/test_training_memory.py"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H141",
    status="EVIDENCE_VERIFIED",
    gate="PASS integration verification",
    training_updates=12,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
