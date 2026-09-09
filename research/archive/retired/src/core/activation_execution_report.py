"""Report the bounded rational-activation execution audit, including failures."""

import hashlib
import json
import zipfile
from pathlib import Path

from src.core.reproducibility import sha256, write_json


def report(root=Path("results/activation_execution_v1")):
    protocol = json.loads((root / "protocol.json").read_text())
    complete = (root / "result.json").exists()
    outcome = json.loads((root / ("result.json" if complete else "failure.json")).read_text())
    records = {}
    critical = (
        "src/core/activation_execution.py",
        "src/learnable_activation_ffn/compiled_product.py",
        "src/learnable_activation_ffn/__init__.py",
        "src/core/transformer.py",
        "src/core/optimization.py",
    )
    with zipfile.ZipFile(root / "source.zip") as archive:
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    for case in protocol["cases"]:
        path = root / case / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        assert r["status"] == "complete" and r["checkpoint_unchanged"]
        assert sha256(Path("results/runs") / r["run"] / "checkpoint.pt") == r["checkpoint_sha256"]
        assert r["original_parameter_objects_preserved"] and r["validation_targets"] == 322688
        assert all(
            r["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in critical
        )
        assert len(r["step_seconds"]) == 20 and len(r["minibatch_hashes"]) == 30
        if complete:
            assert sha256(path) == outcome["result_hashes"][case]
        records[case] = r
    assert all(
        r["minibatch_hashes"] == next(iter(records.values()))["minibatch_hashes"]
        for r in records.values()
    )
    if complete:
        compiled = records["rational_compiled"]
        gates = {
            f"memory_within_ten_percent_{ref}": compiled["peak_allocated_vram_bytes"]
            <= 1.1 * records[ref]["peak_allocated_vram_bytes"]
            for ref in ("full_gelu", "full_swiglu")
        }
        assert gates == outcome["gates"] and all(gates.values()) == outcome["execution_passes"]
        verdict = "PASS" if outcome["execution_passes"] else "FAIL"
    else:
        compiled = json.loads((root / "rational_compiled" / "failure.json").read_text())
        gates = {}
        verdict = "FAILED BEFORE COMPLETE MEASUREMENT"
    lines = [
        "# Rational activation: isolated execution audit",
        "",
        f"The frozen execution decision is **{verdict}**. This tests compiler fusion of the learned rational activation/product on the selected checkpoint. It does not retrain or replace the twelve-trial architecture screen.",
        "",
        "Read the [frozen protocol](activation_execution_plan.md), [architecture result](learnable_activation_results.md) and [raw audit](../results/activation_execution_v1/protocol.json).",
        "",
        "## Fresh isolated training-memory comparison",
        "",
        "| Case | Allocated peak MiB | Reserved peak MiB | Serial training tokens/s |",
        "|---|---:|---:|---:|",
    ]
    for case, r in records.items():
        lines.append(
            f"| {case.replace('_', ' ')} | {r['peak_allocated_vram_bytes'] / 2**20:.2f} | {r['peak_reserved_vram_bytes'] / 2**20:.2f} | {r['training_tokens_per_second']:.0f} |"
        )
    if complete:
        lines += [
            "",
            f"Compiled/native rational allocated-memory ratio is {outcome['compiled_vs_native_memory_ratio']:.4f}. The 10% full-reference limits use these fresh measurements, not historical peaks.",
            "",
            "| Gate | Decision |",
            "|---|---|",
        ]
        for key, value in gates.items():
            lines.append(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |")
    else:
        lines += [
            "",
            "The compiler case failed before a complete memory result. Its failure record and worker log are retained; missing measurements are not counted as passes.",
            "",
            f"Recorded error: `{compiled['error']}`.",
        ]
    lines += ["", "## Numerical and gradient fidelity", ""]
    if "fidelity" in compiled:
        lines += ["| Check | Measured |", "|---|---:|"]
        for key, value in compiled["fidelity"].items():
            lines.append(f"| {key.replace('_', ' ')} | {value} |")
    if "compiled_nll" in compiled:
        lines += [
            "",
            f"Original full-validation NLL is {compiled['native_nll']:.9f}; compiled-product NLL is {compiled['compiled_nll']:.9f}, a {compiled['compiled_relative_nll_percent']:+.7f}% change over 322,688 targets. Original checkpoint hashes and learned parameter objects are preserved.",
        ]
    lines += [
        "",
        "## What was measured",
        "",
        "Each worker loads its selected seed-17, 200-step checkpoint and AdamW state. Correctness checks and original-checkpoint validation precede profiling. Model and optimizer state are restored; all workers use the same thirty sampled minibatches. Ten warmup updates precede twenty measured updates. The updates exercise real BF16 forward/backward, clipping and AdamW allocation, but are transient: no resulting weights are saved or scored as a new model.",
        "",
        "Only the rational activation and its product are compiled. Projections, attention and optimizer remain native; gate checkpointing remains enabled. Static fullgraph Inductor uses low-precision cast emulation and no CUDA graphs. Compilation and CPU reference buffers are excluded from steady-state GPU peaks. PyTorch allocated memory does not include all driver allocation. The native full controls and serial timings do not support an equal-compiler full-model speed claim.",
        "",
        "A passing memory/derivative audit is still not a compiled training-trajectory replication. The original architecture screen's memory failure remains a historical result. A bounded compiled training repeat, independent seeds, longer training and corpus transfer are required before stronger claims. No new activation primitive or full-network stability theorem is established.",
        "",
    ]
    if Path("results/activation_training_repeat_v1/result.json").exists():
        lines.extend(
            [
                "",
                "Follow-up: the [full 200-step compiled training repeat](activation_training_repeat_results.md) fails quality despite passing memory. The isolated PASS above does not establish training-trajectory equivalence.",
                "",
            ]
        )
    Path("research/activation_execution_results.md").write_text(
        chr(10).join(lines), encoding="utf-8"
    )
    write_json(
        Path("results/activation_execution_summary.json"),
        {
            "verdict": verdict,
            "outcome": outcome,
            "records": records,
            "compiled_fidelity": compiled.get("fidelity"),
            "protocol_sha256": sha256(root / "protocol.json"),
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "peaks_mib": {
                    k: v["peak_allocated_vram_bytes"] / 2**20 for k, v in records.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    report()
