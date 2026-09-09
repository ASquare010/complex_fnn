"""Verify and report the multi-head comparator's bounded pipeline qualification."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch

from src.core.multihead_integration import PLAN, configuration
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/multihead_integration_v1")


def report():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    assert result["status"] == "complete" and result["earns_training_screen"]
    assert result["plan_sha256"] == protocol["plan_sha256"] == sha256(PLAN)
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    assert initial["old_variants_exact"] == 28 and initial["calibration_ratio_passes"]
    rows = {}
    for case in result["cases"]:
        path = ROOT / case["variant"]
        assert sha256(path / "result.json") == case["result_sha256"]
        r = json.loads((path / "result.json").read_text())
        assert (
            r["status"] == "PASS"
            and r["provenance"]["source_files"] == protocol["provenance"]["source_files"]
        )
        assert r["precision_error"]["relative_l2"] <= 0.05 and r["precision_error"]["finite"]
        assert r["unique_batch_tokens"] == 2048 and r["training_token_exposures"] == 20480
        assert sha256(path / "checkpoint.pt") == r["checkpoint_sha256"]
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert (
            checkpoint["step"] == 10
            and checkpoint["model_config"] == r["model"]
            and checkpoint["training_config"] == r["training"]
        )
        assert (
            sum(t.numel() for t in checkpoint["model"].values())
            == configuration(case["variant"]).total_parameters
        )
        history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
        assert [v["step"] for v in history] == list(range(1, 11))
        assert all(
            math.isfinite(v["fixed_training_batch_loss"])
            and math.isfinite(v["gradient_norm_pre_clip"])
            for v in history
        )
        assert all(
            math.isfinite(g) and g > 0
            for v in history
            for g in v["router_gradient_norms_post_clip"].values()
        )
        assert (
            history[0]["fixed_training_batch_loss"] == r["initial_fixed_batch_loss"]
            and history[-1]["fixed_training_batch_loss"] == r["last_preupdate_fixed_batch_loss"]
        )
        for stage in ("initial", "final"):
            diagnostics = json.loads((path / f"{stage}_diagnostics.json").read_text())
            assert all(v["finite"] and v.get("routing_finite", True) for v in diagnostics.values())
        rows[case["variant"]] = r
        del checkpoint
    assert len({r["batch_sha256"] for r in rows.values()}) == 1
    lines = [
        "# Multi-head comparator: integration and GPU qualification",
        "",
        "**Both initialization controls pass the frozen pipeline checks.** The architecture is now registered and ready for a separately frozen training screen. No language-model validation result, convergence or reproduced flash-kernel performance is claimed.",
        "",
        "[Frozen H045 plan](multihead_integration_plan.md), [raw decision](../results/multihead_integration_v1/result.json), [source-recorded launcher](../results/verification/multihead_integration_launcher_v1.json), and [model equations](../src/multihead_ffn/model.md).",
        "",
        "## Initial output scale",
        "",
        "The training-independent Gaussian diagnostic uses 4096 rows at width384, seed73, first FFN at layer0/seed17. Orthogonal mixers and explicit private-weight scales give the calibrated variant a full-SwiGLU RMS ratio of1.119251, within the frozen [0.5,2.0] range. The reference initialization has a much smaller initial output scale at these tiny head dimensions.",
        "",
        "| Architecture | First-FFN output RMS |",
        "|---|---:|",
    ]
    for name, r in initial["moments"].items():
        lines.append(f"| {name} | {r['rms']:.10f} |")
    lines += [
        "",
        "The variance argument assumes isotropic inputs and approximately independent centered subnetwork outputs; it retains the epsilon mass factor. Orthogonal mixing supplies an exact norm identity, not an isometric nonlinear FFN or a whole-network gradient guarantee. The observed ratio is approximate calibration, not trained superiority.",
        "",
        "## Two fresh GPU workers",
        "",
        "Each model uses width384/layers8, batch16/context128 BF16 and ten constant-rate AdamW updates at0.0006 on the SAME frozen training batch. There are2048 batch tokens and20,480 repeated token exposures per worker. No validation or test targets were scored.",
        "",
        "| Variant | Initial FP32/BF16 logit relative L2 | Peak allocated MiB | Fixed-batch loss, step1 | Fixed-batch loss, step10 | Clip fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['precision_error']['relative_l2']:.8f} | {r['peak_training_allocated_vram_bytes'] / 2**20:.2f} | {r['initial_fixed_batch_loss']:.6f} | {r['last_preupdate_fixed_batch_loss']:.6f} | {100 * r['clipped_step_fraction']:.0f}% |"
        )
    lines += [
        "",
        "Losses are measured before each listed update, on the repeatedly trained batch. Their decrease is a pipeline check, not a generalization comparison or evidence that one initialization trains better. All recorded losses, gradients and routing diagnostics are finite; every layer's router has nonzero finite gradients at every update. All ten steps clip in both cases. No global nonvanishing-gradient claim follows.",
        "",
        "At this configuration both variants have2,807,808 FFN weights and9,105,792 total, a70.2474% FFN reduction. Native execution materializes intermediate tensors. The832.10MiB peak is a pipeline measurement without a paired full-model memory control; it does not reproduce the paper's SRAM kernel or qualify a memory/speed advantage.",
        "",
        "## Verification and next step",
        "",
        "All28 earlier registered variants retain exact tiny CPU parameters, logits, loss, gradients and matrix-work counts. Twelve focused comparator tests pass. The full suite passes213 tests in the [recorded unchanged continuation](../results/verification/multihead_tests_v2.json); its first attempt suffered a native integer divide-by-zero during Matplotlib import in collection. Both records are retained; root cause is unresolved. The two GPU workers finish successfully without retries.",
        "",
        "Both source archives and checkpoint/config counts, fixed-batch hashes, ten-step histories and all finite/routing checks are verified. The registered reference and calibrated variants share the same forward implementation. Only initialization differs. The calibrated initialization is our local small-dimension control, not a claimed reproduction of the paper's exact training recipe.",
        "",
        "The pass earns a separately frozen equal-budget learning-rate screen against the full, calibrated narrow and stronger plain BlockShuffle controls. It does not earn automatic800-step promotion. Keep the corrected learnable-activation result, all failed branches and the broad convergence/novelty/stability requirements intact.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.multihead_integration_report",
        "```",
    ]
    text = "\n".join(lines) + "\n"
    for a, b in {
        "width384": "width 384",
        "layers8": "layers 8",
        "seed73": "seed 73",
        "layer0/seed17": "layer 0 / seed 17",
        "ratio of1.119251": "ratio of 1.119251",
        "batch16/context128": "batch 16 / context 128",
        "at0.0006": "at 0.0006",
        "are2048": "are 2048",
        "and20,480": "and 20,480",
        "have2,807,808": "have 2,807,808",
        "and9,105,792": "and 9,105,792",
        "a70.2474%": "a 70.2474%",
        "The832.10MiB": "The 832.10 MiB",
        "All28": "All 28",
        "passes213": "passes 213",
        "automatic800-step": "automatic 800-step",
    }.items():
        text = text.replace(a, b)
    Path("research/multihead_integration_results.md").write_text(text, encoding="utf-8")
    summary = {
        "status": "PASS",
        "old_variants_exact": 28,
        "calibrated_full_rms_ratio": initial["calibrated_full_rms_ratio"],
        "workers": {
            name: {
                "relative_l2": r["precision_error"]["relative_l2"],
                "peak_mib": r["peak_training_allocated_vram_bytes"] / 2**20,
                "result_sha256": sha256(ROOT / name / "result.json"),
            }
            for name, r in rows.items()
        },
        "all_artifacts_verified": True,
        "validation_targets": 0,
        "earns_training_screen": True,
    }
    write_json(Path("results/multihead_integration_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
