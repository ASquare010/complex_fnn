"""Report H056 using completed records only; no training or scoring."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path("results/overcomplete_screen_v1")
r = read(root / "result.json")
pre = read(root / "preflight.json")
assert r["status"] == "complete" and not r["earns_separate_longer_comparison"]
labels = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "calibrated_narrow": "Calibrated narrow",
    "blockshuffle": "Plain BlockShuffle",
    "headwise": "Square headwise",
    "overcomplete_headwise": "Overcomplete headwise",
}
rows = pre["controls"] + r["trials"]
metrics = {row["run"]: read(Path("results/runs") / row["run"] / "metrics.json") for row in rows}
selected = {**r["references"], "overcomplete_headwise": r["selected"]}
inputs, observations = {}, {}
for recipe, run in selected.items():
    p = Path("results/runs") / run
    for n in (
        "metrics.json",
        "history.jsonl",
        "initial_diagnostics.json",
        "final_diagnostics.json",
    ):
        inputs[(p / n).as_posix()] = sha(p / n)
    initial, final = read(p / "initial_diagnostics.json"), read(p / "final_diagnostics.json")
    history = [json.loads(line) for line in (p / "history.jsonl").read_text().splitlines()]
    observations[recipe] = {
        "run": run,
        "layers": {
            str(i): {
                "initial_ffn_output_rms": initial[f"layer_{i}.ffn"]["output_rms"],
                "final_ffn_output_rms": final[f"layer_{i}.ffn"]["output_rms"],
                "final_residual_output_rms": final[f"layer_{i}.residual"]["output_rms"],
                "last_logged_ffn_gradient_norm": history[-1]["layer_gradient_norms_post_clip"][
                    f"layer_{i}.ffn"
                ],
            }
            for i in range(8)
        },
        "last_logged_training_loss": history[-1]["training_loss"],
        "last_logged_preclip_norm": history[-1]["gradient_norm_pre_clip"],
    }
record = {
    "status": "complete",
    "scope": "Descriptive extraction from existing fixed validation-input diagnostics and last logged training batch; no new training/scoring and no causal attribution",
    "inputs": inputs,
    "observations": observations,
}
p = root / "observations.json"
assert not p.exists()
p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
lines = [
    "# H056: Overcomplete headwise fails the first language-quality screen",
    "",
    "**Decision: REJECTED for promotion at this budget.** The best of three fixed rates",
    "reaches NLL **6.043891**, 2.305% above full SwiGLU, 2.800% above full GELU and",
    "2.223% above calibrated narrow. All three quality gates fail. Parameter and",
    "memory gates pass; every new trial remains finite. This recipe earns no longer",
    "run, added activation or automatic optimizer search.",
    "",
    "The constructive representation proof remains valid. This experiment shows that",
    "representing the excluded quadratic and matching initialization RMS did not",
    "produce competitive language learning under the tested recipe. It does not",
    "establish a general limit of rectangular mixing or a causal explanation.",
    "",
    "## Equal three-rate development screen",
    "",
    "All rows use d=384, eight layers, six attention heads, vocabulary 4096, context",
    "128, batch 16, seed 17 and 200 optimizer steps. Each trial samples 409,600 training",
    "targets and evaluates all 322,688 validation targets. Lowest final NLL selects",
    "each recipe; exact ties select the lower rate. All three candidate cells complete.",
    "The twelve original controls and three square-headwise controls are retained,",
    "with verified hashes, source compatibility, actual groups and sampler states.",
    "",
    "| Recipe | NLL at LR .0003 | NLL at LR .0006 | NLL at LR .0012 |",
    "|---|---:|---:|---:|",
]
for recipe, label in labels.items():
    cells = sorted([row for row in rows if row["recipe"] == recipe], key=lambda row: row["rate"])
    lines.append(
        "| "
        + label
        + " | "
        + " | ".join(f"{metrics[row['run']]['validation_loss']:.6f}" for row in cells)
        + " |"
    )
lines += [
    "",
    "The candidate selects the upper boundary, LR .0012; an optimal rate is unproven.",
    "It loses to full SwiGLU, full GELU, narrow and plain BlockShuffle at every matched",
    "rate. It beats square headwise at .0003/.0006 but loses by 0.375% when each selects",
    "its best rate. Against selected BlockShuffle (.0006), its cost is 1.227%.",
    "The separate 0.2% narrow-margin diagnostic also fails. These are fixed engineering",
    "gates on a development set used for selection, not significance or test-set claims.",
    "",
    "![All rates and selected trajectories](../results/plots/overcomplete_screen.png)",
    "",
    "## Parameter and systems measurements",
    "",
    "| Selected recipe | FFN weights | Total weights | Training peak MiB | Clipped steps |",
    "|---|---:|---:|---:|---:|",
]
for recipe, run in selected.items():
    m = metrics[run]
    lines.append(
        f"| {labels[recipe]} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.3f} | {100 * m['clipped_step_fraction']:.1f}% |"
    )
c = metrics[r["selected"]]
lines += [
    "",
    f"The candidate retains **{c['ffn_reduction_percent']:.4f}% fewer FFN weights** and",
    f"**{c['total_reduction_percent']:.3f}% fewer total-model weights** than either full control.",
    f"All three rates use **{c['peak_allocated_vram_bytes'] / 2**20:.3f} MiB** allocated training peak.",
    "This passes both <=10% memory allowances and is distinct from H055's fixed-batch",
    "qualification. The screen measures the real training loop, including Adam state,",
    "gradients, validation and allocated workspaces before final diagnostics/timing.",
    "",
    "New candidate clipping fractions are 42% / 20% / 7% across ascending rates.",
    "All final weights, Adam moments, logged gradients and initial/final layer outputs",
    "are finite. Clipping alone does not diagnose stable optimization or convergence.",
    "",
    f"The selected run records {c['training_tokens_per_second']:,.0f} timed training tokens/s",
    f"and {c['inference_tokens_per_second']:,.0f} full-sequence inference tokens/s on the RTX",
    "4070 Laptop. Training excludes the first ten steps, evaluation and checkpoint I/O;",
    "inference uses the shared synchronized warmup/repeat procedure. These are descriptive",
    "cross-session measurements; no paired speedup or generation-speed claim follows.",
    f"Logical FFN matrix forward FLOPs are {c['ffn_matrix_forward_flops_per_token']:,} per token",
    "across eight layers. The raw metrics retain model/training estimates and their",
    "exclusions; factorization, nonlinear operations and wall-clock cost differ.",
    "",
    "## What changed, and what the diagnostics show",
    "",
    "The [frozen protocol](overcomplete_screen_plan.md) uses the H053/H054 module and",
    "H055 native recomputed gate, with the existing factor fan-in LR policy. Input",
    "mixer factor scales are 8/8, output scales 8/(32/3); private head tensors stay",
    "at scale 1. Parameter decay is .1. This differs from H055's uniform-rate memory",
    "qualification. All dense controls produce identical actual groups when their",
    "factor-policy flag changes to fan_in, preserving narrow width calibration.",
    "The architecture comparison changes mixer structure, head geometry and",
    "initialization together; it is not an isolated expansion ablation.",
    "",
    "At the selected rate, initial layer-0 FFN RMS is 0.012844, close to full SwiGLU",
    "0.012627. By step 200 it reaches **13.7203**, versus **3.8765** for full SwiGLU",
    "and **3.8589** for square headwise. The last logged candidate FFN gradient norms",
    "in layers 1-7 are 0.00436-0.01028; full SwiGLU records 0.01744-0.06536.",
    "All are finite and nonzero. These norms depend on parameterization and probe",
    "inputs, so they do not prove vanishing gradients, capacity failure or causation.",
    "",
    "| Layer | Candidate initial FFN RMS | Candidate final FFN RMS | Full SwiGLU final FFN RMS | Candidate last FFN gradient norm |",
    "|---|---:|---:|---:|---:|",
]
for i in range(8):
    a = observations["overcomplete_headwise"]["layers"][str(i)]
    b = observations["full_swiglu"]["layers"][str(i)]
    lines.append(
        f"| {i} | {a['initial_ffn_output_rms']:.6f} | {a['final_ffn_output_rms']:.6f} | {b['final_ffn_output_rms']:.6f} | {a['last_logged_ffn_gradient_norm']:.6f} |"
    )
lines += [
    "",
    "The candidate's last training-batch loss is 5.989406 versus full SwiGLU 5.859748",
    "and narrow 5.851320 on the matched stream. One logged batch is not mean training",
    "NLL. The joint training/validation deficit and scale growth give a concrete",
    "optimization question; they do not establish its answer. A separate checkpoint",
    "analysis of residual scale and factor geometry can precede any new hypothesis.",
    "No longer run or activation repair is earned by this failed screen.",
    "",
    "## Reproducibility and retained failure",
    "",
    "All **245 tests pass** in the [recorded suite](../results/verification/overcomplete_screen_tests_v2.json)",
    "(`245 passed in 37.29s`, `-p no:anyio`, all assertions/GPU checks enabled). Six",
    "focused configuration/decision/worker checks pass. Existing computation is",
    "unchanged from H055, preserving its 32-model exact-signature evidence. Three",
    "new trials use **1,228,800 sampled training tokens**, with no numerical failures.",
    "No official test data is fetched or scored.",
    "",
    "The first preflight stopped before GPU training because it required byte-identical",
    "diagnostics from before later architecture hooks existed. Its [failure log](../results/verification/overcomplete_screen_preflight_v1.log)",
    "and [exact driver](../results/verification/overcomplete_screen_before_preflight_fix_v1.py)",
    "are retained. After inspecting that diff, the corrected preflight compares old",
    "and current diagnostics on each retained CPU probe, including gradient statistics,",
    "unchanged weights and unchanged RNG. Those values match exactly. Training-step",
    "AST, evaluation functions and shared forward/loss checks also pass. The frozen",
    "plan/config/gates stay unchanged. This was one diagnosed verification failure;",
    "all three training processes and the corrected full pipeline complete first attempt.",
    "",
    "Artifacts: [decision](../results/overcomplete_screen_v1/result.json),",
    "[preflight](../results/overcomplete_screen_v1/preflight.json),",
    "[protocol](../results/overcomplete_screen_v1/protocol.json),",
    "[source archive](../results/overcomplete_screen_v1/source.zip),",
    "[layer observations](../results/overcomplete_screen_v1/observations.json),",
    "[plot inputs](../results/verification/overcomplete_screen_plot_v1.json), and",
    "[candidate config](../configs/wikitext2_overcomplete_headwise_screen.json).",
    "The result hashes every new checkpoint, history, metrics and diagnostic file.",
    "",
    "To reproduce one cell into a fresh output directory:",
    "",
    "```powershell",
    "uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_overcomplete_headwise_screen.json --learning-rate 0.0012 --cache data/wikitext2_v1 --output results/runs/overcomplete_reproduction_s17_200",
    "```",
    "",
    "The complete archived experiment uses the qualified minimal worker; its protocol",
    "and launcher prevent overwriting or silently retrying a cell. No novelty,",
    "convergence, seed robustness, scale transfer or breakthrough is established.",
    "The full research goal remains open; H051 and earlier activation failures remain.",
    "",
]
p = Path("research/overcomplete_screen_results.md")
assert not p.exists()
p.write_text("\n".join(lines), encoding="utf-8")
assert all(sha(n) == h for n, h in inputs.items())
print(
    json.dumps(
        {
            "status": "complete",
            "report": str(p),
            "observation_recipes": len(observations),
            "total_reduction_percent": c["total_reduction_percent"],
        }
    )
)
