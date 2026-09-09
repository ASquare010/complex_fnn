"""Write an H081 report from the independently verified complete screen."""

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path("results/blast_learning_screen_v1")
REPORT = Path("research/blast_learning_screen_results.md")
assert not REPORT.exists()
a = json.loads(Path("results/verification/blast_learning_screen_analysis_v1.json").read_bytes())
r = json.loads((ROOT / "result.json").read_bytes())
assert a["status"] == "PASS" and a["checkpoint_rescores_exact"] == 24
labels = {"full_swiglu": "Full SwiGLU", "full_gelu": "Full GELU", "narrow_swiglu": "Narrow SwiGLU",
          "narrow_gelu": "Narrow GELU", "plain": "BlockShuffle legacy",
          "plain_calibrated": "BlockShuffle calibrated", "blast_swiglu": "BLAST SwiGLU", "blast_gelu": "BLAST GELU"}
selected = {f: a["rows"][c] for f, c in a["selected_cells"].items()}
verdicts = {f: "PASSES SHORT SCREEN" if all(g.values()) else "FAILS FIXED SCREEN"
            for f, g in a["gates"].items()}
table = "| Model | Selected LR | Final NLL | Peak MiB | Mean update ms | Clip % |\n|---|---:|---:|---:|---:|---:|\n"
for f, row in selected.items():
    table += f"| {labels[f]} | {row['rate']:.4f} | {row['validation_loss']:.6f} | {row['peak_mib']:.2f} | {row['mean_training_step_ms']:.2f} | {100*row['clipped_fraction']:.1f} |\n"
all_rates = "| Model | LR | Final NLL | Peak MiB | Mean update ms |\n|---|---:|---:|---:|---:|\n"
for row in a["rows"].values():
    all_rates += f"| {labels[row['form']]} | {row['rate']:.4f} | {row['validation_loss']:.6f} | {row['peak_mib']:.2f} | {row['mean_training_step_ms']:.2f} |\n"
decisions = "\n".join(f"- **{labels[f]}: {verdicts[f]}**. Failed gates: " +
                       (", ".join(k.replace('_', ' ') for k, v in g.items() if not v) or "none") + "."
                       for f, g in a["gates"].items())
relative = "| Candidate | vs full SwiGLU | vs full GELU | vs narrow SwiGLU | vs narrow GELU | vs legacy BlockShuffle |\n|---|---:|---:|---:|---:|---:|\n"
for f, comp in a["selected_nll_relative_percent"].items():
    relative += f"| {labels[f]} | " + " | ".join(f"{comp[ref]:+.3f}%" for ref in ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain")) + " |\n"
same_rate = "| LR | Calibrated BlockShuffle vs legacy | BLAST SwiGLU vs calibrated BlockShuffle | BLAST GELU vs calibrated BlockShuffle |\n|---|---:|---:|---:|\n"
for rate in (300, 600, 1200):
    base = a["rows"][f"plain_lr{rate}"]["validation_loss"]
    calibrated = a["rows"][f"plain_calibrated_lr{rate}"]["validation_loss"]
    sg = a["rows"][f"blast_swiglu_lr{rate}"]["validation_loss"]
    ge = a["rows"][f"blast_gelu_lr{rate}"]["validation_loss"]
    same_rate += f"| {rate/1e6:.4f} | {100*(calibrated/base-1):+.3f}% | {100*(sg/calibrated-1):+.3f}% | {100*(ge/calibrated-1):+.3f}% |\n"
status = json.loads((ROOT / "coordinator_status.json").read_bytes())
duration = (datetime.fromisoformat(status["finished_utc"]) - datetime.fromisoformat(status["started_utc"])).total_seconds()/60
boundary = ", ".join(labels[f] for f, v in selected.items() if v["rate"] == 0.0012) or "none"
text = f"""# H081 - BLAST learning screen and factor calibration

**Complete short screen; the broader research goal remains unmet.** Twenty-four
fresh WikiText-2 trials finish, and independent BF16 rescoring exactly reproduces
all 24 final checkpoint losses. One seed and 200 updates do not establish
significance, convergence, broader-data performance or state-of-the-art.

{decisions}

## Selected final results

Each form receives the same three peak learning rates and training budget. Select
the lowest final validation NLL; lower is better. These short-budget scores are
not directly comparable to H078's 3,200-update scores.

{table}
All compressed forms have 2,801,664 FFN / 9,099,648 total weights, versus
9,437,184 / 15,735,168 for both full controls: 70.3125% FFN and 42.1700% total
reduction. BLAST hidden widths are 1,984 (SwiGLU) and 3,200 (GELU), rank 48 and eight groups.
Lower weight counts are not a measured latency claim. Times are sequential
laptop-GPU measurements with no repeat-based confidence interval.

## Quality comparisons and frozen gates

Relative NLL changes; negative favors the candidate. Both full controls permit
at most +1%; both narrows must be beaten strictly. Actual peak must be within 10%
of both selected full peaks. The selected NLL run also supplies its resource gate.

{relative}
Upper-rate boundary selections: {boundary}. No rate extension or repair is made
after these outcomes. A passing screen earns a separate longer, then multi-seed
comparison; it does not qualify a new active model. Failed fixed recipes close.

## Optimizer ablation and interpretation

{same_rate}
The calibrated BlockShuffle starts byte-identical to legacy, with equal initial
validation NLL and first training loss at every rate. Their difference is a
complete optimizer recipe: initial-RMS multipliers plus product decay replace
fan-in multipliers plus parameter decay. These data do not isolate the two parts.

BLAST is [published prior art](https://arxiv.org/html/2410.21262v1). Its qualified
operator and layer-local orthogonal initialization are reused without changing
the contraction. The shared dense constructor runs before replacing temporary
FFNs, preventing its generic initializer from overwriting BLAST factors.
Common non-FFN tensors match all controls exactly. BLAST comparisons also change
factor structure and hidden width; they are not pure activation ablations.

For K factors, initial RMS rho_j and dense reference scale sigma, fixed
s_j=rho_j/(K*sigma) makes a hypothetical unit-RMS normalized Adam direction have
relative factor update eta/(K*sigma). Actual directions, clipping and Jacobians
need not match. With decay lambda/(K*s_j), zero-gradient/zero-moment shrinkage
of the represented map is (1-eta*lambda/K)^K: it matches dense only to first order.
The [frozen derivation and limits](blast_learning_screen_plan.md) are local
calculations, not nonvanishing-gradient or convergence guarantees.

## Data, execution and verification

WikiText-2 raw-v1, the unchanged train-only 4,096-token BPE cache: 3,083,650 train tokens,
322,802 validation tokens. Batch 16 x 128 gives 2,048 targets/update; 200 updates give
409,600 targets/run. Totals: 4,800 language updates / 9,830,400 training targets.
Shared seven-pass validation presents 54,211,584 development targets; independent
rescoring adds 7,744,512 targets, zero updates. The official test split is unscored.

All nine preflight checks pass in 25.03 seconds, including full-model CPU FP32 and
CUDA BF16 eager/checkpoint fidelity: 24 temporary updates / 24,960 target presentations,
{a['qualification_gradient_pairs_exact']} exact gradient tensor pairs across the paired steps.
They never initialize a language run. The coordinator took {duration:.2f} minutes,
including preflight, validation, startup, artifact checks and checkpoint writing.

Native eager BF16/FP32 parameters, TF32 off, four CPU threads, one GPU worker at
a time. Whole-block recomputation except the qualified narrow-GELU block+inner
mode. The unchanged shared trainer handles AdamW, sampling, schedule, loss,
evaluation and clipping. Both narrows retain their established width/decay
calibration. Per-step loss/preclip norm/rate, per-layer gradients/activations and
sampled activation slopes are saved, together with full weights, moments and RNG.

Independent checks verify every checkpoint, actual counts, optimizer membership,
moments/step counters, sampling RNG, source/config/data hashes, histories, gates
and scores. Completed artifacts are fsynced and read back; this does not diagnose
or repair the older H079 corruption. Prior source and evidence remain unchanged.
No scientific retries occurred. No new activation or active folder is added.

A [historical control comparison](../results/blast_learning_screen_v1/control_reproduction.json)
finds exact prior final weights/moments/NLL for 14 of 15 controls. Full GELU at
LR 0.0006 has identical initial NLL but a final NLL difference of -0.0000138162;
its final weights and moments also differ. The cause is not diagnosed. This
cross-run diagnostic is separate from the exact checkpoint rescores. All gates
use fresh H081 controls; no older score is substituted and no run is repeated.

## Every allocated rate

{all_rates}
[Measurements](../results/blast_learning_screen_v1/result.json),
[independent audit](../results/verification/blast_learning_screen_analysis_v1.json),
[frozen protocol](../results/blast_learning_screen_v1/protocol.json).
"""


def save(path, content, exclusive=False):
    payload = content.encode()
    with path.open("xb" if exclusive else "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_bytes() == payload


save(REPORT, text, exclusive=True)
short = "; ".join(f"{labels[f]}: {verdicts[f].lower()}" for f in verdicts)
summary = ("H081 completes all 24 fresh 200-update WikiText-2 runs and all final checkpoint\n"
           "rescores match exactly. " + short + ".\n"
           "This is one-seed short-budget evidence; longer, multi-seed, scale and broader\n"
           "data requirements remain open. [Complete screen](blast_learning_screen_results.md).\n")
path = Path("research/PROGRESS_OVERVIEW.md")
content = path.read_text(encoding="utf-8")
content = content.split("## Current work\n", 1)[0] + "## Current work\n\n" + summary + "\n" + table
save(path, content)
path = Path("research/CURRENT_STATE.md")
content = path.read_text(encoding="utf-8")
start = content.index("## In-progress learning screen")
end = content.index("## Latest operator qualification", start)
content = content[:start] + "## Latest completed short screen\n\n" + summary + "\n" + table + "\n" + content[end:]
content = content.replace("The latest completed language experiment is matched-budget", "The latest completed longer-budget language experiment is matched-budget", 1)
content = content.replace("## Latest complete language comparison", "## Latest complete longer-budget language comparison", 1)
save(path, content)
path = Path("README.md")
content = path.read_text(encoding="utf-8")
start = content.index("The [BLAST learning screen]")
end = content.index("**Latest language result:", start)
intro = ("The [BLAST learning screen](research/blast_learning_screen_results.md) is complete:\n"
         "24 fresh 200-update runs, nine passing preflight checks and 24 exact final\n"
         "checkpoint rescores. " + short + ".\n"
         "These are short-budget results; the active model tree is unchanged.\n\n")
content = content[:start] + intro + content[end:]
content = content.replace("**Latest language result:", "**Latest longer-budget language result:", 1)
save(path, content)
path = Path("research/idea_bank.md")
content = path.read_text(encoding="utf-8")
start = content.index("## H081 -")
content = content[:start] + "## H081 - BLAST language and factor calibration (SCREEN COMPLETE)\n\n" + summary + "\n" + table
save(path, content)
print("PASS: wrote the complete screen report and updated four navigation documents.", flush=True)
