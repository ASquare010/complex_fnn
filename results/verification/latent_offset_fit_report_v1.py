"""Write H085's verified controlled-training result and replace active-study text."""

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path("results/latent_offset_fit_v1")


def read(path):
    return json.loads(Path(path).read_bytes())


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join("---" for _ in headers)+"|",
                      *("| "+" | ".join(str(v) for v in row)+" |" for row in rows)])


def write(path, text, exclusive=False):
    payload = text.encode("utf-8")
    with path.open("xb" if exclusive else "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_bytes() == payload


def section(text, start, end, new):
    assert text.count(start) == text.count(end) == 1
    a, b = text.index(start), text.index(end)
    assert a < b
    return text[:a]+new.rstrip()+"\n\n"+text[b:]


p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
a = read("results/verification/latent_offset_fit_analysis_v1.json")
assert a["status"] == "PASS" and a["selection_rescores_exact"] == a["all_rate_reporting_rescores_exact"] == 240
assert a["selected_reporting_rescores_exact"] == 120 and a["reset_ablations_exact"] == 48
assert a["gates"] == r["gates"] and not r["research_goal_achieved"]
labels = dict(zip(p["forms"], ("Plain BlockShuffle", "Plain, first LR x5/8", "Learned gain", "Learned offset",
                               "Offset, first LR x5/8", "Full affine", "Narrow SwiGLU", "Narrow GELU", "Full SwiGLU", "Full GELU")))
candidates = p["candidates"]
passed = [labels[f] for f in candidates if r["gates"][f]["earns_full_model_resource_qualification"]]
verdict = ("The following recipes earn a separately frozen full-model resource study: "+", ".join(passed)+"."
           if passed else "Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed."
           if r["assay_passed"] else "The positive-control assay fails; no candidate can be promoted and this allocation is inconclusive.")
changes = {f: 100*(r["ratios"][f]["plain"]-1) for f in p["forms"]}
summary = f"""H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by {changes['latent_offset']:+.3f}%
versus plain; offset with the fixed first-factor LR correction changes it by
{changes['latent_offset_first_lr']:+.3f}%. {verdict}
The broad research goal remains unmet."""
aggregate = table(["Form", "Learned weights", "MSE change vs plain", "Ratio vs gain", "Ratio vs narrow GELU"],
                  [(labels[f], f"{p['counts'][f]:,}", f"{changes[f]:+.3f}%", f"{r['ratios'][f]['latent_gain']:.6f}",
                    f"{r['ratios'][f]['narrow_gelu']:.6f}") for f in p["forms"]])
task_table = table(["Form", *[t.title() for t in p["tasks"]]],
                   [(labels[f], *[f"{r['task_mean_mse'][t][f]:.6f}" for t in p["tasks"]]) for f in p["forms"]])
gates = table(["Frozen gate", *[labels[f] for f in candidates]],
              [(name.replace("_", " "), *["PASS" if r["gates"][f]["tests"][name] else "FAIL" for f in candidates])
               for name in r["gates"][candidates[0]]["tests"]])
fixed_rows = []
for form, reference in (("latent_gain", "plain"), ("latent_offset", "plain"),
                         ("latent_offset_first_lr", "plain_first_lr"), ("latent_affine", "plain")):
    for rate in p["rates"]:
        fixed_rows.append((labels[form], labels[reference], rate,
                           f"{100*(r['fixed_rate_ratios'][str(rate)][form][reference]-1):+.3f}%",
                           *[f"{r['fixed_rate_by_seed_ratios'][str(rate)][form][str(seed)]:.6f}" for seed in p["seeds"]]))
fixed_table = table(["Form", "Reference", "Base LR", "Aggregate MSE change", "Seed17 ratio", "Seed29 ratio", "Seed43 ratio"], fixed_rows)
resources = table(["Form", "Mean median update ms", "Max peak MiB", "Mean clipped %", "Upper LR / 12"],
                  [(labels[f], f"{a['resources'][f]['mean_cell_median_update_ms']:.3f}",
                    f"{a['resources'][f]['max_peak_allocated_mib']:.2f}",
                    f"{100*a['resources'][f]['mean_clip_fraction']:.3f}", a["resources"][f]["upper_rate_selected"])
                   for f in p["forms"]])
resets = table(["Form", "Reset / original MSE", *[t.title() for t in p["tasks"]]],
               [(labels[f], f"{value['reset_to_original_ratio']:.6f}",
                 *[f"{value['per_task'][t]:.6f}" for t in p["tasks"]]) for f, value in a["reset_dependencies"].items()])
selections = table(["Task", "Form", "Seed17 LR", "Seed29 LR", "Seed43 LR"],
                   [(t, labels[f], *[next(v["rate"] for v in r["selected_rows"] if (v["task"], v["form"], v["seed"]) == (t, f, seed))
                                     for seed in p["seeds"]]) for t in p["tasks"] for f in p["forms"]])
process = read(ROOT / "coordinator_status.json")
minutes = (datetime.fromisoformat(process["finished_utc"])-datetime.fromisoformat(process["started_utc"])).total_seconds()/60
report = f"""# H085 - Controlled training of internal offset and gain

{summary}

These are synthetic, finite-budget measurements. The experiment has three
optimization seeds on one fresh dataset, not three independently drawn datasets
or a language-model result. Its selected-recipe gates and matched-rate gates are
reported separately below. No curve recipe from H083 is reopened.

## What was isolated

H084 showed affine gain can be folded into existing first factors and an offset
can remove the isolated SwiGLU's zero derivative at zero. Checkpoint interventions
did not explain training gains. H085 therefore trains gain-only, offset-only and
full-affine controls from matching initial functions, alongside a fixed optimizer
control and both conventional narrow baselines.

The partial forms preserve the original affine arithmetic. The unused group
control becomes a fixed zero buffer, while the other remains learned. They have
24 learned controls and 24 fixed zero buffer elements per FFN; full affine has
48 learned controls. No optimized offset kernel or changed backward formula is
introduced. Plain and narrow forms have 24 fewer learned weights than the partial
forms, a 0.00685% gap. Gain and offset have exactly equal learned counts.

The first-LR recipes multiply only first-factor learning rates by 5/8. They leave
initial tensors, the represented function, second-factor rates, shape rates,
epsilon, clipping and decay unchanged. The value5/8 was chosen before this new
dataset as a simple point in H083's observed slope range. It is history-informed,
not a tuned optimum or an exact reconstruction of learned-gain Adam dynamics.
There are two offset candidate recipes; all other forms are controls.

## Selected-recipe quality

Lower error is better. Aggregate ratios are geometric means over 12 paired
task/seed endpoints. Individual task errors are arithmetic means over three
optimization seeds. Rates are selected on a separate split before final
reporting; all rates and every checkpoint remain available.

{aggregate}

{task_table}

{gates}

The narrow controls must both be beaten. The selected offset recipe must also
beat gain-only and the fixed-LR plain control, remain within 1% of full affine
and pass the separately required matched-rate robustness condition. These are
MSE-specific gates; the language NLL allowance is not applied to synthetic MSE.
{verdict}

## Effect at each learning rate

Each row compares the same base learning rate, dataset, sampling stream and
training budget. The fixed-LR offset form is paired with the matching fixed-LR
plain form. For either offset candidate, both rates must show at least1% lower
aggregate error and improvements in all three seeds to support a robust offset
benefit and earn resource qualification.

{fixed_table}

The [result JSON](../results/latent_offset_fit_v1/result.json) also contains the
full fixed-rate comparison matrices and all240 reporting rows. Independent
verification rescored both rates for every form, so the robustness gate is
checked against the actual saved checkpoints, including unselected rates.
This is stronger evidence than a post-training reset, but still limited to the
fixed dataset, architecture, optimizer family and training budget.

## Dependence on learned controls after training

All trainable shape controls are reset to zero in each selected learned-control
checkpoint; factors and fixed zero buffers stay unchanged. Values above one
mean reset worsens error. These 48 interventions are independently reproduced.

{resets}

Resets measure dependence after coadaptation. They do not identify the causal
effect of including those controls throughout training; the matched-rate
trained comparisons above are the relevant controlled evidence.

## GPU resources

{resources}

Update times average each selected run's median after 50 warmup updates.
Allocated peaks are maxima across selected runs and include the standalone GPU
dataset cache. Timing uses synchronization around every update, FP32 native
CUDA, TF32 off, four CPU threads and one worker on the RTX4070 Laptop GPU.
These are sequential laptop measurements, not full-model memory, serving
throughput or a compiled-kernel benchmark. Partial controls intentionally retain
the original affine arithmetic for this comparison.

## Data, budget and verification

Fresh CPU input seed9813 replaces H083's9812. The fixed input permutation9283,
output rotation9282 and four analytic target formulas stay the same. The dataset
contains73,728 width384 examples:65,536 training,4,096 rate-selection and4,096
reporting. Output scaling uses training population standard deviation without
centering. Absolute MSE is not directly interchangeable with H083's different
sample draw; fresh full and narrow controls are used throughout.

Ten forms x four tasks x three seeds x two rates give240 fresh cells. Every run
uses600 updates and batch256:144,000 optimizer updates and36,864,000 training
example presentations. AdamW betas(0.9,0.95), epsilon1e-8, zero decay, clip1,
constant base rates and established fan-in/narrow calibration are retained.
There are no language targets, Transformer resource workers or training retries.
The coordinator took {minutes:.2f} minutes including qualification, fitting,
evaluation, startup and durable writing.

All eight preflight checks pass with zero updates, including80 exact retained/
checkpoint gradient tensor pairs, all30 form/seed initialization and optimizer
checks, finite differences, fresh-data contracts and rejection behavior.
The independent audit reconstructs all30 initial states,240 RNG/optimizer
histories, fresh targets/training scales/streams, fixed zero buffers and actual
counts. It exactly reproduces240 selection scores, all240 reporting scores and
48 resets. That is983,040 selection and1,179,648 reporting/reset example
presentations, zero optimizer updates. The extra120 unselected reporting passes
verify the mandatory two-rate gate. Audit duration:
{read(ROOT / 'analysis_process.json')['elapsed_seconds']:.2f} seconds.

A status-inspection PowerShell process exited abnormally while listing CIM
processes after printing the passing preflight result. A native Get-Process
recheck confirmed the original coordinator and fitting handles were live and
logs advanced. Training was not restarted. The cause is undiagnosed and the
[observer record](../results/latent_offset_fit_v1/observation_shell_failure.json)
is separate from the experiment's terminal records.

All135 scientific sources,74 plans and prior H083/H084 anchors remain intact.
The active repository still has three model folders, six variants and nine
recipes. No universal gradient, convergence, scale or broad-corpus claim follows.
Any earned resource qualification requires its own frozen full-model study;
failure does not authorize another rate, step count or architecture variation.

## Every selected rate

{selections}

[Frozen plan](latent_offset_fit_plan.md),
[independent audit](../results/verification/latent_offset_fit_analysis_v1.json),
[all measurements](../results/latent_offset_fit_v1/result.json),
[preceding mechanism study](latent_affine_mechanism_results.md).
"""
path = Path("research/latent_offset_fit_results.md")
assert not path.exists()
docs = {n: Path(n).read_text(encoding="utf-8") for n in ("README.md", "research/CURRENT_STATE.md",
        "research/PROGRESS_OVERVIEW.md", "research/idea_bank.md", "research/learnable_activation_domain.md")}
root_summary = summary+"\n[Completed controlled offset study](research/latent_offset_fit_results.md)."
docs["README.md"] = section(docs["README.md"], "The [controlled offset study]", "The [affine mechanism study]", root_summary)
note = "## Latest controlled offset fitting\n\n"+summary+"""

All240 selection and reporting scores and48 resets reproduce exactly. Both
selected-recipe and matched-rate conditions are evaluated against fresh full,
narrow, gain and optimizer controls. [Completed study](latent_offset_fit_results.md).
"""
docs["research/CURRENT_STATE.md"] = section(docs["research/CURRENT_STATE.md"], "## Active controlled offset fitting", "## Latest mechanism result", note)
docs["research/PROGRESS_OVERVIEW.md"] = section(docs["research/PROGRESS_OVERVIEW.md"], "## Active controlled offset fitting", "## Latest follow-up: what the affine control does", note)
marker = "## H085 - Controlled internal offset training (EXPERIMENTING)"
assert docs["research/idea_bank.md"].count(marker) == 1
docs["research/idea_bank.md"] = docs["research/idea_bank.md"].split(marker)[0]+"## H085 - Controlled internal offset training (COMPLETE)\n\n"+summary+"""

Eight qualification checks pass; all240 selection and reporting scores and48
resets reproduce exactly. The two-rate robustness condition is independently
verified alongside selected-recipe gates. The fixed first-factor LR control is
history-informed and all forms train on fresh inputs. No old curved recipe is
reopened. [Full result](latent_offset_fit_results.md), [plan](latent_offset_fit_plan.md).
"""
marker = "## Controlled training now under way"
assert docs["research/learnable_activation_domain.md"].count(marker) == 1
docs["research/learnable_activation_domain.md"] = docs["research/learnable_activation_domain.md"].split(marker)[0]+"## Controlled internal-offset training result\n\n"+summary+"""

Partial controls retain the original affine arithmetic with the unused controls
fixed as zero buffers. The experiment includes both gain-only and fixed-LR
controls, and tests the offset effect at each learning rate. All final reporting
scores, including unselected rates, reproduce exactly. These trained comparisons
address a question that resets alone cannot answer, within the stated synthetic
scope. [Detailed controlled comparison](latent_offset_fit_results.md).
"""
write(path, report, exclusive=True)
for name, text in docs.items():
    write(Path(name), text)
print(json.dumps({"status": "PASS", "report": path.as_posix(), "updated_docs": list(docs),
                  "scientific_verdict": r["scientific_verdict"], "qualified_recipes": passed,
                  "optimizer_updates": 0}), flush=True)
