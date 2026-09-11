"""Readable tables, fixed-gate decisions and a publication-style memory figure."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("results/token_memory_replication_v1")


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    return {"mean":statistics.mean(values),"median":statistics.median(values),
            "sample_variance":statistics.variance(values),"min":min(values),"max":max(values)}


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    assert read(ROOT/"audit.json")["passed"]
    cases = [read(p) for p in sorted(ROOT.glob("*/cases/*/metrics.json"))]
    assert len(cases) == 36
    comparisons = []
    for row in cases:
        if row["policy"] != "loss_chunks":
            continue
        cohort = {r["policy"]:r for r in cases if
                  (r["variant"],r["context"],r["seed"]) ==
                  (row["variant"],row["context"],row["seed"])}
        native, block = cohort["native"],cohort["block"]
        assert len({r["data_order_sha256"] for r in cohort.values()}) == 1
        assert len({r["parameter_count"] for r in cohort.values()}) == 1
        memory = row["peak_allocated_bytes"]/block["peak_allocated_bytes"]
        speed = row["timing"]["update_ms"]["median"]/block["timing"]["update_ms"]["median"]
        loss = row["validation_nll"]/native["validation_nll"]
        gates = {"memory":memory <= .85,"time":speed <= 1.25,
                 "quality":abs(loss-1) <= .01,"finite":row["weights_finite"]}
        comparisons.append({"variant":row["variant"],"context":row["context"],"seed":row["seed"],
                            "memory_ratio":memory,"time_ratio":speed,"nll_ratio":loss,
                            "native_block_states_exact":native["state_hashes"] == block["state_hashes"],
                            "native_block_nll_relative_difference":block["validation_nll"]/native["validation_nll"]-1,
                            "gates":gates,"all_pass":all(gates.values())})
    groups = []
    for variant in ("gelu_narrow","swiglu"):
        for context in (128,512):
            subset = [c for c in comparisons if c["variant"]==variant and c["context"]==context]
            assert len(subset) == 3
            group = {"variant":variant,"context":context,"seeds":len(subset),
                     "all_seeds_pass":all(c["all_pass"] for c in subset),
                     **{key:stats([c[key] for c in subset]) for key in ("memory_ratio","time_ratio","nll_ratio")}}
            groups.append(group)
    rows = []
    for r in cases:
        rows.append({"variant":r["variant"],"context":r["context"],"seed":r["seed"],
                     "policy":r["policy"],"parameters":r["parameter_count"],
                     "peak_allocated_mib":r["peak_allocated_bytes"]/2**20,
                     "peak_reserved_mib":r["peak_reserved_bytes"]/2**20,
                     "median_update_ms":r["timing"]["update_ms"]["median"],
                     "nll":r["validation_nll"],"clip_fraction":r["clip_fraction"]})
    with (ROOT/"metrics.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {"status":"COMPLETE","cases":len(cases),"comparisons":comparisons,"groups":groups,
              "native_block_exact_final_pairs":sum(c["native_block_states_exact"] for c in comparisons),
              "audit":read(ROOT/"audit.json"),
              "duration":read(ROOT/"completion.json"),"novelty_claim":False,"breakthrough":False}
    write(ROOT/"result.json",result)
    partial = [read(p) for p in sorted(Path("results/token_memory_v1/cases").glob("*/metrics.json"))]
    write("results/token_memory_v1/result.json",{
        "status":"STOPPED_FOR_FIDELITY_DIAGNOSIS","completed_cases":len(partial),
        "planned_cases":180,"qualification":read("results/token_memory_v1/qualification.json"),
        "endpoints":[{k:r[k] for k in ("key","policy","variant","seed","context","validation_nll",
                                      "peak_allocated_bytes","timing")} for r in partial],
        "diagnosis":"Fresh-process replay and checkpoint audit reproduce full GELU failure; not promoted.",
        "followup":"research/token_memory_followup_plan.md"})
    lines = ["# Token memory results: H101 and H102", "",
             "**Narrow GELU qualifies for this short memory study at both tested shapes.**",
             "Loss chunking saves 21.01% / 32.23% allocated training memory versus whole-block",
             "checkpointing across three seeds. Updates cost 7.45-22.51% more; the largest",
             "relative final NLL change is +0.189%. Retain this as an experimental option.", "",
             "This is an execution study, with no new activation, parameter reduction or novelty claim.",
             "H101 was stopped after22/180 planned cases because a full GELU trajectory failed fidelity.",
             "H102 completed36 fresh cases;38 saved checkpoints, including two diagnostic replays,",
             "were independently loaded and their unchunked validation scores reproduced exactly.", "",
             "## Fixed-gate replication", "",
             "Each row summarizes three seeds. Changes are relative to whole-block checkpointing;",
             "NLL is relative to native execution of the same model. Lower memory/time/NLL is better.", "",
             "| Model | Context | Allocated memory change | Median update-time change | NLL change | All seeds pass |",
             "|---|---:|---:|---:|---:|---|"]
    for g in groups:
        lines.append(f"| {g['variant']} | {g['context']} | {(g['memory_ratio']['mean']-1)*100:+.2f}% | "
                     f"{(g['time_ratio']['median']-1)*100:+.2f}% | {(g['nll_ratio']['mean']-1)*100:+.4f}% | "
                     f"{'Yes' if g['all_seeds_pass'] else 'No'} |")
    lines += ["", "Gates fixed before measurements: >=15% lower allocated training memory, <=25%",
              "update-time increase and <=1% absolute relative final NLL difference, for every seed.",
              "Mean, median, sample variance, ranges and all individual gates are in",
              "[result.json](../results/token_memory_replication_v1/result.json); all36 rows are in",
              "[metrics.csv](../results/token_memory_replication_v1/metrics.csv).", "",
              "Full SwiGLU fails the all-seed runtime gate: its worst observed penalties are",
              "+93.34% at context 128 and +71.57% at context 512. The median alone would hide",
              "these failures. Short separate-process GPU timings vary substantially; the raw",
              "forward/backward/optimizer timings are retained. No general speed claim follows.", "",
              "| Model | Context | Parameters (unchanged) | Block checkpoint MiB | With loss chunks MiB |",
              "|---|---:|---:|---:|---:|"]
    for g in groups:
        group_rows = [r for r in rows if r["variant"]==g["variant"] and r["context"]==g["context"]]
        peaks = {p:statistics.mean(r["peak_allocated_mib"] for r in group_rows if r["policy"]==p)
                 for p in ("block","loss_chunks")}
        lines.append(f"| {g['variant']} | {g['context']} | {group_rows[0]['parameters']:,} | "
                     f"{peaks['block']:.2f} | {peaks['loss_chunks']:.2f} |")
    lines += ["",
              "![Allocated memory and measured runtime](figures/token_memory.png)", "",
              "## The failure is part of the result", "",
              "At B16/T128, seed17, full GELU native NLL is6.935556. FFN-only chunking gives9.108537;",
              "loss-only gives8.224111. The latter failure reproduces in a fresh process and an",
              "independent checkpoint rescore. Combined chunking gives6.939283 in that case, but",
              "saves only1.25% memory and costs56.26% more time than block checkpointing.", "",
              "Tiny full-model qualification passed40 CPU-double/CUDA-BF16 cases. Full-scale",
              "matched-state gradients differ by up to0.627% initially and1.977% at the native",
              "GELU endpoint for loss chunking. Floating-point reduction order changes even when",
              "the real-arithmetic equations agree. These observations show trajectory sensitivity;",
              "they do not establish its exact causal mechanism or a general precision remedy.", "",
              "FFN-only chunking adds cost without a material memory benefit in the completed",
              "small-shape cases. That fixed recipe is closed. The full GELU loss recipe also",
              "remains closed; H102 does not retroactively repair or hide it.", "",
              "## What is proved, and what is measured", "",
              "For deterministic token-independent f, partitioning rows commutes with applying f.",
              "For a linear classifier and CE, summing chunk losses and dividing by the total",
              "number of valid targets gives the same mathematical loss. Differentiating that",
              "finite sum gives the same mathematical gradients. Uneven final chunks must not",
              "receive equal weight as full chunks. The14 helper tests cover those identities,",
              "numerical gradcheck, masks, noncontiguous inputs and invalid chunk sizes.", "",
              "Checkpointing retains inputs and recomputes intermediates. Expanded FFN and loss",
              "workspaces change from O(Nh)/O(NV) to O(Ch)/O(CV), with C512 here. O(Nd) saved",
              "features, parameters, gradients, optimizer and attention costs remain. This bound",
              "is not a prediction that total peak VRAM falls by N/C, nor a proof of faster runtime.", "",
              "## Reproducible scope", "",
              "One RTX4070 Laptop GPU8GiB; UV, PyTorch2.14.0+cu132, BF16 autocast, FP32 weights",
              "and AdamW, TF32 off, four CPU threads. Width384, eight layers, six heads, V4096.",
              "Narrow GELU h456 and full SwiGLU h1024. Shapes B16/T128 and B8/T512, chunks512.",
              "Seeds17/29/43;12 updates from scratch, constant LR0.0006, matrix decay0.1,",
              "betas(.9,.95), eps1e-8, clip1. Four warmup/eight synchronized timed updates.",
              "Data is the existing hashed WikiText-2 cache. All methods share data order;",
              "eight unchunked validation batches score16,384 or32,768 targets. Test data is unused.", "",
              "Allocated training peaks include CUDA corpus cache and optimizer initialization,",
              "but exclude validation/export. Reserved peaks are separately reported. Timing",
              "includes forward, backward, clipping and optimizer, excluding sampling/scoring/I/O.",
              f"Native/block checkpointing end with bitwise-identical weights in {result['native_block_exact_final_pairs']}/12 pairs.",
              "Their largest relative NLL difference is 0.00471%; checkpointing itself is not",
              "bitwise identical in every case at the longer context. No deterministic-kernel",
              "guarantee was enabled, so this study does not attribute that difference to one cause.",
              "Loss chunking does not promise identical weights. No convergence, long-duration",
              "language quality, inference-speed or broader-corpus result follows from12 updates.", "",
              "Sources and recipes are frozen in the protocols/source archive. Checkpoints remain",
              "local and ignored by Git; a clean clone must rerun to independently rescore them.", "",
              "```powershell", "uv run --no-sync python -m pytest -p no:anyio tests/test_token_memory.py -q",
              "# Reproduce one qualified case in a fresh output directory:",
              "uv run --no-sync python -m results.token_memory_v1.source.diagnose loss_chunks --variant gelu_narrow --context 512 --seed 17 --root results/token_memory_reproduction_example",
              "uv run --no-sync python -m results.token_memory_v1.source.audit",
              "uv run --no-sync python -m results.token_memory_v1.source.analyze", "```", "",
              "The original study and replication refuse to overwrite run directories. For a new",
              "experiment, use a fresh worktree/output root and preserve its own protocol. The",
              "H101 frozen-source hashes intentionally reject changes; do not edit them to force a run.", "",
              "## Prior work and next research constraint", "",
              "[Reformer](https://arxiv.org/abs/2001.04451) already uses FFN chunking;",
              "[Cut Your Losses](https://arxiv.org/abs/2411.09009) targets classifier-logit memory",
              "with specialized kernels. This native token-chunk implementation is not CCE and",
              "drops no gradient terms. [PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)",
              "supplies recomputation. No novelty or SOTA advantage is established.", "",
              "The result constrains future architecture research: compare actual memory against",
              "appropriately optimized controls, and qualify training trajectories as well as local",
              "derivatives. Lower FFN parameter counts alone do not identify the memory bottleneck.",
              "The original architectural breakthrough goal remains unmet.", ""]
    # Keep prose readable without changing frozen source or numeric evidence.
    spacing = {"after22":"after 22", "completed36":"completed 36", ";38":"; 38",
               "all36":"all 36", "seed17":"seed 17", "is6.":"is 6.", "gives9.":"gives 9.",
               "gives8.":"gives 8.", "gives6.":"gives 6.", "only1.":"only 1.",
               "costs56.":"costs 56.", "passed40":"passed 40", "to0.":"to 0.",
               "and1.":"and 1.", "The14":"The 14", "C512":"C=512", "RTX4070":"RTX 4070",
               "GPU8GiB":"GPU with 8 GiB", "PyTorch2.":"PyTorch 2.", "Width384":"Width 384",
               "V4096":"V=4096", "h456":"h=456", "h1024":"h=1024", "chunks512":"chunks of 512",
               "Seeds17":"Seeds 17", ";12":"; 12", "LR0.":"LR 0.", "decay0.":"decay 0.",
               "clip1":"clip 1", "score16,":"score 16,", "or32,":"or 32,", "from12":"from 12"}
    for old,new in spacing.items():
        lines = [line.replace(old,new) for line in lines]
    Path("research/token_memory_results.md").write_text("\n".join(lines),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    colors = {"native":"#8a96a8","block":"#356b9c","loss_chunks":"#17856b"}
    for ax,(variant,context) in zip(axes.flat,[(v,c) for v in ("gelu_narrow","swiglu") for c in (128,512)],strict=True):
        for policy in colors:
            subset = [r for r in rows if r["variant"]==variant and r["context"]==context and r["policy"]==policy]
            ax.scatter([r["median_update_ms"] for r in subset],[r["peak_allocated_mib"] for r in subset],
                       color=colors[policy],s=45,label=policy)
        ax.set_title(f"{variant}, context {context}")
        ax.set_xlabel("Median update (ms)")
        ax.set_ylabel("Peak allocated training memory (MiB)")
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Same model and training budget; three seeds per execution policy")
    Path("research/figures").mkdir(exist_ok=True)
    fig.savefig("research/figures/token_memory.png",dpi=160)
    fig.savefig("research/figures/token_memory.svg")
    print(json.dumps(groups,indent=2))
