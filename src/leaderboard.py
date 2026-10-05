"""Rebuild dataset leaderboards from recorded, complete validation evaluations."""

import math
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from settings import ModelConfig
from storage import identity, read_json


def update_leaderboard():
    records = []
    for path in sorted(Path("records").glob("*.json")):
        record = read_json(path)
        if isinstance(record, dict) and {"summary", "config", "evaluations"} <= record.keys():
            records.append((path, record))
    # Include fresh runs even before their compact evidence has been exported.
    for path in sorted(Path("dump/runs").glob("*/summary.json")):
        run = path.parent
        records.append(
            (
                path,
                {
                    "summary": read_json(path),
                    "config": read_json(run / "config.json"),
                    "evaluations": [read_json(p) for p in sorted(run.glob("evaluation-*.json"))],
                },
            )
        )
    rows = {}
    for path, record in records:
        summary, config = record["summary"], record["config"]
        dataset = Path(config["dataset"]).name
        corpus = next(
            (name for name in ("tinystories", "wikitext") if dataset.startswith(name)), None
        )
        if corpus is None or not summary.get("completed"):
            continue
        for evaluation in record["evaluations"]:
            if (
                evaluation.get("split") != "valid"
                or not evaluation.get("complete_split")
                or evaluation.get("checkpoint") != "last"
                or evaluation.get("step") != summary["step"]
            ):
                continue
            if evaluation["dataset_sha256"] != summary["dataset_sha256"]:
                raise ValueError(f"Dataset identity differs in {path}")
            if not math.isfinite(evaluation["nll"]):
                raise ValueError(f"Non-finite validation loss in {path}")
            model, training = summary["model"], summary["training"]
            cohort = (
                corpus,
                summary["dataset_sha256"],
                tuple(model[k] for k in ("width", "layers", "heads", "context", "vocab_size")),
                model.get("rope_base", 10000),
                summary["step"],
                summary["trained_target_tokens"],
                evaluation["targets"],
                summary["precision"],
                tuple(
                    training[k]
                    for k in ("batch_size", "learning_rate", "warmup_steps", "weight_decay")
                ),
            )
            architecture = asdict(ModelConfig(**model))
            if model["name"] == "base_transformer":
                for field in ("groups", "input_rank", "output_rank"):
                    architecture.pop(field)  # Unused by ordinary dense FFNs.
            # Exact recipe/seed/loss replays share a row; retain all their provenance.
            key = identity(
                {
                    "cohort": cohort,
                    "counts": summary["counts"],
                    "model": architecture,
                    "seed": summary["seed"],
                    "nll": evaluation["nll"],
                }
            )
            if key not in rows:
                rows[key] = {
                    "cohort": cohort,
                    "summary": summary,
                    "evaluation": evaluation,
                    "evidence": [],
                    "measurements": {},
                }
            rows[key]["evidence"].append(path)
            rows[key]["measurements"][evaluation["checkpoint_sha256"]] = summary

    lines = [
        "# Dataset leaderboards",
        "",
        "Generated from recorded results. Lower validation NLL and perplexity are better.",
        "Each table ranks completed runs on the **entire validation split**, using the last",
        "checkpoint. Sampled training validation, test scores, numerical checks and resource",
        "screens do not enter these rankings. Exact same-recipe/seed/loss replays rank once,",
        "with all evidence retained and resource ranges shown. Different losses or seeds",
        "remain separate. Replays are not independent seed confirmation. These are measurements,",
        "not multi-seed confirmation or automatic research-success claims.",
        "",
        "Backbone, data fingerprint, token budget, precision and training settings define",
        "separate comparison groups. Source revisions and FFN recipes are preserved in the",
        "linked evidence; grouping does not certify identical implementations. Do not compare",
        "loss across datasets or treat a smaller-backbone table as a matched FFN comparison.",
        "",
        "VRAM is peak **allocated training memory** from each run, including the whole model.",
        "Training seconds exclude validation and are observational, not isolated speed benchmarks.",
        "FFN reduction refers only to FFN weights, relative to the largest full SwiGLU",
        "baseline in the same table. Total weights include the unchanged backbone.",
        "",
        "Refresh: `uv run python -m main leaderboard`. Complete validation through the shared",
        "trainer and result export both refresh this file automatically. Frozen experimental",
        "launchers must export their compact evidence and then refresh the table.",
        "",
    ]
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["cohort"]].append(row)
    for corpus, title in (("tinystories", "TinyStories"), ("wikitext", "WikiText-2")):
        lines.extend([f"## {title}", ""])
        selected = sorted(
            (key for key in groups if key[0] == corpus), key=lambda key: (-key[2][0], repr(key))
        )
        for cohort in selected:
            ranked = sorted(
                groups[cohort], key=lambda r: (r["evaluation"]["nll"], r["summary"]["name"])
            )
            _, fingerprint, backbone, rope, steps, tokens, targets, precision, training = cohort
            width, layers, heads, context, vocab = backbone
            batch, rate, warmup, decay = training
            reference = max(
                (
                    r["summary"]["counts"]["ffn_parameters"]
                    for r in ranked
                    if "full-swiglu" in r["summary"]["name"]
                    and r["summary"]["model"]["name"] == "base_transformer"
                ),
                default=0,
            )
            lines.extend(
                [
                    f"### Width {width}, {layers} layers, {steps:,} updates",
                    "",
                    f"{heads} heads; context {context}; vocabulary {vocab}; RoPE {rope:g}; "
                    f"{tokens:,} trained targets; {targets:,} validation targets; {precision}; "
                    f"batch {batch}; learning rate {rate:g}; warmup {warmup}; weight decay {decay:g}.",
                    f"Data SHA-256: `{fingerprint}`.",
                    "",
                    "| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |",
                    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for rank, row in enumerate(ranked, 1):
                summary, evaluation = row["summary"], row["evaluation"]
                model, counts = summary["model"], summary["counts"]
                recipe = (
                    summary["name"]
                    .removesuffix(f"-s{summary['seed']}-lr{summary['training']['learning_rate']:g}")
                    .replace("|", "/")
                )
                recipe = re.sub(r"^.*?(?:tinystories|wikitext)-", "", recipe)
                cut = (
                    f"{100 * (1 - counts['ffn_parameters'] / reference):.2f}%" if reference else "—"
                )
                # Prefer tracked evidence so links survive removal of ignored dump/ runs.
                evidence = sorted(row["evidence"], key=lambda p: (p.parts[0] != "records", str(p)))
                link = f"[record](../{evidence[0].as_posix()})"
                if len(evidence) > 1:
                    link += f" ([all {len(evidence)} sources](#repeated-measurement-evidence))"
                measurements = row["measurements"].values()
                memory = [m["peak_training_allocated_bytes"] / 2**20 for m in measurements]
                seconds = [m["training_seconds"] for m in measurements]
                memory_range = f"{min(memory):.1f}"
                time_range = f"{min(seconds):.2f}"
                if max(memory) != min(memory):
                    memory_range += f"–{max(memory):.1f}"
                if max(seconds) != min(seconds):
                    time_range += f"–{max(seconds):.2f}"
                lines.append(
                    f"| {rank} | {recipe} (`{model['ffn']}`, h={model['hidden']}) "
                    f"| {summary['seed']} | {evaluation['nll']:.6f} "
                    f"| {evaluation['perplexity']:.3f} | {counts['total_parameters']:,} "
                    f"| {counts['ffn_parameters']:,} | {cut} "
                    f"| {memory_range} | {time_range} | {link} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Tested recipes without a ranked language result",
            "",
            "These are not assigned invented language scores. Their reports retain numerical,",
            "resource or synthetic-task results. Execution revisions are not new mathematical models.",
            "",
            "| Recipe | Evidence / status |",
            "| --- | --- |",
            "| PairFlux (combined), no-exchange, no-curve | [Synthetic-task screen](retired_models/pairflux_transformer/result.md); curve-only has separate language rows above |",
            "| Looped BlockShuffle / repeated dense | [Synthetic and execution checks](retired_models/looped_blockshuffle_transformer/result.md) |",
            "| Shared-basis v1 / gate transport v2 | [Resource screens](shared_gate_transport_hypothesis.md); plain sharing and untied controls have language rows above |",
            "| Polynomial sketch / single sketch | [Resource failures](polynomial_sketch_result.md) |",
            "| Group-product execution v1–v3 | [Resource revisions](group_product_result.md); v4 language rows above |",
            "| Channel-curve execution v1 | [Resource failure](channel_curve_result.md); v2 language rows above |",
            "| Learned basis templates | [Resource failure](basis_readout_result.md); fixed/tied/cached language rows above |",
            "| Feature-flow execution v1–v3 | [Resource revisions](feature_flow_result.md); v4 language rows above |",
            "| Readout reuse v1 | [Resource failure](readout_reuse_result.md) |",
            "| Readout reuse v2 / v3 | [Forward failure](readout_reuse_v2_result.md) / [gradient failure](readout_reuse_v3_result.md) |",
            "| Readout reuse v4 | [Gradient failure](readout_reuse_v4_result.md); retired before language |",
            "| Readout reuse v5 | [Resource failure](readout_reuse_v5_result.md); numerical checks passed, but slowdown exceeds the gate; no language result |",
            "| Paired readout v1 | [Resource report](paired_readout_result.md); numerical checks passed; misses all-control memory/speed gate, no language score |",
            "",
            "Frozen-weight removal experiments stay in each study's result report; they are",
            "interventions on trained checkpoints, not independently trained model entries.",
            "",
        ]
    )
    lines.extend(
        [
            "## Repeated measurement evidence",
            "",
            "<details>",
            "<summary>All grouped sources</summary>",
            "",
        ]
    )
    for row in rows.values():
        if len(row["evidence"]) > 1:
            links = ", ".join(f"[{p.name}](../{p.as_posix()})" for p in row["evidence"])
            lines.append(f"- {row['summary']['name']}: {links}")
    lines.extend(["", "</details>", ""])
    destination = Path("docs/leaderboard.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(destination)
    return {"leaderboard": str(destination), "ranked_results": len(rows), "tables": len(groups)}
