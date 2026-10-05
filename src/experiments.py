"""Study planning and small result exports; never launches a training campaign."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from leaderboard import update_leaderboard
from settings import Experiment
from storage import in_dump, read_json, write_json
from trainer import Trainer


def make_plan(path):
    study = read_json(path)
    output = in_dump(study["output"])
    if output.exists():
        raise ValueError("Plan already exists; choose a new study ID")
    configs, inspection = {}, []
    for variant in study["variants"]:
        if set(variant["model"]) - {
            "name",
            "ffn",
            "hidden",
            "groups",
            "loops",
            "input_rank",
            "output_rank",
            "ffn_output_init_scale",
        }:
            raise ValueError("FFN-only study: variants cannot change the Transformer backbone")
        for seed in study["seeds"]:
            for rate in study["learning_rates"]:
                name = f"{variant['name']}-s{seed}-lr{rate:g}"
                config = {
                    "name": name,
                    "dataset": study["dataset"],
                    "hypothesis": variant["hypothesis"],
                    "model": {**study["model"], **variant["model"]},
                    "training": {**study["training"], "seed": seed, "learning_rate": rate},
                }
                model_config = Experiment.from_dict(config)
                configs[name] = config
                inspection.append(
                    {
                        "name": name,
                        **asdict(
                            Trainer(model_config).model_class(model_config.model, seed).counts()
                        ),
                    }
                )
    output.mkdir(parents=True)
    write_json(output / "study.json", study)
    for name, config in configs.items():
        write_json(output / "configs" / f"{name}.json", config)
    write_json(output / "counts.json", inspection)
    return {
        "output": str(output),
        "runs_planned": len(configs),
        "training_started": False,
        "variants": {row["name"].rsplit("-s", 1)[0]: row for row in inspection},
    }


def export_run(run, name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", name):
        raise ValueError("Use a short lowercase record name")
    run = in_dump(run)
    summary = read_json(run / "summary.json")
    result = {
        "summary": summary,
        "config": read_json(run / "config.json"),
        "evaluations": [read_json(p) for p in sorted(run.glob("evaluation-*.json"))],
    }
    if len(json.dumps(result).encode()) > 65536:
        raise ValueError("Compact record exceeds 64 KiB")
    destination = Path("records") / f"{name}.json"
    if destination.exists():
        raise ValueError("Record exists; give this result a new identity")
    write_json(destination, result)
    update_leaderboard()
    return {"record": str(destination), "committed": False}


def compare(runs):
    rows = [read_json(in_dump(p) / "summary.json") for p in runs]
    required = ("dataset_sha256", "source_sha256", "step", "trained_target_tokens", "precision")
    differences = [key for key in required if len({json.dumps(r[key]) for r in rows}) > 1]
    for key in ("width", "layers", "heads", "context", "vocab_size", "rope_base"):
        if len({r["model"][key] for r in rows}) > 1:
            differences.append(f"model.{key}")
    return {
        "comparison_warnings": differences,
        "results": rows,
        "note": "Compare paired seeds after equal validation-only tuning. No automatic win claim.",
    }
