"""Inspect, prepare, run, resume, evaluate and export compact evidence."""

import argparse
import json
from dataclasses import asdict, is_dataclass

from dataset.prepare import prepare_remote, synthetic
from experiments import compare, export_run, make_plan
from leaderboard import update_leaderboard
from settings import DatasetConfig, Experiment
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config")
    plan = commands.add_parser("plan")
    plan.add_argument("study")
    data = commands.add_parser("prepare")
    data.add_argument("recipe")
    run = commands.add_parser("run")
    run.add_argument("config")
    run.add_argument("--stop-after", type=int)
    resume = commands.add_parser("resume")
    resume.add_argument("run")
    resume.add_argument("--stop-after", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("run")
    evaluate.add_argument("--split", choices=["valid", "test", "ood"], default="valid")
    evaluate.add_argument("--final", action="store_true")
    evaluate.add_argument("--checkpoint", choices=["last", "best"], default="last")
    export = commands.add_parser("export")
    export.add_argument("run")
    export.add_argument("name")
    comparison = commands.add_parser("compare")
    comparison.add_argument("runs", nargs="+")
    commands.add_parser("leaderboard", help="Refresh ranked dataset results from saved evidence")
    args = parser.parse_args()
    if args.command == "inspect":
        config = Experiment.load(args.config)
        result = {
            "config": config.to_dict(),
            "counts": asdict(Trainer(config).model_class(config.model).counts()),
        }
    elif args.command == "plan":
        result = make_plan(args.study)
    elif args.command == "prepare":
        recipe = DatasetConfig.load(args.recipe)
        result = synthetic(recipe) if recipe.kind == "synthetic" else prepare_remote(recipe)
    elif args.command == "run":
        result = Trainer(Experiment.load(args.config)).train(stop_after=args.stop_after)
    elif args.command == "resume":
        result = Trainer.load(args.run).train(stop_after=args.stop_after)
    elif args.command == "evaluate":
        result = Trainer.load(args.run).eval(args.split, args.final, args.checkpoint)
    elif args.command == "export":
        result = export_run(args.run, args.name)
    elif args.command == "leaderboard":
        result = update_leaderboard()
    else:
        result = compare(args.runs)
    print(json.dumps(asdict(result) if is_dataclass(result) else result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
