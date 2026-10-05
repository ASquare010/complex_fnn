"""One runner for every FFN, with immutable provenance and exact local resume."""

import contextlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from dataset.loader import Dataset
from models.base_transformer.transformer import Model as DenseTransformer
from models.channel_curve_transformer.transformer import Model as CurveWideTransformer
from settings import Evaluation, Experiment, ModelCounts, RunResult, TaskScore
from storage import digest, identity, in_dump, read_json, write_json

MODELS = {
    "base_transformer": DenseTransformer,
    "channel_curve_transformer": CurveWideTransformer,
}


def source_files():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / "src").rglob("*.py"))
    }


def git_state():
    root = Path(__file__).resolve().parents[1]
    prefix = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    try:
        head = subprocess.check_output(
            prefix + ["rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(prefix + ["status", "--porcelain"], text=True))
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def autocast(training):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if training.precision == "bf16"
        else contextlib.nullcontext()
    )


@torch.no_grad()
def evaluate(model, dataset, training, split="valid", batches=None):
    model.eval()
    total, count, index = 0.0, 0, 0
    while batches is None or index < batches:
        batch = dataset.batch(split, training.batch_size, offset=index * training.batch_size)
        if batch is None:
            break
        batch = batch.to(training.device)
        x, y = batch.inputs, batch.targets
        with autocast(training):
            logits = model(x)
        total += F.cross_entropy(
            logits.float().flatten(0, 1), y.flatten(), reduction="sum", ignore_index=-100
        ).item()
        count += (y != -100).sum().item()
        index += 1
    if not count:
        raise ValueError("No evaluation targets")
    nll = total / count
    complete = (
        batches is None
        or dataset.batch(split, training.batch_size, offset=index * training.batch_size) is None
    )
    return Evaluation(nll, math.exp(min(nll, 80)), count, split, complete)


@torch.no_grad()
def exact_match(model, dataset, training, split):
    """Greedy answer generation: no gold answer tokens are given to the model."""
    if dataset.kind != "tasks":
        return {}
    model.eval()
    tokenizer = Tokenizer.from_file(str(dataset.path / "tokenizer.json"))
    groups = {}
    for row in dataset.splits[split]:
        ids = list(row["prompt_ids"])
        generated = []
        # ponytail: serial greedy decoding is a correctness reference, not a speed
        # benchmark. Add batched KV-cache decoding before making serving claims.
        for _ in range(16):
            if len(ids) > model.config.context:
                break
            with autocast(training):
                next_id = int(model(torch.tensor([ids], device=training.device))[0, -1].argmax())
            generated.append(next_id)
            ids.append(next_id)
            if next_id == dataset.eos_id:
                break
        key = f"{row['task']}/difficulty_{row['difficulty']}"
        result = groups.setdefault(key, {"correct": 0, "examples": 0})
        result["examples"] += 1
        ended = bool(generated) and generated[-1] == dataset.eos_id
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        result["correct"] += int(ended and 0 not in generated and decoded == row["answer"].strip())
    for result in groups.values():
        result["accuracy"] = result["correct"] / result["examples"]
    return {name: TaskScore(**score) for name, score in groups.items()}


def learning_rate(training, step):
    if step <= training.warmup_steps:
        return training.learning_rate * step / max(1, training.warmup_steps)
    phase = (step - training.warmup_steps) / (training.steps - training.warmup_steps)
    return training.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * phase)))


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _train(config, model_class, resume=None, stop_after=None):
    config.model.validate()
    config.training.validate()
    cfg = config.training
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support BF16")
    data = Dataset(config.dataset, config.model.context)
    if data.vocab_size != config.model.vocab_size:
        raise ValueError("Model and dataset vocabulary budgets differ")
    model = model_class(config.model, cfg.seed).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    sampler = torch.Generator().manual_seed(cfg.seed)
    sources = source_files()
    signature = {
        "config": config.to_dict(),
        "dataset_sha256": data.fingerprint,
        "sources": sources,
        "torch": str(torch.__version__),
        "device": (torch.cuda.get_device_name() if cfg.device == "cuda" else "cpu"),
    }
    start, best, trained_tokens, input_tokens, train_seconds, previous_wall = (
        0,
        math.inf,
        0,
        0,
        0.0,
        0.0,
    )
    peak_allocated = peak_reserved = None
    if resume:
        run = in_dump(resume)
        if read_json(run / "provenance.json")["signature"] != signature:
            raise ValueError("Resume configuration, dataset, code or runtime differs")
        state = torch.load(run / "last.pt", map_location=cfg.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if cfg.device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        start, best = state["step"], state["best_valid_nll"]
        trained_tokens, input_tokens = state["trained_tokens"], state["input_tokens"]
        train_seconds, previous_wall = state["train_seconds"], state["wall_seconds"]
        peak_allocated, peak_reserved = state["peak_allocated"], state["peak_reserved"]
        # Discard log entries beyond the last atomic checkpoint after an interruption.
        events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        (run / "metrics.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in events if x["step"] <= start)
        )
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = in_dump(Path("dump/runs") / f"{stamp}-{uuid.uuid4().hex[:8]}")
        run.mkdir(parents=True)
        write_json(run / "config.json", config.to_dict())
        write_json(
            run / "provenance.json",
            {
                "signature": signature,
                "git": git_state(),
                "python": sys.version,
                "platform": platform.platform(),
                "created_utc": stamp,
                "signature_sha256": identity(signature),
            },
        )
        root = Path(__file__).resolve().parents[1]
        for name in sources:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        (run / "metrics.jsonl").touch()
    terminal = min(cfg.steps, stop_after) if stop_after is not None else cfg.steps
    if terminal <= start:
        raise ValueError("No remaining steps within the requested budget")
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    clock_start = time.perf_counter()
    last_evaluation = None
    for step in range(start + 1, terminal + 1):
        model.train()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        batch = data.batch("train", cfg.batch_size, sampler).to(cfg.device)
        x, y = batch.inputs, batch.targets
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(cfg, step)
        with autocast(cfg):
            logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite training loss at step {step}; last checkpoint retained"
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
            peak_allocated = max(peak_allocated or 0, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved or 0, torch.cuda.max_memory_reserved())
        train_seconds += time.perf_counter() - tick
        trained_tokens += int((y != -100).sum())
        input_tokens += x.numel()
        event = {
            "step": step,
            "train_nll": float(loss.detach()),
            "grad_norm": float(norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "trained_target_tokens": trained_tokens,
            "input_tokens_including_padding": input_tokens,
        }
        if step % cfg.eval_every == 0 or step == terminal:
            last_evaluation = evaluate(model, data, cfg, batches=cfg.eval_batches)
            event["validation"] = asdict(last_evaluation)
            improved = last_evaluation.nll < best
            best = min(best, last_evaluation.nll)
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "sampler": sampler.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else [],
                "best_valid_nll": best,
                "trained_tokens": trained_tokens,
                "input_tokens": input_tokens,
                "train_seconds": train_seconds,
                "wall_seconds": previous_wall + time.perf_counter() - clock_start,
                "peak_allocated": peak_allocated,
                "peak_reserved": peak_reserved,
            }
            save_checkpoint(run / "last.pt", state)
            if improved:
                save_checkpoint(run / "best.pt", state)
            if cfg.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
        else:
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
    summary = {
        "format": 1,
        "name": config.name,
        "hypothesis": config.hypothesis,
        "run": str(run.relative_to(Path.cwd())),
        "completed": terminal == cfg.steps,
        "step": terminal,
        "seed": cfg.seed,
        "model": asdict(config.model),
        "training": asdict(cfg),
        "counts": asdict(model.counts()),
        "validation": asdict(last_evaluation),
        "best_valid_nll": best,
        "trained_target_tokens": trained_tokens,
        "input_tokens_including_padding": input_tokens,
        "training_seconds": train_seconds,
        "wall_seconds": previous_wall + time.perf_counter() - clock_start,
        "training_input_tokens_per_second": input_tokens / train_seconds,
        "peak_training_allocated_bytes": peak_allocated,
        "peak_training_reserved_bytes": peak_reserved,
        "dataset_sha256": data.fingerprint,
        "source_sha256": identity(sources),
        "device": signature["device"],
        "precision": cfg.precision,
        "claim": "measurement only; not an automatic research success",
    }
    write_json(run / "summary.json", summary)
    return summary


def _evaluate_run(run, model_class, split="valid", final=False, checkpoint="last"):
    if split not in ("valid", "test", "ood") or checkpoint not in ("last", "best"):
        raise ValueError("Unknown split or checkpoint")
    if split in ("test", "ood") and not final:
        raise ValueError("Held-out evaluation requires --final after freezing the recipe")
    run = in_dump(run)
    destination = run / f"evaluation-{split}-{checkpoint}.json"
    if destination.exists():
        raise ValueError("Evaluation already recorded; do not overwrite held-out evidence")
    config = Experiment.load(run / "config.json")
    torch.set_num_threads(config.training.threads)
    data = Dataset(config.dataset, config.model.context)
    signature = read_json(run / "provenance.json")["signature"]
    if signature["dataset_sha256"] != data.fingerprint or signature["sources"] != source_files():
        raise ValueError("Dataset or evaluation code differs from the recorded run")
    model = model_class(config.model, config.training.seed).to(config.training.device)
    state = torch.load(
        run / f"{checkpoint}.pt", map_location=config.training.device, weights_only=False
    )
    model.load_state_dict(state["model"])
    result = evaluate(model, data, config.training, split=split)
    result = replace(result, exact_match=exact_match(model, data, config.training, split))
    record = {
        **asdict(result),
        "checkpoint": checkpoint,
        "step": state["step"],
        "checkpoint_sha256": digest(run / f"{checkpoint}.pt"),
        "dataset_sha256": data.fingerprint,
        "final_opt_in": final,
    }
    write_json(destination, record)
    if split == "valid":
        from leaderboard import update_leaderboard

        update_leaderboard()
    return result


class Trainer:
    """Public lifecycle: train/eval own the run; Model.train() keeps PyTorch semantics."""

    def __init__(self, config: Experiment, run: str | Path | None = None):
        self.config = config
        self.model_class = MODELS[config.model.name]
        self.run = Path(run) if run else None

    def train(self, stop_after: int | None = None) -> RunResult:
        result = _train(self.config, self.model_class, resume=self.run, stop_after=stop_after)
        self.run = Path(result["run"])
        return RunResult(
            result["run"],
            result["completed"],
            result["step"],
            Evaluation(**result["validation"]),
            ModelCounts(**result["counts"]),
        )

    def eval(
        self, split: str = "valid", final: bool = False, checkpoint: str = "last"
    ) -> Evaluation:
        if self.run is None:
            raise ValueError("Train or load a run before evaluation")
        return _evaluate_run(self.run, self.model_class, split, final, checkpoint)

    @classmethod
    def load(cls, run: str | Path):
        run = in_dump(run)
        return cls(Experiment.load(run / "config.json"), run)
