"""Isolated higher-compression transfer pilot; never replaces the working model."""

import argparse
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from models.position_compressor.data import batch, load_data
from models.position_compressor.evaluation import evaluate, sources
from models.position_compressor.training import Session
from models.position_compressor.transformer import Config, Model
from storage import digest, in_dump, write_json


def synthetic_rows(count, seed, patterns=False):
    """Coverage data, never generated from evaluation text."""
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        if patterns:
            alphabet = rng.sample(range(3, 4096), rng.randint(1, 64))
            length = rng.randint(1, 256)
            if rng.random() < 0.5:
                ids = rng.choices(alphabet, k=length)
            else:
                motif = rng.choices(alphabet, k=rng.randint(1, 16))
                ids = (motif * ((length + len(motif) - 1) // len(motif)))[:length]
        else:
            ids = [rng.randrange(3, 4096) for _ in range(rng.randint(32, 128))]
        rows.append({"ids": ids})
    return rows


@torch.no_grad()
def token_recovery(model, rows):
    correct = total = exact = 0
    model.eval()
    for start in range(0, len(rows), 8):
        tokens, mask, _, _ = batch(rows[start : start + 8], "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(*model.encode(tokens, mask))
        equal = output == tokens
        exact += (equal | ~mask).all(1).sum().item()
        correct += (equal & mask).sum().item()
        total += mask.sum().item()
    return {
        "examples": len(rows),
        "exact": exact,
        "token_accuracy": correct / total,
        "tokens": total,
        "exact_rate": exact / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--span", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = in_dump(args.output)
    if output.exists():
        raise ValueError("Do not replace existing study evidence")
    parent = Path("dump/position-pattern-5000-clean-v1/last.pt")
    state = torch.load(parent, map_location="cpu", weights_only=False)
    for name, expected in state["protocol"]["sources"].items():
        if digest(name) != expected:
            raise ValueError(f"Parent source changed: {name}")
    tokenizer, rows, manifest = load_data("dump/data/paragraph-fineweb-v1")
    config = Config(**{**state["protocol"]["model"], "span": args.span})
    model = Model(config, seed=17).cuda()
    # Share pretrained encoder/decoder knowledge. Reinitialize both span-dependent maps.
    shared = {
        k: v for k, v in state["model"].items() if k not in ("compress.weight", "expand.weight")
    }
    missing = model.load_state_dict(shared, strict=False)
    assert set(missing.missing_keys) == {"compress.weight", "expand.weight"}
    assert not missing.unexpected_keys
    protocol = {
        "model": asdict(config),
        "seed": 17,
        "steps": args.steps,
        "parent_step": 0,
        "microbatch": 4,
        "accumulation": 4,
        "epsilon": 0.0,
        "parent_checkpoint_sha256": digest(parent),
        "dataset": manifest,
        "sources": sources(),
        "schedule": "cosine 3e-4 to 3e-5 over pilot",
        "initialization": "pretrained shared weights; fresh compress/expand maps and AdamW",
        "comparison_limit": "transfer pilot, not a matched-total-compute architecture comparison",
        "data": "50k natural + 25k uniform(seed 219) + 25k patterns(seed 217)",
    }
    train = rows["train"] + synthetic_rows(25000, 219) + synthetic_rows(25000, 217, True)
    uniform = synthetic_rows(256, 220)
    pattern = synthetic_rows(512, 218, True)
    seen = {tuple(r["ids"]) for r in train}
    uniform = [r for r in uniform if tuple(r["ids"]) not in seen]
    pattern = [r for r in pattern if tuple(r["ids"]) not in seen]
    session = Session(model, tokenizer, protocol)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    for name in protocol["sources"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    print("PARAMETERS", sum(p.numel() for p in model.parameters()), flush=True)
    torch.cuda.reset_peak_memory_stats()
    while session.step < args.steps:
        stats = session.train_step(train)
        if session.step % 100 == 0:
            print(stats, flush=True)
        if session.step % 250 == 0 or session.step == args.steps:
            session.save(output / "last.pt")
        if session.step % 500 == 0:
            valid = evaluate(model, rows["valid"], tokenizer)
            write_json(output / f"validation-{session.step}.json", valid)
            print(
                "FULL VALIDATION",
                session.step,
                {k: valid[k] for k in ("nll", "token_accuracy", "exact_match")},
                flush=True,
            )
    valid = evaluate(model, rows["valid"], tokenizer)
    record = {
        "task": "paragraph_reconstruction",
        "protocol": protocol,
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation": valid,
        "uniform": token_recovery(model, uniform),
        "patterns": token_recovery(model, pattern),
        "history": session.history,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_sha256": digest(output / "last.pt"),
        "attention_pairs_at_256_tokens": (256 // args.span) ** 2,
        "attention_pair_note": "theoretical memory self-attention only; no consuming LLM yet",
    }
    write_json(output / "result.json", record)
    write_json(Path("records") / f"{output.name}.json", record)
    from leaderboard import update_leaderboard

    update_leaderboard()
    print(
        "COMPLETE",
        {k: valid[k] for k in ("nll", "token_accuracy", "exact_match")},
        record["uniform"],
        record["patterns"],
        flush=True,
    )


if __name__ == "__main__":
    main()
