"""Equal-parameter, equal-depth FFN screen on current language datasets."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.branch_sigmoid.data import batch
from models.branch_sigmoid.data import windows
from models.branch_sigmoid.data import load_data
from models.branch_sigmoid.runtime import autocast, save
from storage import digest, in_dump, read_json, write_json


from models.branch_sigmoid.loader import sources, build


@torch.no_grad()
def evaluate(model, examples, device, control="correct", microbatch=4):
    model.eval()
    groups = {
        k: dict(
            loss_sum=0.0, targets=0, correct=0, history_loss_sum=0.0, history_targets=0, windows=0
        )
        for k in ("text", "chat")
    }
    for kind, metric in groups.items():
        selected = [e for e in examples if e[0]["kind"] == kind]
        for start in range(0, len(selected), microbatch):
            current = selected[start : start + microbatch]
            x, y, past, valid = batch(current, model.encoded, device)
            has_history = valid.any(1)
            if control == "masked":
                valid.zero_()
            with autocast(device):
                logits = model(x, past, valid)
                losses = F.cross_entropy(
                    logits.float().flatten(0, 1), y.flatten(), reduction="none"
                ).view_as(y)
            scored = y.ne(-100)
            metric["loss_sum"] += losses.sum().item()
            metric["targets"] += scored.sum().item()
            metric["correct"] += ((logits.argmax(-1) == y) & scored).sum().item()
            metric["history_loss_sum"] += losses[has_history].sum().item()
            metric["history_targets"] += scored[has_history].sum().item()
            metric["windows"] += len(current)
        metric["nll"] = metric["loss_sum"] / metric["targets"]
        metric["token_accuracy"] = metric["correct"] / metric["targets"]
        metric["history_nll"] = (
            metric["history_loss_sum"] / metric["history_targets"]
            if metric["history_targets"]
            else None
        )
    return dict(control=control, complete_split=True, groups=groups)


def run(recipe, kind, resume=False, stop_after=None):
    torch.set_num_threads(4)
    t, device = recipe["training"], recipe["training"]["device"]
    data, tokenizer, manifest = load_data(recipe["data"])
    if digest(recipe["encoder"]) != manifest["encoder_sha256"]:
        raise ValueError("Encoder identity mismatch")
    examples = windows(data["train"])
    pools = {k: [e for e in examples if e[0]["kind"] == k] for k in ("text", "chat")}
    valid_examples = windows(data["valid"])
    model = build(recipe, kind).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
        fused=device == "cuda",
    )
    source = sources()
    rng = random.Random(t["seed"])
    torch.manual_seed(t["seed"])
    output = in_dump(recipe["output"]) / kind
    history, schedule, total, trained_tokens = [], [], 0, 0
    if resume:
        state = torch.load(output / "last.pt", weights_only=True, map_location="cpu")
        if (
            state["recipe"] != recipe
            or state["sources"] != source
            or state["dataset"] != manifest
            or state["torch_version"] != str(torch.__version__)
        ):
            raise ValueError("Resume requires identical recipe, sources, dataset and Torch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, schedule, total, trained_tokens = (
            state["history"],
            state["schedule_ids"],
            state["total_steps"],
            state["trained_tokens"],
        )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Run exists; use --resume")
        output.mkdir(parents=True)
        for path in source:
            target = output / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        write_json(output / "recipe.json", recipe)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    added = 0
    for step in range(total + 1, t["updates"] + 1):
        begin = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        lr = (
            t["learning_rate"]
            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(1, t["updates"] - 1))))
            * min(1, step / t["warmup"])
        )
        for group in opt.param_groups:
            group["lr"] = lr
        selected = [
            [
                rng.choice(pools["chat" if rng.random() < t["chat_fraction"] else "text"])
                for _ in range(t["microbatch"])
            ]
            for _ in range(t["accumulation"])
        ]
        targets = sum(
            int(row["score"][j * 256 + 1 : (j + 1) * 256 + 1].sum())
            for micro in selected
            for row, j in micro
        )
        schedule.append(
            hashlib.sha256(
                json.dumps([(r["document_id"], j) for micro in selected for r, j in micro]).encode()
            ).hexdigest()
        )
        value = 0.0
        for micro in selected:
            x, y, past, mask = batch(micro, model.encoded, device)
            with autocast(device):
                loss = (
                    F.cross_entropy(
                        model(x, past, mask).float().flatten(0, 1), y.flatten(), reduction="sum"
                    )
                    / targets
                )
            loss.backward()
            value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"], error_if_nonfinite=True)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        trained_tokens += targets
        added += 1
        stats = dict(
            update=step,
            nll=value,
            targets=targets,
            seconds=time.perf_counter() - begin,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0,
        )
        history.append(stats)
        if step % t["log_every"] == 0:
            write_json(output / "progress.json", stats)
            print(kind, stats, flush=True)
        pause = stop_after is not None and added >= stop_after
        if pause or step % t["save_every"] == 0 or step == t["updates"]:
            state = dict(
                format="ffn-hybrid-v1",
                kind=kind,
                recipe=recipe,
                dataset=manifest,
                sources=source,
                torch_version=str(torch.__version__),
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                tokenizer=tokenizer.to_str(),
                rng=rng.getstate(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                total_steps=step,
                trained_tokens=trained_tokens,
                history=history,
                schedule_ids=schedule,
            )
            save(output / "last.pt", state)
        if pause:
            return output / "last.pt"
    evaluation = [evaluate(model, valid_examples, device, microbatch=t["microbatch"])]
    if model.encoded:
        evaluation.append(evaluate(model, valid_examples, device, "masked", t["microbatch"]))
    record = dict(
        task="ffn_single_comparison",
        completed=True,
        kind=kind,
        recipe=recipe,
        dataset=manifest,
        sources=source,
        updates=t["updates"],
        trained_tokens=trained_tokens,
        schedule_sha256=hashlib.sha256("".join(schedule).encode()).hexdigest(),
        core_parameters=sum(p.numel() for p in model.parameters()),
        ffn_parameters=sum(p.numel() for b in model.blocks for p in b.ffn.parameters()),
        encoder_parameters=4798720 if model.encoded else 0,
        peak_allocated_mib=max(h["peak_allocated_mib"] for h in history),
        peak_reserved_mib=max(h["peak_reserved_mib"] for h in history),
        training_seconds=sum(h["seconds"] for h in history),
        validation=evaluation,
        checkpoint=str(output / "last.pt"),
        checkpoint_sha256=digest(output / "last.pt"),
    )
    write_json(output / "result.json", record)
    write_json(Path("records") / f"{recipe['name']}-{kind}.json", record)
    print("FULL VALIDATION", kind, json.dumps(evaluation), flush=True)
    return output / "last.pt"
