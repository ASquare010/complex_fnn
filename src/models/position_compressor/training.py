"""Readable continuation training with optional Poly-1 loss and measured recovery.

Poly-1 adds epsilon * (1 - probability_of_correct_token) to token cross-entropy.
The default epsilon is zero. Exact-match counts remain evaluation metrics.
"""

import math
import random
import time

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from models.position_compressor.codec import Codec
from models.position_compressor.data import batch
from models.position_compressor.evaluation import sources
from models.position_compressor.transformer import Config, Model
from storage import digest, in_dump


class Session:
    def __init__(self, model, tokenizer, protocol, optimizer=None, step=0, rng=None, history=None):
        self.model, self.tokenizer, self.protocol = model, tokenizer, protocol
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.01
        )
        self.step = step
        self.rng = rng or random.Random(protocol["seed"])
        self.history = history or []
        self.device = next(model.parameters()).device

    @classmethod
    def continue_from(cls, checkpoint, *, total_steps=5000, epsilon=0.0, device="cuda"):
        """Explicit new continuation study: preserves parent weights, optimizer and sampling RNG."""
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for path, expected in state["protocol"]["sources"].items():
            if digest(path) != expected:
                raise ValueError(f"Parent source changed: {path}; use its frozen snapshot")
        if total_steps <= state["step"] or epsilon < 0:
            raise ValueError("Continuation needs more updates and a nonnegative loss coefficient")
        model = Model(Config(**state["protocol"]["model"])).to(device)
        model.load_state_dict(state["model"])
        protocol = {
            **state["protocol"],
            "parent_checkpoint_sha256": digest(checkpoint),
            "parent_step": state["step"],
            "steps": total_steps,
            "epsilon": epsilon,
            "schedule": "cosine 3e-4 to 3e-5 over continuation only",
            "loss": "CE + epsilon*(1-p_correct)",
            "sources": sources(),
        }
        result = cls(
            model,
            Tokenizer.from_str(state["tokenizer"]),
            protocol,
            step=state["step"],
            history=state["history"],
        )
        result.optimizer.load_state_dict(state["optimizer"])
        result.rng.setstate(state["rng"])
        return result

    @classmethod
    def resume(cls, checkpoint, device="cuda"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for path, expected in state["protocol"]["sources"].items():
            if digest(path) != expected:
                raise ValueError(f"Source changed: {path}")
        model = Model(Config(**state["protocol"]["model"])).to(device)
        model.load_state_dict(state["model"])
        result = cls(
            model,
            Tokenizer.from_str(state["tokenizer"]),
            state["protocol"],
            step=state["step"],
            history=state["history"],
        )
        result.optimizer.load_state_dict(state["optimizer"])
        result.rng.setstate(state["rng"])
        return result

    def train_step(self, rows):
        p = self.protocol
        if self.step >= p["steps"]:
            raise ValueError("Registered update budget reached")
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress = (self.step - p["parent_step"]) / max(1, p["steps"] - p["parent_step"] - 1)
        lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        micros = [self.rng.choices(rows, k=p["microbatch"]) for _ in range(p["accumulation"])]
        count = sum(len(row["ids"]) for micro in micros for row in micro)
        value = ce_value = correct = 0.0
        for micro in micros:
            tokens, mask, _, _ = batch(micro, self.device)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                logits = self.model(tokens, mask)
                ce = F.cross_entropy(
                    logits.float().flatten(0, 1),
                    tokens.masked_fill(~mask, -100).flatten(),
                    reduction="none",
                ).view_as(tokens)
                objective = ce + p["epsilon"] * (1 - torch.exp(-ce))
                loss = objective[mask].sum() / count
            loss.backward()
            value += loss.item()
            ce_value += ce[mask].detach().sum().item() / count
            correct += ((logits.argmax(-1) == tokens) & mask).sum().item()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.step += 1
        result = {
            "step": self.step,
            "loss": value,
            "ce": ce_value,
            "token_accuracy": correct / count,
            "targets": count,
            "lr": lr,
            "seconds": time.perf_counter() - start,
        }
        self.history.append(result)
        return result

    def save(self, path):
        path = in_dump(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "protocol": self.protocol,
                "step": self.step,
                "history": self.history,
                "rng": self.rng.getstate(),
                "tokenizer": self.tokenizer.to_str(),
            },
            temporary,
        )
        temporary.replace(path)
        return path

    def codec(self):
        return Codec(
            self.model,
            self.tokenizer,
            self.device.type,
            "bf16" if self.device.type == "cuda" else "fp32",
        )
