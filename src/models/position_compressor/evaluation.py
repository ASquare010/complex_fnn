"""Full reconstruction evaluation and exact source identities for training."""

from pathlib import Path

import torch
from torch.nn import functional as F

from models.position_compressor.data import batch
from storage import digest


class Levenshtein:
    """Small dependency-free edit distance for reconstruction diagnostics."""
    @staticmethod
    def distance(left, right):
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for i, a in enumerate(left, 1):
            current = [i]
            for j, b in enumerate(right, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (a != b)))
            previous = current
        return previous[-1]


def sources():
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "src/models/position_compressor").glob("*.py"))
    paths += [
        root / "src/models/components.py",
        root / "src/models/branch_sigmoid/transformer.py",
    ]
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


@torch.no_grad()
def evaluate(model, rows, tokenizer, intervention="correct"):
    model.eval()
    sums = dict(
        nll=0.0,
        targets=0,
        correct_tokens=0,
        exact=0,
        token_edits=0,
        char_edits=0,
        characters=0,
        latent_vectors=0,
        latent_features=0,
        latent_bytes=0,
        utf8_bytes=0,
    )
    predictions = []
    for start in range(0, len(rows), 8):
        current = rows[start : start + 8]
        tokens, mask, _, _ = batch(current, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, lengths = model.encode(tokens, mask)
            if intervention == "zero":
                z = torch.zeros_like(z)
            elif intervention == "shuffle":
                z = z.roll(1, 0)
            logits = model.decode(z, lengths)
            sums["nll"] += F.cross_entropy(
                logits.float().flatten(0, 1),
                tokens.masked_fill(~mask, -100).flatten(),
                reduction="sum",
            ).item()
            logits[..., :3] = -torch.inf
            ids = logits.argmax(-1)
            sums["correct_tokens"] += ((ids == tokens) & mask).sum().item()
        for i, row in enumerate(current):
            predicted = ids[i, : len(row["ids"])].tolist()
            text = tokenizer.decode(predicted, skip_special_tokens=False)
            n = (len(row["ids"]) + model.config.span - 1) // model.config.span
            sums["exact"] += text == row["text"]
            sums["targets"] += len(row["ids"])
            sums["token_edits"] += Levenshtein.distance(predicted, row["ids"])
            sums["char_edits"] += Levenshtein.distance(text, row["text"])
            sums["characters"] += len(row["text"])
            sums["latent_vectors"] += n
            sums["latent_features"] += n * model.config.width
            sums["latent_bytes"] += n * model.config.width * z.element_size() + 8
            sums["utf8_bytes"] += len(row["text"].encode())
            predictions.append({"input": row["text"], "output": text, "exact": text == row["text"]})
    return {
        "examples": len(rows),
        "nll": sums["nll"] / sums["targets"],
        "token_accuracy": sums["correct_tokens"] / sums["targets"],
        "exact_match": sums["exact"] / len(rows),
        "token_edit_rate": sums["token_edits"] / sums["targets"],
        "char_edit_rate": sums["char_edits"] / sums["characters"],
        "totals": sums,
        "predictions": predictions,
        "intervention": intervention,
    }
