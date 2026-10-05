"""Pinned dataset preparation, train-only tokenization and explicit split integrity."""

import hashlib
import os
import re
import unicodedata
from dataclasses import asdict

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from dataset.tasks import generate
from settings import DatasetConfig
from storage import digest, identity, in_dump, write_json


def document_id(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def example_id(row):
    oracle = row.get("oracle", {})
    if "edges" in oracle:
        return identity(
            {"edges": sorted(oracle["edges"]), "start": oracle["start"], "hops": oracle["hops"]}
        )
    return document_id(row["prompt"])


def prepare(output, splits, metadata, vocab_size=4096, kind="language"):
    output = in_dump(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset output is not empty; choose a new version")
    if kind not in ("language", "tasks") or vocab_size < 258:
        raise ValueError("Invalid data kind or vocabulary smaller than the byte alphabet")
    cleaned, removed, seen = {}, {}, set()
    for split in ("test", "ood", "valid", "train"):
        if split not in splits:
            continue
        kept, removed[split] = [], 0
        for row in splits[split]:
            key = example_id(row) if kind == "tasks" else document_id(row)
            if key in seen or (kind == "language" and not row.strip()):
                removed[split] += 1
                continue
            seen.add(key)
            kept.append(row)
        if not kept:
            raise ValueError(f"Empty {split} after cross-split exact deduplication")
        cleaned[split] = kept
    if not {"train", "valid"} <= cleaned.keys():
        raise ValueError("Need independent train and validation documents")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    texts = (row["prompt"] + row["answer"] if kind == "tasks" else row for row in cleaned["train"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"))
    counts = {}
    for split, rows in cleaned.items():
        ids = [example_id(row) if kind == "tasks" else document_id(row) for row in rows]
        write_json(output / f"{split}_ids.json", ids)
        if kind == "tasks":
            encoded = []
            for row in rows:
                # Separate encoding fixes the prompt/answer boundary for masked loss.
                encoded.append(
                    {
                        **row,
                        "prompt_ids": tokenizer.encode(row["prompt"]).ids,
                        "answer_ids": tokenizer.encode(row["answer"]).ids + [1],
                    }
                )
            write_json(output / f"{split}.json", encoded)
            count = sum(len(x["prompt_ids"]) + len(x["answer_ids"]) for x in encoded)
        else:
            tokens = []
            for encoded in tokenizer.encode_batch(rows):
                tokens.extend(encoded.ids)
                tokens.append(1)
            np.save(output / f"{split}.npy", np.asarray(tokens, dtype=np.int32))
            count = len(tokens)
        counts[split] = {
            "documents": len(rows),
            "tokens": count,
            "removed_exact_duplicates": removed[split],
        }
    manifest = {
        "format": 1,
        "kind": kind,
        "metadata": metadata,
        "splits": counts,
        "vocab_size": vocab_size,
        "tokenizer_size": tokenizer.get_vocab_size(),
        "tokenizer_training_split": "train",
        "eos_id": 1,
        "deduplication": "NFKC/casefold/whitespace exact; no near-duplicate guarantee",
        "files": {p.name: digest(p) for p in sorted(output.iterdir())},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def synthetic(recipe: DatasetConfig):
    counts = recipe.counts
    splits = {
        split: generate(recipe.seed + i * 100003, count, split == "ood")
        for i, (split, count) in enumerate(counts.items())
    }
    return prepare(
        recipe.output,
        splits,
        {"recipe": asdict(recipe), "scope": "controlled tasks"},
        recipe.vocab_size,
        "tasks",
    )


def articles(rows):
    """WikiText split rows are lines; keep complete articles together."""
    current = []
    for row in rows:
        text = row["text"]
        if re.match(r"^\s*= [^=].* =\s*$", text) and current:
            yield "".join(current)
            current = []
        current.append(text)
    if current:
        yield "".join(current)


def prepare_remote(recipe: DatasetConfig):
    output = in_dump(recipe.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset output already exists; choose a new version")
    cache = in_dump("dump/cache/huggingface")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    from datasets import load_dataset

    if not re.fullmatch(r"[0-9a-f]{40}", recipe.revision):
        raise ValueError("Remote datasets require a pinned 40-character commit")
    splits, total_chars = {}, 0
    for split, spec in recipe.splits.items():
        rows = load_dataset(
            recipe.repo,
            name=recipe.subset,
            revision=recipe.revision,
            split=spec.source,
            streaming=True,
        )
        documents = articles(rows) if recipe.unit == "article" else (row["text"] for row in rows)
        selected = []
        for i, text in enumerate(documents):
            if i < spec.skip:
                continue
            if len(selected) >= spec.limit:
                break
            if not text.strip():
                continue
            selected.append(text)
            total_chars += len(text)
            if total_chars > recipe.max_characters:
                raise ValueError("Preparation character budget exceeded; choose an explicit subset")
        splits[split] = selected
    return prepare(
        recipe.output, splits, {"recipe": asdict(recipe), "scope": recipe.scope}, recipe.vocab_size
    )
