"""Small immutable TinyStories cache and one batch policy for all architectures."""

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.core.reproducibility import sha256, write_json

REPO = "roneneldan/TinyStories"


def read_stories(url: str, count: int) -> list[str]:
    request = urllib.request.Request(url, headers={"User-Agent": "complex-fnn-research/0.1"})
    stories, lines = [], []
    with urllib.request.urlopen(request, timeout=90) as response:
        for line in io.TextIOWrapper(response, encoding="utf-8"):
            if "<|endoftext|>" in line:
                lines.append(line.split("<|endoftext|>")[0])
                text = "".join(lines).strip()
                if text:
                    stories.append(text)
                lines = []
                if len(stories) >= count:
                    break
            else:
                lines.append(line)
    if len(stories) < count:
        raise ValueError(f"Only received {len(stories)} of {count} requested stories")
    return stories


def normalized_hash(story: str) -> str:
    return hashlib.sha256(" ".join(story.split()).encode()).hexdigest()


def prepare(
    path: Path,
    train_count: int = 12000,
    validation_count: int = 1000,
    tokenizer_file: Path | None = None,
) -> dict:
    if min(train_count, validation_count) <= 0:
        raise ValueError("Story counts must be positive")
    if (path / "manifest.json").exists():
        manifest = load_manifest(path)
        if manifest["requested_stories"] != {"train": train_count, "valid": validation_count}:
            raise ValueError("Existing cache uses different story counts; select a fresh cache")
        if (
            tokenizer_file is not None
            and sha256(tokenizer_file) != manifest["files"]["tokenizer.json"]
        ):
            raise ValueError("Existing cache uses a different tokenizer")
        return manifest
    if path.exists() and any(path.iterdir()):
        raise ValueError("Partial cache exists; choose a fresh cache directory")
    revision = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # Frozen before experiments
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}"
    raw = {}
    for split, count in (("train", train_count), ("valid", validation_count)):
        print(f"Fetching {count} {split} stories at {revision}", flush=True)
        raw[split] = read_stories(f"{base}/TinyStoriesV2-GPT4-{split}.txt", count)
    seen: set[str] = set()
    data = {}
    for split in ("valid", "train"):
        data[split] = []
        for story in raw[split]:
            key = normalized_hash(story)
            if key not in seen:
                data[split].append(story)
                seen.add(key)
    if tokenizer_file is not None:
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    else:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=4096,
            special_tokens=["<unk>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(data["train"], trainer=trainer)
    if tokenizer.token_to_id("<eos>") is None or tokenizer.get_vocab_size() > 65536:
        raise ValueError("Tokenizer requires EOS and at most 65536 entries")
    path.mkdir(parents=True, exist_ok=True)
    if tokenizer_file is not None:
        (path / "tokenizer.json").write_bytes(tokenizer_file.read_bytes())
    else:
        tokenizer.save(str(path / "tokenizer.json"))
    counts = {}
    for split, stories in data.items():
        (path / f"{split}.json").write_text(
            json.dumps(stories, ensure_ascii=False), encoding="utf-8"
        )
        tokens = []
        for encoded in tokenizer.encode_batch(stories):
            tokens.extend(encoded.ids)
            tokens.append(tokenizer.token_to_id("<eos>"))
        np.save(path / f"{split}.npy", np.array(tokens, dtype=np.uint16))
        counts[split] = {
            "stories": len(stories),
            "tokens": len(tokens),
            "removed_duplicates": len(raw[split]) - len(stories),
        }
    manifest = {
        "dataset": REPO,
        "revision": revision,
        "source_template": base,
        "selection": "first N complete stories from each V2-GPT4 official split",
        "requested_stories": {"train": train_count, "valid": validation_count},
        "tokenizer": "train-only byte-level BPE",
        "vocab_size": tokenizer.get_vocab_size(),
        "splits": counts,
        "files": {p.name: sha256(p) for p in sorted(path.iterdir())},
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def load_manifest(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Cache hash mismatch: {name}")
    return manifest


class TokenData:
    def __init__(self, path: Path, device: str, seed: int) -> None:
        self.manifest = load_manifest(path)
        self.train = torch.from_numpy(np.load(path / "train.npy").astype("int64")).to(device)
        self.valid = torch.from_numpy(np.load(path / "valid.npy").astype("int64")).to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def batch(self, batch_size: int, context: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.train) <= context:
            raise ValueError("Training cache is shorter than context")
        starts = torch.randint(
            len(self.train) - context, (batch_size,), device=self.device, generator=self.generator
        )
        ids = starts[:, None] + torch.arange(context + 1, device=self.device)
        chunk = self.train[ids]
        return chunk[:, :-1], chunk[:, 1:]

    def validation(self, batch_size: int, context: int, batches: int):
        windows = min((len(self.valid) - 1) // context, batch_size * batches)
        if not windows:
            raise ValueError("Validation cache is shorter than context")
        for start in range(0, windows, batch_size):
            offsets = (
                torch.arange(start, min(start + batch_size, windows), device=self.device) * context
            )
            ids = offsets[:, None] + torch.arange(context + 1, device=self.device)
            chunk = self.valid[ids]
            yield chunk[:, :-1], chunk[:, 1:]
