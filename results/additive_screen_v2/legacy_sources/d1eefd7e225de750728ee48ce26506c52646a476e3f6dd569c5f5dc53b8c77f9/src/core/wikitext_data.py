"""Pinned WikiText-2 raw cache with preserved article text and train-only BPE."""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.core.data import load_manifest, normalized_hash
from src.core.reproducibility import provenance, sha256, write_json

REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"
BASE = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}"
SOURCES = {
    "train": ("train", "49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701"),
    "valid": ("validation", "a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f"),
}


def articles_from_rows(rows):
    """Top-level '= Title =' starts an article; '= = Section = =' does not."""
    articles, current = [], []
    started = False
    for row in rows:
        if not isinstance(row, str):
            raise ValueError("WikiText rows must be strings")
        heading = re.fullmatch(r"= [^=].* =", row.strip()) is not None
        if heading:
            if started:
                articles.append("".join(current))
                current = []
            started = True
        elif not started and row.strip():
            raise ValueError("Nonblank content before the first article heading")
        current.append(row)
    if not started:
        raise ValueError("No top-level article headings")
    articles.append("".join(current))
    assert "".join(articles) == "".join(rows)
    return articles


def deduplicate(raw):
    """Validation takes precedence; retain original article text and order."""
    seen, result, removed = set(), {}, {}
    for split in ("valid", "train"):
        result[split], removed[split] = [], []
        for index, article in enumerate(raw[split]):
            key = normalized_hash(article)
            if key in seen:
                removed[split].append({"article_index": index, "normalized_sha256": key})
            else:
                seen.add(key)
                result[split].append(article)
    return result, removed


def prepare(path: Path, source: Path, tokenizer_file: Path | None = None):
    import pyarrow
    import pyarrow.parquet as pq

    if (path / "manifest.json").exists():
        manifest = load_manifest(path)
        if manifest["dataset"] != "Salesforce/wikitext" or manifest["revision"] != REVISION:
            raise ValueError("Existing cache uses another dataset revision")
        if (
            tokenizer_file is not None
            and sha256(tokenizer_file) != manifest["files"]["tokenizer.json"]
        ):
            raise ValueError("Existing cache uses another tokenizer")
        return manifest
    if path.exists() and any(path.iterdir()):
        raise ValueError("Partial cache exists; preserve it and choose a fresh output")
    source.mkdir(parents=True, exist_ok=True)
    raw, source_info = {}, {}
    for split, (official, expected) in SOURCES.items():
        file = source / f"{official}.parquet"
        url = f"{BASE}/wikitext-2-raw-v1/{official}-00000-of-00001.parquet"
        if not file.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                blob = response.read(16 * 1024 * 1024 + 1)
            if len(blob) > 16 * 1024 * 1024:
                raise ValueError("Source exceeds frozen download bound")
            file.write_bytes(blob)
        if sha256(file) != expected:
            raise ValueError(f"Pinned source hash mismatch: {file}")
        rows = pq.read_table(file, columns=["text"])["text"].to_pylist()
        raw[split] = articles_from_rows(rows)
        source_info[split] = {
            "url": url,
            "sha256": expected,
            "rows": len(rows),
            "articles": len(raw[split]),
            "characters": sum(map(len, rows)),
            "text_preserved_exactly": True,
        }
    data, removed = deduplicate(raw)
    if tokenizer_file is None:
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
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    assert tokenizer.get_vocab_size() == 4096 and tokenizer.token_to_id("<eos>") == 1
    path.mkdir(parents=True, exist_ok=True)
    if tokenizer_file is None:
        tokenizer.save(str(path / "tokenizer.json"))
    else:
        (path / "tokenizer.json").write_bytes(tokenizer_file.read_bytes())
    counts = {}
    for split in ("train", "valid"):
        articles = data[split]
        (path / f"{split}.json").write_text(
            json.dumps(articles, ensure_ascii=False), encoding="utf-8"
        )
        tokens, offsets = [], []
        for article, encoded in zip(articles, tokenizer.encode_batch(articles), strict=True):
            start = len(tokens)
            tokens.extend(encoded.ids)
            tokens.append(1)
            offsets.append(
                {"start": start, "stop": len(tokens), "normalized_sha256": normalized_hash(article)}
            )
        array = np.asarray(tokens, dtype=np.uint16)
        np.save(path / f"{split}.npy", array)
        write_json(path / f"{split}_articles.json", offsets)
        counts[split] = {
            "articles": len(articles),
            "tokens": len(tokens),
            "removed_duplicates": len(removed[split]),
            "unknown_token_count": int((array == 0).sum()),
            "complete_context128_windows": (len(tokens) - 1) // 128,
        }
    write_json(path / "deduplication.json", removed)
    manifest = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "revision": REVISION,
        "source_files": source_info,
        "license_metadata": ["CC BY-SA 3.0", "GFDL"],
        "selection": "All official train and validation articles, normalized exact article deduplication with validation priority; original row text preserved, one EOS per article; test not fetched or scored.",
        "tokenizer": "train-only byte-level BPE",
        "vocab_size": 4096,
        "splits": counts,
        "pyarrow_version": pyarrow.__version__,
        "files": {p.name: sha256(p) for p in sorted(path.iterdir())},
    }
    write_json(path / "manifest.json", manifest)
    write_json(path / "preparation_provenance.json", provenance())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/wikitext2_v1"))
    parser.add_argument("--source", type=Path, default=Path("data/wikitext2_raw_source_v1"))
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.source, args.tokenizer), indent=2))
