"""Fail-closed data loading against an independent, complete decoded identity.

Buffering alone did not resolve prior intermittent loading errors. Every worker
must verify every decoded byte and metadata field before any optimizer update.
There is no automatic retry and no reserved-test deserialization or scoring.
Historical loaders and checkpoints remain unchanged.
"""
import hashlib
import json
from io import BytesIO
from pathlib import Path

import torch
from tokenizers import Tokenizer

from storage import digest, read_json
from models.branch_sigmoid.data_identity import PROTOCOL, decoded_fingerprint


IDENTITY_FILE = Path(__file__).resolve().parents[2] / "config" / "branch_data_identity.json"


def load_verified_split(path, split, manifest):
    if split not in ("train", "valid"):
        raise ValueError("Only training and validation are authorized")
    identities = read_json(IDENTITY_FILE)
    if identities.get("protocol") != PROTOCOL or identities.get("passed") is not True:
        raise ValueError("Independent decoded-data reference has not passed")
    expected = identities["splits"][split]
    if expected["archive_sha256"] != manifest["split_sha256"][split]:
        raise ValueError(f"{split}: no independently verified archive identity")
    raw = (Path(path) / f"{split}.pt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["archive_sha256"]:
        raise ValueError(f"{split}: loaded archive bytes changed")
    rows = torch.load(BytesIO(raw), weights_only=True)
    del raw
    actual = decoded_fingerprint(rows)
    if (len(rows) != expected["rows"] or len(rows) != manifest["counts"][split]
            or actual != expected["decoded_sha256"]):
        raise ValueError(f"{split}: decoded data failed its independent identity check")
    print("VERIFIED_DATA " + json.dumps(dict(
        split=split, rows=len(rows), protocol=PROTOCOL,
        archive_sha256=expected["archive_sha256"], decoded_sha256=actual,
    )), flush=True)
    return rows


def load_data(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    for split, expected in manifest["split_sha256"].items():
        if digest(path / f"{split}.pt") != expected:
            raise ValueError("Prepared data changed")
    if digest(path / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    data = {split: load_verified_split(path, split, manifest) for split in ("train", "valid")}
    return (
        data,
        Tokenizer.from_file(str(path / "tokenizer.json")),
        manifest,
    )

def windows(rows, context=256):
    result = []
    for row in rows:
        for start in range(0, len(row["ids"]) - 1, context):
            if row["score"][start + 1 : start + context + 1].any():
                result.append((row, start // context))
    return result

def batch(examples, encoded, device):
    if encoded:
        raise ValueError('Branch Sigmoid uses tokens only; memory integration is retired')
    b = len(examples)
    x = torch.zeros(b, 256, dtype=torch.long)
    y = torch.full_like(x, -100)
    valid = torch.zeros(b, 256, dtype=torch.bool)
    for i, (row, j) in enumerate(examples):
        start = j * 256
        n = min(256, len(row["ids"]) - start - 1)
        x[i, :n] = row["ids"][start : start + n]
        target = row["ids"][start + 1 : start + n + 1].long().clone()
        target[~row["score"][start + 1 : start + n + 1]] = -100
        y[i, :n] = target
        if j:
            valid[i] = True
    return x.to(device), y.to(device), None, valid.to(device)
