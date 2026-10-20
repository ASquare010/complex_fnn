"""Immutable baseline segments/labels, with one reusable FP32 code per full block."""
import json
from pathlib import Path

import numpy as np
import torch

from models.token_memory.runtime import digest


def load_encoder(device):
    from models.position_compressor.codec import Codec
    from .runtime import ENCODER, ENCODER_SHA
    if digest(ENCODER) != ENCODER_SHA:
        raise ValueError('Encoder checksum mismatch')
    codec = Codec.load_encoder(ENCODER, device=device, precision='fp32')
    codec.model.requires_grad_(False).eval()
    return codec


@torch.no_grad()
def encode_history(codec, histories):
    """Encode completed64-token blocks, then insert IDs in the existing low bits."""
    from .wordbits import pack_values
    if any(len(h) % 64 for h in histories):
        raise ValueError('Only completed64-token blocks enter memory')
    counts = [len(h)//64 for h in histories]
    memory = torch.zeros(len(histories), max(counts, default=0), 256, device=codec.device)
    valid = torch.zeros(memory.shape[:2], dtype=torch.bool, device=codec.device)
    for row, history in enumerate(histories):
        blocks = history.to(codec.device).reshape(-1, 64)
        for at in range(0, len(blocks), 256):
            tokens = blocks[at:at+256]
            codes, _ = codec.model(tokens, torch.ones_like(tokens, dtype=torch.bool))
            memory[row, at:at+len(tokens)] = pack_values(codes[:, 0].float(), tokens)
        valid[row, :len(blocks)] = True
    return memory, valid


class Dataset:
    def __init__(self, directory, split):
        self.directory, self.split = Path(directory), split
        if split not in ("train", "dev", "test"):
            raise ValueError("Unknown split")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["format"] not in ("chat-memory-v2", "chat-memory-v3"):
            raise ValueError("Expected baseline chat corpus")
        rows, docs, by_offset, sources = [], [], {}, []
        with (self.directory / f"{split}.segments.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                off, start, end = row["offset"], row["start"], row["end"]
                if off < 0 or start % 512 or not 0 < end-start <= 512 or start > 2048:
                    raise ValueError("Invalid original baseline segment")
                if row["source"] not in sources:
                    sources.append(row["source"])
                if off not in by_offset:
                    by_offset[off] = len(docs)
                    docs.append([off, 0])
                doc = by_offset[off]
                docs[doc][1] = max(docs[doc][1], (end-1)//64)
                rows.append((doc, start, end, row["targets"], sources.index(row["source"])))
        self.rows = np.asarray(rows, dtype=np.int64)
        self.docs = np.asarray(docs, dtype=np.int64)
        self.bases = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(self.docs[:, 1])))
        self.sources = sources
        self.tokens = np.memmap(self.directory / f"{split}.tokens.bin", dtype="<u2", mode="r")
        self.loss = np.memmap(self.directory / f"{split}.loss.bin", dtype="u1", mode="r")
        if len(self.tokens) != len(self.loss):
            raise ValueError("Token/mask sizes differ")
        if any(self.docs[d, 0]+end > len(self.tokens) for d, _, end, _, _ in self.rows):
            raise ValueError("Segment outside token file")

    def __len__(self):
        return len(self.rows)

    @property
    def code_count(self):
        return int(self.bases[-1])

    @property
    def cache_bytes(self):
        return self.code_count * 256 * 4

    def block_tokens(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        doc = np.searchsorted(self.bases[1:], indices, side="right")
        starts = self.docs[doc, 0] + (indices-self.bases[doc])*64
        return np.asarray(self.tokens[starts[:, None]+np.arange(64)], dtype=np.int64)

    def raw(self, index):
        doc, start, end, targets, source = map(int, self.rows[index])
        off = int(self.docs[doc, 0])
        ids = np.asarray(self.tokens[off+start:off+end], dtype=np.int64)
        mask = np.asarray(self.loss[off+start:off+end], dtype=bool)
        if int(mask.sum()) != targets:
            raise ValueError("Assistant labels changed")
        return doc, start, ids, mask, source

    def batch(self, indices, codes, device):
        items = [self.raw(i) for i in indices]
        n = max(len(t) for _, _, t, _, _ in items)
        m = max((start+len(t)-1)//64 for _, start, t, _, _ in items)
        b = len(items)
        memory = np.zeros((b, m, 256), dtype=np.float32)
        mv = np.zeros((b, m), dtype=bool)
        x = np.zeros((b, n), dtype=np.int64)
        y = np.full((b, n), -100, dtype=np.int64)
        tv = np.zeros((b, n), dtype=bool)
        visible = np.zeros((b, (n+63)//64), dtype=np.int64)
        for j, (doc, start, ids, mask, _) in enumerate(items):
            size = len(ids)
            count = (start+size-1)//64
            base = int(self.bases[doc])
            memory[j, :count] = codes[base:base+count]
            mv[j, :count] = True
            x[j, 0:size:64] = 1
            for a in range(0, size, 64):
                end = min(a+64, size)
                x[j, a+1:end] = ids[a:end-1]
                visible[j, a//64] = start//64+a//64
            y[j, :size] = np.where(mask, ids, -100)
            tv[j, :size] = True
        return tuple(torch.from_numpy(t).to(device) for t in (memory, mv, x, y, tv, visible))


def verify_data(directory, splits=("train", "dev")):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for split in splits:
        for suffix in ("tokens.bin", "loss.bin", "segments.jsonl"):
            name = f"{split}.{suffix}"
            if digest(directory / name) != manifest["hashes"][name]:
                raise ValueError(f"Data changed: {name}")
    if digest(directory / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    return digest(directory / "manifest.json")
