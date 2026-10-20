"""Read immutable chat-memory-v2 segments without encoding the corpus up front."""
import json
import math
from pathlib import Path
import random

import numpy as np


class Dataset:
    def __init__(self, directory, split='train'):
        if split not in ('train', 'dev', 'test'):
            raise ValueError('Unknown split')
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest['format'] not in ('chat-memory-v2', 'chat-memory-v3'):
            raise ValueError('Expected chat-memory-v2 or chat-memory-v3')
        with (self.directory / f'{split}.segments.jsonl').open() as f:
            self.segments = [json.loads(line) for line in f]
        self.tokens = np.memmap(self.directory / f'{split}.tokens.bin', dtype='<u2', mode='r')
        self.loss = np.memmap(self.directory / f'{split}.loss.bin', dtype='u1', mode='r')
        if len(self.tokens) != len(self.loss):
            raise ValueError('Token/mask length mismatch')

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        import torch
        row = self.segments[index]
        offset, start, end = row['offset'], row['start'], row['end']
        if start % 512 or end - start > 512 or start > 2048:
            raise ValueError('Unsupported segment boundary')
        history = torch.tensor(self.tokens[offset:offset + start].astype('int64'))
        target = torch.tensor(self.tokens[offset + start:offset + end].astype('int64'))
        supervised = torch.tensor(self.loss[offset + start:offset + end].astype('bool'))
        return history, target, supervised


def batches_for_epoch(size, batch_size=32, seed=47, epoch=0):
    """Every segment exactly once per epoch, including the smaller final batch."""
    if size < 1 or batch_size < 1 or epoch < 0:
        raise ValueError('Invalid epoch settings')
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


def update_count(size, batch_size=32, epochs=2):
    if size < 1 or batch_size < 1 or epochs < 1:
        raise ValueError('Counts must be positive')
    return epochs * math.ceil(size / batch_size)


def batch(items, codec):
    """Keep user tokens visible to attention while excluding them from CE."""
    import torch
    from models.token_memory.data import encode_history
    history, targets, supervised = zip(*items)
    memory, memory_valid = encode_history(codec, history)
    count = max(map(len, targets))
    tokens = torch.zeros(len(items), count, dtype=torch.long, device=codec.device)
    labels = torch.full_like(tokens, -100)
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    for i, (target, mask) in enumerate(zip(targets, supervised)):
        target, mask = target.to(codec.device), mask.to(codec.device)
        length = len(target)
        tokens[i, 0] = 1
        tokens[i, 1:length] = target[:-1]
        labels[i, :length] = torch.where(mask, target, -100)
        valid[i, :length] = True
    return memory, memory_valid, tokens, labels, valid
