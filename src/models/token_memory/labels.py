import json
from pathlib import Path
import numpy as np
import torch
from .runtime import ROOT,digest,write_json

DATA = ROOT / 'dump/native256-query-entity-labels-20261018'
CORPUS = ROOT / 'dump/data/unified-chat-v4'
FORMAT = 'native256-query-entity-original-labels-v1'
QUERY_COEFFICIENT = 0.05
SOURCE_COEFFICIENT = 0.02

def read(path):
    return json.loads(Path(path).read_text())

class Labels:

    def __init__(self, directory=DATA, verify=True):
        self.directory = Path(directory)
        self.manifest = read(self.directory / 'manifest.json')
        if self.manifest['format'] != FORMAT:
            raise ValueError('Wrong query-identity labels')
        if verify:
            for path, sha in self.manifest['inputs'].items():
                if digest(path) != sha:
                    raise ValueError(f'Changed original input:{path}')
            for name, sha in self.manifest['files'].items():
                if digest(self.directory / name) != sha:
                    raise ValueError(f'Changed labels:{name}')
        self.offsets = {}
        self.terms = {}
        for kind in ('query', 'source'):
            self.offsets[kind] = np.load(self.directory / f'{kind}-offsets.npy', mmap_mode='r')
            self.terms[kind] = np.load(self.directory / f'{kind}-terms.npy', mmap_mode='r')
            assert self.offsets[kind].shape == (780187,) and self.terms[kind].shape[1:] == (3,)
            assert self.offsets[kind][0] == 0 and self.offsets[kind][-1] == len(self.terms[kind])

    def batch(self, indices, device='cpu'):
        result = {}
        for kind in ('query', 'source'):
            pieces = []
            for row, index in enumerate(indices):
                a, b = map(int, self.offsets[kind][index:index + 2])
                values = np.asarray(self.terms[kind][a:b], dtype=np.int64)
                if len(values):
                    pieces.append(np.column_stack((np.full(len(values), row), values)))
            array = np.concatenate(pieces) if pieces else np.empty((0, 4), dtype=np.int64)
            result[kind] = torch.from_numpy(array).to(device)
        return result

    def counts(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        result = {}
        for kind in ('query', 'source'):
            length = self.offsets[kind][indices + 1] - self.offsets[kind][indices]
            result[kind] = dict(segments=int(np.count_nonzero(length)), terms=int(length.sum()))
        return result
