import json
from pathlib import Path
import numpy as np
from models.token_memory.runtime import digest
class Dataset:

    def __init__(self, path, split):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'manifest.json').read_text())
        if self.meta['format'] != 'token-memory-v1':
            raise ValueError('Run token_memory/prepare_data.py first')
        self.length = self.meta['length']
        self.blocks = (self.length + 511) // 512
        count = self.meta['counts'][split]
        if (self.path / f'{split}.bin').stat().st_size != count * self.length * 2:
            raise ValueError('Token file size mismatch')
        self.rows = np.memmap(self.path / f'{split}.bin', dtype='<u2', mode='r', shape=(count, self.length))

    def __len__(self):
        return len(self.rows) * self.blocks

    def __getitem__(self, index):
        row, block = divmod(index, self.blocks)
        start = block * 512
        ids = torch.from_numpy(self.rows[row].astype(np.int64))
        return (ids[:start], ids[start:start + 512])
import torch


def batch(items,codec):
    histories,targets=zip(*items);device=codec.device
    m=max(map(len,histories));n=max(map(len,targets));b=len(items)
    history=torch.zeros(b,m,dtype=torch.long,device=device)
    valid=torch.zeros_like(history,dtype=torch.bool)
    x=torch.zeros(b,n,dtype=torch.long,device=device);y=torch.full_like(x,-100)
    for i,(h,t) in enumerate(items):
        history[i,:len(h)]=h.to(device);valid[i,:len(h)]=True
        x[i,0]=1;x[i,1:len(t)]=t[:-1].to(device);y[i,:len(t)]=t.to(device)
    return (history,valid,x,y!=-100),y
