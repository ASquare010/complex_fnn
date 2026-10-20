"""Original two-epoch data; frozen cached semantic codes and labels are read only."""
import json,random,hashlib
from pathlib import Path
import numpy as np
import torch
from .data import Dataset,verify_data
from .runtime import ROOT,ENCODER_SHA,digest
from .wordbits import pack_values,unpack_values,word_ids
CORPUS=ROOT/'dump/data/unified-chat-v4'
CACHE=ROOT/'dump/token-memory-64-cache-v4'
ARMS=('compressed64','raw1')
def read(path):return json.loads(Path(path).read_text())
def cache_identity(ds):
    return dict(format='token-memory-64-cache-v1',split=ds.split,dtype='float32',shape=[ds.code_count,256],encoder_sha256=ENCODER_SHA,manifest_sha256=digest(ds.directory/'manifest.json'),layout_sha256=hashlib.sha256(ds.docs.tobytes()+ds.bases.tobytes()).hexdigest(),preparation_sources={'data.py': '126c1e6f4179242ff0a03d4a45c58efcfb1d8ff2e3bbb5ca332604cc07ab08c6', 'cache.py': '75764c20bfc94d5b7a013c13c6fda5d4be0aa8692c297fb1badfa1069ec578a4'})
class PackedCodes:
    """RO original vectors/source IDs; new packed representation only in RAM."""
    def __init__(self,ds,cache):self.ds,self.cache=ds,cache
    def __getitem__(self,sl):
        if not isinstance(sl,slice) or sl.step not in (None,1) or sl.start is None or sl.stop is None:raise ValueError('Contiguous code slice required')
        values=np.array(self.cache[sl],copy=True);ids=self.ds.block_tokens(np.arange(sl.start,sl.stop));bits=values.view(np.uint32)
        digits=((ids[...,None]>>np.array([0,4,8,12],np.int64))&15).reshape(len(ids),256).astype(np.uint32)
        bits[:]=(bits&np.uint32(0xfffffff0))|digits;return values


class Data:
    def __init__(self,corpus=CORPUS,cache=CACHE,verify=True):
        self.corpus,self.cache_dir=Path(corpus),Path(cache);self.datasets={s:Dataset(corpus,s) for s in ('train','dev')};self.codes={};self.original={}
        if verify:verify_data(corpus)
        for split,ds in self.datasets.items():
            meta=read(Path(cache)/f'{split}.json');path=Path(cache)/f'{split}.f32'
            assert meta['identity'] == cache_identity(ds) and path.stat().st_size == ds.cache_bytes
            if verify:assert digest(path) == meta['sha256']
            self.original[split]=np.memmap(path,mode='r',dtype='<f4',shape=(ds.code_count,256));self.codes[split]=PackedCodes(ds,self.original[split])
        self.per_epoch=(len(self.datasets['train'])+31)//32;self.total=2*self.per_epoch
        assert len(self.datasets['train']) == 780186 and self.total == 48762 and int(self.datasets['train'].rows[:,3].sum()) == 180678552

    def order(self,epoch):
        result=list(range(len(self.datasets['train'])));random.Random(47+epoch).shuffle(result);return result

    def batch(self,split,indices,arm,device):
        if arm not in ARMS:raise ValueError('Unknown arm')
        memory,mv,x,y,tv,visible=self.datasets[split].batch(indices,self.codes[split],device)
        if arm == 'raw1':
            ids=word_ids(unpack_values(memory));b,m,_=memory.shape
            memory=pack_values(torch.zeros(b,m*64,256,device=device),ids.reshape(b,m*64,1),validate=False)
            mv=mv.repeat_interleave(64,1);visible=visible*64
        return memory,mv,x,y,tv,visible
