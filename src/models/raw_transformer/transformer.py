from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from models.components import RMSNorm, initialize
from models.raw_transformer.layers import PlainBlock


@dataclass
class Config:
    width:int=512
    heads:int=8
    layers:int=8
    hidden:int=1536
    vocab_size:int=4096
    context:int=2560
    rope_base:float=10000.


class Model(nn.Module):
    def __init__(self,config=None,seed=47):
        super().__init__();self.config=c=config or Config()
        self.embedding=nn.Embedding(c.vocab_size,c.width)
        self.blocks=nn.ModuleList([PlainBlock(c) for _ in range(c.layers)])
        self.norm=RMSNorm(c.width)
        initialize(self,seed)

    def forward(self,history,history_valid,tokens,token_valid):
        m=history.shape[1]
        if m+tokens.shape[1]>self.config.context:raise ValueError('Raw context exceeds capacity')
        valid=torch.cat((history_valid,token_valid),1)
        positions=torch.cat((torch.arange(m,device=tokens.device).expand(len(tokens),m),
            history_valid.sum(1)[:,None]+torch.arange(tokens.shape[1],device=tokens.device)),1)
        x=self.embedding(torch.cat((history,tokens),1))
        for block in self.blocks:
            x=block(x,valid,positions)
        return F.linear(self.norm(x[:,m:]),self.embedding.weight)
