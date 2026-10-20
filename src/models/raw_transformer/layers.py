import torch
from torch import nn
from torch.nn import functional as F
from models.components import RMSNorm, parameter_generator
from models.token_memory.transformer import Attention


class GELUFFN(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up=nn.Linear(width,hidden,bias=False)
        self.down=nn.Linear(hidden,width,bias=False)

    def forward(self,x):
        return self.down(F.gelu(self.up(x)))

    @torch.no_grad()
    def initialize(self,seed,prefix,residual_scale):
        for name,p in self.named_parameters():
            p.normal_(0,.02*(residual_scale if name=='down.weight' else 1),
                      generator=parameter_generator(seed,f'{prefix}.{name}'))


class PlainBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.attention_norm=RMSNorm(c.width)
        self.attention=Attention(c)
        self.ffn_norm=RMSNorm(c.width)
        self.ffn=GELUFFN(c.width,c.hidden)

    def forward(self,x,valid,positions):
        x=x+self.attention(self.attention_norm(x),valid,positions)
        return x+self.ffn(self.ffn_norm(x))
