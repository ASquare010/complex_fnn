"""Exact word metadata in existing FP32 storage; semantic values change <=15ULP.

Each 16-bit payload uses four low mantissa bits in each of four FP32 values.
256 floats therefore store64 payloads without increasing vector/KV width.
Vocabulary IDs use12 bits, speaker roles2 and message-start1; bit15 is reserved.
Only input-known role/start metadata is permitted. Never encode future EOS/end.

This is attention/KV compression, not byte compression versus token-ID streams.
No decoded source-token bank is retained. Metadata must be read before any
normalization, arithmetic or dtype conversion; after a learned V projection,
reinsert it into that existing V slot. The future LM integration is not here.
"""
import torch

FORMAT='latent64-wordmeta-lsb4-fp32-v1'
VOCAB=4096
WIDTH=256
NIBBLE=15


def pack_metadata(ids,role,starts):
    if ids.shape != role.shape or ids.shape != starts.shape:raise ValueError('Metadata shapes differ')
    if ids.dtype not in (torch.int32,torch.int64,torch.uint16) or role.dtype not in (torch.int32,torch.int64,torch.uint16):raise ValueError('IDs and roles must be integer tensors')
    ids,role=ids.long(),role.long()
    if ((ids < 0)|(ids >= VOCAB)).any() or ((role < 0)|(role > 3)).any():raise ValueError('Invalid word/role')
    if ((starts != 0)&(starts != 1)).any():raise ValueError('Message starts must be binary')
    return ids.long() | (role.long()<<12) | (starts.long()<<14)


def word_ids(payload):return payload.long()&4095
def roles(payload):return (payload.long()>>12)&3
def message_starts(payload):return ((payload.long()>>14)&1).bool()


def _encode(values,payload):
    count=payload.shape[-1];bits=values.detach().contiguous().view(torch.int32)
    shifts=torch.arange(0,16,4,device=values.device,dtype=torch.long)
    digits=((payload.long()[...,None]>>shifts)&NIBBLE).flatten(-2).to(torch.int32)
    n=count*4
    prefix=(bits[...,:n]&~NIBBLE)|digits
    return torch.cat((prefix,bits[...,n:]),-1).view(torch.float32)


class _SemanticSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,payload):return _encode(values,payload)
    @staticmethod
    def backward(ctx,gradient):return gradient,None


def pack_values(values,payload,validate=True):
    """Pack64 words/code or one word/raw-reference V. Gradient to values is identity.

    A custom straight-through function avoids float arithmetic in the forward:
    arithmetic could flush payload-bearing subnormal values to zero. The gradient
    deliberately ignores this <=15ULP quantization, like ordinary float rounding.
    """
    if values.dtype != torch.float32 or values.shape[-1] != WIDTH:raise ValueError('Require256 FP32 values')
    if payload.dtype not in (torch.int32,torch.int64,torch.uint16):raise ValueError('Require integer payload')
    if payload.shape[:-1] != values.shape[:-1] or payload.shape[-1] not in (1,64):raise ValueError('Require1 or64 payloads/row')
    if validate:
        if not torch.isfinite(values).all():raise ValueError('Nonfinite semantic values')
        checked=payload.long()
        if ((checked < 0)|(checked > 65535)).any():raise ValueError('Payload exceeds16 bits')
    return _SemanticSTE.apply(values,payload)


def unpack_values(values,count=64):
    if values.dtype != torch.float32 or values.shape[-1] != WIDTH or count not in (1,64):raise ValueError('Require FP32 format and1/64 words')
    bits=values.contiguous().view(torch.int32)[...,:count*4].reshape(*values.shape[:-1],count,4)
    shifts=torch.arange(0,16,4,device=values.device,dtype=torch.long)
    return (((bits&NIBBLE).long()<<shifts).sum(-1))


def unpack_at(values,positions,validate=True):
    if values.dtype != torch.float32 or values.shape[-1] != WIDTH or positions.shape != values.shape[:-1]:raise ValueError('Invalid selected-word shapes/dtype')
    if validate and ((positions < 0)|(positions >= 64)).any():raise ValueError('Word position outside64')
    index=positions.long()[...,None]*4+torch.arange(4,device=values.device)
    selected=values.gather(-1,index).contiguous().view(torch.int32)
    shifts=torch.arange(0,16,4,device=values.device,dtype=torch.long)
    return (((selected&NIBBLE).long()<<shifts).sum(-1))
