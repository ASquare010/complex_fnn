"""Vectorized causal word geometry and public tokenizer metadata."""
from dataclasses import dataclass
from pathlib import Path
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
TOKENIZER = Path(__file__).resolve().parents[3] / 'dump/data/unified-chat-v4/tokenizer.json'
QUERY_LIMIT = 128
MAX_PIECES = 8

def metadata(path=TOKENIZER):
    tokenizer = Tokenizer.from_file(str(path))
    size = tokenizer.get_vocab_size()
    word = torch.zeros(size, dtype=torch.bool)
    boundary = torch.zeros_like(word)
    capital = torch.zeros_like(word)
    sentence = torch.zeros_like(word)
    for text, index in tokenizer.get_vocab().items():
        value = text.lstrip('ĠĊ')
        word[index] = bool(value) and value.isascii() and value.isalnum() and (index > 2)
        boundary[index] = text.startswith(('Ġ', 'Ċ'))
        capital[index] = bool(value) and value[0].isascii() and value[0].isupper()
        sentence[index] = index == 2 or any((c in value for c in '.!?')) or 'Ċ' in text
    roles = [tokenizer.encode('\n' + name + ':').ids for name in ('User', 'Assistant', 'System', 'Tool')]
    return (word, boundary, capital, sentence, roles)

@dataclass
class Words:
    starts: torch.Tensor
    ends: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    length: torch.Tensor
    sentence: torch.Tensor

def geometry(ids, valid, word, boundary, sentence):
    if ids.ndim != 2 or ids.shape != valid.shape:
        raise ValueError('Expected batch by literal positions')
    b, n = ids.shape
    if n == 0:
        empty = torch.zeros_like(ids)
        return Words(valid.clone(), empty, empty, empty, empty, empty)
    lexical = word[ids] & valid
    previous = F.pad(lexical[:, :-1], (1, 0))
    starts = lexical & (boundary[ids] | ~previous)
    starts[:, 0] &= boundary[ids[:, 0]]
    position = torch.arange(n, device=ids.device)[None].expand(b, n)
    head = torch.where(starts, position, -1).cummax(-1).values
    active = lexical & (head >= 0)
    index, lag = (head.clamp_min(0), position - head)
    enabled = active & (lag < MAX_PIECES)
    value = torch.where(enabled, ids, 0) << 12 * (lag % 4)
    lo = torch.zeros_like(ids).scatter_add_(-1, index, value.masked_fill(lag >= 4, 0))
    hi = torch.zeros_like(ids).scatter_add_(-1, index, value.masked_fill(lag < 4, 0))
    length = torch.zeros_like(ids).scatter_add_(-1, index, active.to(torch.long))
    after = (position + 1).clamp_max(n - 1)
    closes = active & (position + 1 < n) & valid.gather(-1, after) & (boundary[ids.gather(-1, after)] | ~word[ids.gather(-1, after)])
    ends = torch.zeros_like(ids).scatter_reduce_(-1, index, torch.where(closes, position + 1, 0), reduce='amax', include_self=True)
    ends = ends.masked_fill(length > MAX_PIECES, 0)
    starts &= ends > 0
    punctuation = (sentence[ids] & valid).to(torch.long)
    return Words(starts, ends, lo, hi, length.clamp_max(MAX_PIECES), punctuation.cumsum(-1) - punctuation)

def scope(ids, valid, table, lengths, user_length):
    size = ids.shape[-1]
    position = torch.arange(size, device=ids.device)
    lag = torch.arange(table.shape[-1], device=ids.device)
    at = position[:, None] + lag[None]
    take = at.clamp_max(size - 1)
    candidate = ids[..., take]
    available = valid[..., take] & (at < size)
    needed = lag[None] < lengths[:, None]
    hits = ((candidate[..., None, :] == table) & available[..., None, :] | ~needed).all(-1)
    user = torch.where(hits[..., 0], position, -1).amax(-1)
    start = user + user_length
    later = hits.any(-1) & (position > user[..., None])
    end = torch.where(later, position, size).amin(-1)
    body = (position >= start[..., None]) & (position < end[..., None]) & (user >= 0)[..., None] & valid
    return (body, start, user >= 0)

def recent(source_ids, source_valid, tokens, token_valid, counts, coordinates, per):
    if per not in (1, 64):
        raise ValueError('One or64 words per global memory position required')
    b, q = counts.shape
    coordinates = coordinates.to(device=tokens.device, dtype=torch.long)
    if coordinates.shape != (q,):
        raise ValueError('Missing raw query coordinates')
    base = counts * per
    absolute = base + (coordinates % 64)[None]
    positions = absolute[:, :, None] + torch.arange(-QUERY_LIMIT, 0, device=tokens.device)
    old = (positions >= 0) & (positions < base[:, :, None])
    n = source_ids.shape[1]
    if n:
        si = positions.clamp(0, n - 1)
        old_ids = source_ids.gather(1, si.reshape(b, -1)).reshape(b, q, QUERY_LIMIT)
        old_valid = source_valid.gather(1, si.reshape(b, -1)).reshape_as(old) & old & (positions < n)
    else:
        old_ids = torch.zeros_like(positions)
        old_valid = torch.zeros_like(old)
    ri = (coordinates // 64 * 64)[None, :, None] + positions - base[:, :, None] + 1
    raw = (positions >= base[:, :, None]) & (ri <= coordinates[None, :, None]) & (ri >= 0) & (ri < tokens.shape[1])
    take = ri.clamp(0, tokens.shape[1] - 1)
    raw_ids = tokens.gather(1, take.reshape(b, -1)).reshape_as(positions)
    raw_valid = token_valid.gather(1, take.reshape(b, -1)).reshape_as(raw) & raw
    ids = torch.where(old, old_ids, raw_ids)
    valid = old_valid | raw_valid
    return (ids.masked_fill(~valid, 0), valid, positions, absolute)
