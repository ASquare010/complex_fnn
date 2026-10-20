"""Native256 backbone and local reader, raw64; no memory width projection.

Frozen encoder codes stay256 at the input. Payloads are extracted BEFORE any
normalization, and reinserted into existing read-layer256-wide V slots.
The optional raw1 control has one256-word-embedding residual
and one ordinary global position per old word. Neither mode keeps an extra bank.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from .core import Config as CoreConfig, Core, PARAMETER_BUDGET, Attention
from .reader import Reader, SourceFeatures
from .wordbits import FORMAT, pack_values, unpack_values, word_ids

@dataclass
class Config:
    seed: int = 47
    words_per_memory: int = 64
    format: str = FORMAT
    read_layers: tuple = (2, 5)

class KVCache:

    def __init__(self, layers, batch, capacity, device, width=256):
        self.keys = [torch.zeros(batch, capacity + 64, width, device=device) for _ in range(layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.memory_valid = torch.zeros(batch, capacity, dtype=torch.bool, device=device)
        self.raw_valid = torch.zeros(batch, 64, dtype=torch.bool, device=device)
        self.raw_tokens = torch.zeros(batch, 64, dtype=torch.long, device=device)
        self.capacity, self.memory_count, self.raw_count = (capacity, 0, 0)

    def payload_values(self, layer):
        return self.values[layer][:, :self.memory_count]

    def allocated_bytes(self):
        tensors = self.keys + self.values + [self.memory_valid, self.raw_valid, self.raw_tokens]
        return sum((t.numel() * t.element_size() for t in tensors))

def insert_payload(values, payload):
    return pack_values(values, payload, validate=False)


class Model(nn.Module):


    def __init__(self, config=None, *, core_config=None, tokenizer_path=None):
        super().__init__()
        self.config = c = config or Config()
        self.core_config = core_config or replace(CoreConfig(), width=256, heads=4, layers=8, ffn_hidden_total=38062)
        cc = self.core_config
        if c.words_per_memory not in (1, 64) or c.format != FORMAT:
            raise ValueError('Require explicit1/64 word-bit source format')
        if cc.width != 256 or cc.context != 64 or cc.width % cc.heads or cc.width // cc.heads % 2:
            raise ValueError('Require256-wide/raw64 compatible core')
        if len(c.read_layers) != 2 or list(c.read_layers) != sorted(set(c.read_layers)) or c.read_layers[0] < 0 or (c.read_layers[-1] >= cc.layers):
            raise ValueError('Two distinct ordered read layers required')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed)
            self.core = Core(cc, c.seed)
            self.reader = Reader(c.seed, tokenizer_path)
        self.memory_gain = nn.Parameter(torch.full((256,), 0.02))
        self.memory_type = nn.Parameter(torch.zeros(256))

        if core_config is None and sum((v.numel() for v in self.parameters())) != PARAMETER_BUDGET:
            raise ValueError('Require exactly23,209,216 reader parameters')

    @staticmethod
    def _finish(block, x, q, k, v, allowed):
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed[:, None])
        x = x + block.attention.out(y.transpose(1, 2).reshape_as(x))
        return x + block.ffn(block.ffn_norm(x))

    def _qkv(self, block, x, positions):
        b, n, width = x.shape
        a = block.attention
        q, k, v = a.qkv(block.attention_norm(x)).reshape(b, n, 3, a.heads, width // a.heads).unbind(2)
        angle = positions[:, :, None, None] * a.frequency[None, None, None]
        cosine, sine = (angle.cos(), angle.sin())

        def rotate(t):
            left, right = (t[..., ::2], t[..., 1::2])
            return torch.stack((left * cosine - right * sine, left * sine + right * cosine), -1).flatten(-2)
        return (rotate(q), rotate(k), v)

    def _source_input(self, memory):
        if memory.dtype != torch.float32 or memory.shape[-1] != 256:
            raise ValueError('Original256-wide FP32 payload input required')
        payload = unpack_values(memory, self.config.words_per_memory)
        if self.config.words_per_memory == 1:
            words = self.core.embedding(word_ids(payload)[..., 0])
            x = words + F.rms_norm(words, (256,)) * self.memory_gain + self.memory_type
        else:
            x = F.rms_norm(memory, (256,)) * self.memory_gain + self.memory_type
        return (x, payload)

    def hidden(self, memory, memory_valid, tokens, token_valid, visible):
        b, m, d = memory.shape
        n = tokens.shape[1]
        if d != 256 or tokens.shape[0] != b or memory_valid.shape != (b, m) or (token_valid.shape != tokens.shape):
            raise ValueError('Invalid input shapes')
        if not 1 <= n <= 512 or visible.shape != (b, (n + 63) // 64):
            raise ValueError('Invalid raw64 packing')
        xmem, payload = self._source_input(memory)
        offset = torch.arange(n, device=tokens.device)
        window, within = (offset // 64, offset % 64)
        counts = visible[:, window]
        positions = torch.cat((torch.arange(m, device=tokens.device)[None].expand(b, m), counts + within[None]), 1)
        allowed = torch.zeros(b, m + n, m + n, dtype=torch.bool, device=tokens.device)
        allowed[:, :m, :m] = torch.ones(m, m, device=tokens.device, dtype=torch.bool).tril() & memory_valid[:, None]
        allowed[:, m:, :m] = (torch.arange(m, device=tokens.device)[None, None] < counts[:, :, None]) & memory_valid[:, None]
        allowed[:, m:, m:] = ((window[:, None] == window[None]) & (offset[:, None] >= offset[None]))[None] & token_valid[:, None]
        x = torch.cat((xmem, self.core.embedding(tokens)), 1)
        source = pointer = null = None
        coordinates = torch.arange(n, device=tokens.device)
        with self.reader.inputs(tokens, token_valid, coordinates, self.core.embedding.weight):
            for layer, block in enumerate(self.core.blocks):
                q, k, v = self._qkv(block, x, positions)
                if layer in self.config.read_layers:
                    packed = insert_payload(v[:, :m].flatten(-2), payload)
                    v = torch.cat((packed.reshape(b, m, block.attention.heads, self.core_config.width // block.attention.heads), v[:, m:]), 1)
                x = self._finish(block, x, q, k, v, allowed)
                if layer in self.config.read_layers:
                    if source is None:
                        source = self.reader.features(packed, memory_valid, self.core.embedding.weight, self.config.words_per_memory)
                    raw, pointer, null = self.reader.read(x[:, m:], source, counts, self.config.read_layers.index(layer))
                    x = torch.cat((x[:, :m], raw), 1)
        return (self.core.norm(x[:, m:]), source, pointer, null, counts)

    def packed(self, memory, memory_valid, tokens, token_valid, visible):
        h, source, pointer, null, _ = self.hidden(memory, memory_valid, tokens, token_valid, visible)
        return self.reader.distribution(h, source, pointer, null, self.core.embedding.weight)

    def forward(self, memory, memory_valid, tokens, token_valid=None, visible=None):
        if not 1 <= tokens.shape[1] <= 64:
            raise ValueError('Live raw input must fit64')
        if token_valid is None:
            token_valid = torch.ones_like(tokens, dtype=torch.bool)
        if visible is None:
            visible = memory_valid.sum(1, keepdim=True)
        return self.packed(memory, memory_valid, tokens, token_valid, visible)

    def loss_sum(self, memory, memory_valid, tokens, labels, token_valid, visible):
        h, source, pointer, null, _ = self.hidden(memory, memory_valid, tokens, token_valid, visible)
        rr, cc = (labels != -100).nonzero(as_tuple=True)
        selected = SourceFeatures(source.ids[rr], None, None, None, source.words_per_value, source.memory_positions)
        lp = self.reader.distribution(h[rr, cc][:, None], selected, pointer[rr, cc][:, None], null[rr, cc][:, None], self.core.embedding.weight)
        return F.nll_loss(lp[:, 0], labels[rr, cc], reduction='sum')

    def cache(self, batch, capacity):
        return KVCache(len(self.core.blocks), batch, capacity, self.core.embedding.weight.device, self.core_config.width)

    @torch.no_grad()
    def append_codes(self, cache, codes, valid=None):
        b, n, d = codes.shape
        start, end = (cache.memory_count, cache.memory_count + n)
        if codes.dtype != torch.float32 or d != 256 or cache.raw_count or (end > cache.capacity):
            raise ValueError('FP32 source/cache capacity/raw boundary error')
        if n == 0:
            return
        if valid is None:
            valid = torch.ones(b, n, dtype=torch.bool, device=codes.device)
        x, payload = self._source_input(codes)
        cache.memory_valid[:, start:end].copy_(valid)
        pos = torch.arange(start, end, device=codes.device)[None].expand(b, n)
        allowed = (torch.arange(end, device=codes.device)[None, None] <= pos[:, :, None]) & cache.memory_valid[:, None, :end]
        for layer, block in enumerate(self.core.blocks):
            q, k, v = self._qkv(block, x, pos)
            value = v.flatten(-2)
            if layer in self.config.read_layers:
                value = insert_payload(value, payload)
            cache.keys[layer][:, start:end].copy_(k.flatten(-2))
            cache.values[layer][:, start:end].copy_(value)
            k = cache.keys[layer][:, :end].reshape(b, end, block.attention.heads, -1)
            v = cache.values[layer][:, :end].reshape_as(k)
            x = self._finish(block, x, q, k, v, allowed)
        cache.memory_count = end

    @torch.no_grad()
    def feed_tokens(self, cache, tokens, valid=None):
        b, n = tokens.shape
        start, end = (cache.raw_count, cache.raw_count + n)
        m, c = (cache.memory_count, cache.capacity)
        if n < 1 or end > 64:
            raise ValueError('Raw window overflow')
        if valid is None:
            valid = torch.ones_like(tokens, dtype=torch.bool)
        cache.raw_tokens[:, start:end].copy_(tokens)
        cache.raw_valid[:, start:end].copy_(valid)
        pos = torch.arange(start, end, device=tokens.device)[None].expand(b, n) + m
        rawmask = torch.arange(end, device=tokens.device)[None, None] <= torch.arange(start, end, device=tokens.device)[None, :, None]
        allowed = torch.cat((cache.memory_valid[:, None, :m].expand(b, n, m), rawmask & cache.raw_valid[:, None, :end]), -1)
        x = self.core.embedding(tokens)
        source = pointer = null = None
        counts = torch.full((b, n), m, dtype=torch.long, device=tokens.device)
        coordinates = torch.arange(start, end, device=tokens.device)
        with self.reader.inputs(cache.raw_tokens, cache.raw_valid, coordinates, self.core.embedding.weight):
            for layer, block in enumerate(self.core.blocks):
                q, k, v = self._qkv(block, x, pos)
                cache.keys[layer][:, c + start:c + end].copy_(k.flatten(-2))
                cache.values[layer][:, c + start:c + end].copy_(v.flatten(-2))
                k = torch.cat((cache.keys[layer][:, :m], cache.keys[layer][:, c:c + end]), 1).reshape(b, m + end, block.attention.heads, -1)
                v = torch.cat((cache.values[layer][:, :m], cache.values[layer][:, c:c + end]), 1).reshape_as(k)
                x = self._finish(block, x, q, k, v, allowed)
                if layer in self.config.read_layers:
                    if source is None:
                        source = self.reader.features(cache.payload_values(layer), cache.memory_valid[:, :m], self.core.embedding.weight, self.config.words_per_memory)
                    x, pointer, null = self.reader.read(x, source, counts, self.config.read_layers.index(layer))
        cache.raw_count = end
        return (self.reader.distribution(self.core.norm(x), source, pointer, null, self.core.embedding.weight), torch.cat((pointer, null), -1))

    @torch.no_grad()
    def advance(self, cache, next_token, encode=None):
        if cache.raw_count == 0:
            raise ValueError('Feed BOS first')
        if cache.raw_count < 64:
            return self.feed_tokens(cache, next_token[:, None])
        count = 1 if self.config.words_per_memory == 64 else 64
        if cache.memory_count + count > cache.capacity:
            raise ValueError('No source cache capacity for rollover')
        block = torch.cat((cache.raw_tokens[:, 1:64], next_token[:, None]), 1)
        if self.config.words_per_memory == 64:
            if encode is None:
                raise ValueError('Compressed rollover requires frozen encoder')
            source = pack_values(encode(block)[:, None], block[:, None], validate=False)
        else:
            source = pack_values(torch.zeros(*block.shape, 256, device=block.device), block[:, :, None], validate=False)
        cache.raw_count = 0
        cache.raw_tokens.zero_()
        cache.raw_valid.zero_()
        for k, v in zip(cache.keys, cache.values):
            k[:, cache.capacity:].zero_()
            v[:, cache.capacity:].zero_()
        self.append_codes(cache, source)
        return self.feed_tokens(cache, torch.ones_like(next_token[:, None]))
