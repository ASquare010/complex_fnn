"""Temporary dictionary, pooled context and query-conditioned copy reader."""
from .wordbits import unpack_values, word_ids
from .surface import metadata, geometry, recent, scope, QUERY_LIMIT
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.components import RMSNorm, parameter_generator
from contextlib import contextmanager
from models.components import parameter_generator
READER_PARAMETERS = 559616
CAUSAL_WORDS = 9

@dataclass
class SourceFeatures:
    ids: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    words_per_value: int
    memory_positions: int

class LexicalReader(nn.Module):

    def __init__(self, seed=47):
        super().__init__()
        self.source_norm = RMSNorm(256)
        self.value_norm = RMSNorm(256)
        self.key = nn.Linear(CAUSAL_WORDS * 256, 128, bias=False)
        self.key_up = nn.Linear(128, 256, bias=False)
        self.key_down = nn.Linear(256, 128, bias=False)
        self.query_norm = nn.ModuleList([RMSNorm(256) for _ in range(2)])
        self.query = nn.ModuleList([nn.Linear(256, 128, bias=False) for _ in range(2)])
        self.out = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(2)])
        self.gate = nn.ModuleList([nn.Linear(256, 1, bias=False) for _ in range(2)])
        self.recency = nn.ModuleList([nn.Linear(256, 1, bias=False) for _ in range(2)])
        self.mixture = nn.Linear(256, 2, bias=False)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.ndim > 1:
                    parameter.normal_(0, 0.02, generator=parameter_generator(seed, 'lexical.' + name))
            for gate in self.gate:
                gate.weight.zero_()
            for recency in self.recency:
                recency.weight.zero_()
            self.mixture.weight.zero_()
        if sum((p.numel() for p in self.parameters())) != READER_PARAMETERS:
            raise ValueError('Lexical reader budget mismatch')

    def literal_features(self, ids, valid, embedding, words_per_value=64):
        """Training/test reference also used after exact metadata extraction.

        Causal predecessor-only keys ensure a visible word never incorporates a
        future word, even when the batch already contains future code payloads.
        """
        if words_per_value not in (1, 64) or ids.ndim != 3 or ids.shape[-1] != words_per_value or (valid.shape != ids.shape[:2]):
            raise ValueError('Invalid lexical source shapes')
        b, m, _ = ids.shape
        flat = ids.reshape(b, m * words_per_value)
        enabled = valid[:, :, None].expand_as(ids).reshape_as(flat) & (flat > 1)
        return SourceFeatures(flat, self.source_norm(embedding), self.value_norm(embedding), enabled, words_per_value, m)

    def features(self, existing_values, valid, embedding, words_per_value=64):
        ids = word_ids(unpack_values(existing_values, words_per_value))
        return self.literal_features(ids, valid, embedding, words_per_value)

    def probabilities(self, h, source, visible, index):
        if index not in (0, 1) or visible.shape != h.shape[:2] or h.shape[0] != source.ids.shape[0]:
            raise ValueError('Invalid read query')
        q = self.query[index](self.query_norm[index](h))
        q = (q + self.key_down(F.gelu(self.key_up(q)))) / math.sqrt(128)
        effective = torch.einsum('bqf,fld->bqld', q, self.key.weight.reshape(128, CAUSAL_WORDS, 256))
        scores = h.new_zeros(*h.shape[:2], source.ids.shape[1])
        word_positions = torch.arange(source.ids.shape[1], device=h.device)
        for lag in range(CAUSAL_WORDS):
            vocab = F.linear(effective[:, :, lag], source.keys)
            neighbor = (word_positions - lag).clamp_min(0)
            ids = source.ids[:, neighbor]
            score = vocab.gather(-1, ids[:, None].expand(*h.shape[:2], source.ids.shape[1]))
            enabled = (word_positions >= lag)[None] & source.valid[:, neighbor]
            scores = scores + score.masked_fill(~enabled[:, None], 0.0)
        positions = torch.arange(source.ids.shape[1], device=h.device) // source.words_per_value
        age = (word_positions[None, None] - visible[:, :, None] * source.words_per_value).to(h.dtype) / 64
        scores = scores + self.recency[index](self.query_norm[index](h)) * age.clamp(-64, 0)
        allowed = (positions[None, None] < visible[:, :, None]) & source.valid[:, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        all_scores = torch.cat((scores, torch.zeros_like(scores[..., :1]) if scores.shape[-1] else torch.zeros(*h.shape[:2], 1, device=h.device, dtype=h.dtype)), -1)
        probs = all_scores.softmax(-1)
        return (probs[..., :-1], probs[..., -1:])

    def read(self, h, source, visible, index):
        probs, null = self.probabilities(h, source, visible, index)
        vocabulary = h.new_zeros(*h.shape[:2], source.values.shape[0]).scatter_add(-1, source.ids[:, None].expand_as(probs), probs)
        selected = F.linear(vocabulary, source.values.T)
        result = h + self.gate[index](self.query_norm[index](h)).tanh() * self.out[index](selected)
        return (result, probs, null)

    def distribution(self, h, source, probs, null, embedding):
        if probs.shape != (*h.shape[:2], source.ids.shape[1]) or null.shape != (*h.shape[:2], 1):
            raise ValueError('Invalid pointer distribution')
        gen = F.linear(h, embedding).log_softmax(-1)
        logits = self.mixture(h) + h.new_tensor([-4.0, 0.0])
        logmix = logits.log_softmax(-1)
        copy = torch.zeros_like(gen).scatter_add(-1, source.ids[:, None].expand_as(probs), probs)
        lognull = null.clamp_min(torch.finfo(null.dtype).tiny).log()
        genmass = torch.logaddexp(logmix[..., 1:2], logmix[..., 0:1] + lognull)
        return torch.logaddexp(gen + genmass, copy.clamp_min(torch.finfo(copy.dtype).tiny).log() + logmix[..., 0:1])

@dataclass
class Features(SourceFeatures):
    vocabulary_ids: torch.Tensor
    indices: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_valid: torch.Tensor
    positions: torch.Tensor
    word_positions: torch.Tensor

class DictionaryReader(LexicalReader):

    def literal_features(self, ids, valid, embedding, words_per_value=64):
        if words_per_value not in (1, 64) or ids.ndim != 3 or ids.shape[-1] != words_per_value or (valid.shape != ids.shape[:2]):
            raise ValueError('Invalid source shapes')
        b, m, _ = ids.shape
        flat = ids.reshape(b, m * words_per_value)
        vocabulary, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        inverse = inverse.reshape_as(flat)
        enabled = valid[:, :, None].expand_as(ids).reshape_as(flat) & (flat > 1)
        positions = torch.arange(flat.shape[1], device=ids.device)
        lags = torch.arange(CAUSAL_WORDS, device=ids.device)[:, None]
        neighbors = (positions[None] - lags).clamp_min(0)
        table = F.embedding(vocabulary, embedding)
        return Features(flat, self.source_norm(table), self.value_norm(table), enabled, words_per_value, m, vocabulary, inverse, inverse[:, neighbors], (positions[None] >= lags)[None] & enabled[:, neighbors], positions // words_per_value, positions)

    def probabilities(self, h, source, visible, index):
        if index not in (0, 1) or visible.shape != h.shape[:2] or h.shape[0] != source.ids.shape[0]:
            raise ValueError('Invalid query')
        normalized = self.query_norm[index](h)
        q = self.query[index](normalized)
        q = (q + self.key_down(F.gelu(self.key_up(q)))) / math.sqrt(128)
        effective = torch.einsum('bqf,fld->bqld', q, self.key.weight.reshape(128, CAUSAL_WORDS, 256))
        vocab = F.linear(effective, source.keys)
        indices = source.neighbor_indices[:, None].expand(h.shape[0], h.shape[1], CAUSAL_WORDS, source.ids.shape[1])
        scores = vocab.gather(-1, indices).masked_fill(~source.neighbor_valid[:, None], 0.0).sum(-2)
        age = (source.word_positions[None, None] - visible[:, :, None] * source.words_per_value).to(h.dtype) / 64
        scores = scores + self.recency[index](normalized) * age.clamp(-64, 0)
        allowed = (source.positions[None, None] < visible[:, :, None]) & source.valid[:, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        null = torch.zeros(*h.shape[:2], 1, device=h.device, dtype=h.dtype)
        probs = torch.cat((scores, null), -1).softmax(-1)
        return (probs[..., :-1], probs[..., -1:])

    def read(self, h, source, visible, index):
        probs, null = self.probabilities(h, source, visible, index)
        vocabulary = h.new_zeros(*h.shape[:2], source.vocabulary_ids.shape[0]).scatter_add(-1, source.indices[:, None].expand_as(probs), probs)
        selected = F.linear(vocabulary, source.values.T)
        return (h + self.gate[index](self.query_norm[index](h)).tanh() * self.out[index](selected), probs, null)
RANGES = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 12), (13, 20), (21, 32))

@dataclass
class SpanFeatures(Features):
    pool_start: torch.Tensor
    pool_end: torch.Tensor
    pool_count: torch.Tensor

class SpanReader(DictionaryReader):

    def __init__(self, seed=47, ranges=RANGES):
        super().__init__(seed)
        self.ranges = tuple((tuple(r) for r in ranges))
        if len(self.ranges) != CAUSAL_WORDS or any((not 0 <= lo <= hi for lo, hi in self.ranges)):
            raise ValueError('Nine causal ranges required')

    def literal_features(self, ids, valid, embedding, words_per_value=64):
        base = super().literal_features(ids, valid, embedding, words_per_value)
        n = base.ids.shape[1]
        offsets = torch.tensor(self.ranges, device=ids.device, dtype=torch.long)
        start = (base.word_positions[None] - offsets[:, 1, None]).clamp(0, n)
        end = (base.word_positions[None] - offsets[:, 0, None] + 1).clamp(0, n)
        counts = F.pad(base.valid.to(torch.long).cumsum(-1), (1, 0))
        count = (counts[:, end] - counts[:, start]).clamp_min(1)
        return SpanFeatures(**vars(base), pool_start=start, pool_end=end, pool_count=count)

    def probabilities(self, h, source, visible, index):
        if index not in (0, 1) or visible.shape != h.shape[:2] or h.shape[0] != source.ids.shape[0]:
            raise ValueError('Invalid query')
        normalized = self.query_norm[index](h)
        q = self.query[index](normalized)
        q = (q + self.key_down(F.gelu(self.key_up(q)))) / math.sqrt(128)
        effective = torch.einsum('bqf,fld->bqld', q, self.key.weight.reshape(128, CAUSAL_WORDS, 256))
        vocabulary = F.linear(effective, source.keys)
        indices = source.indices[:, None, None].expand(h.shape[0], h.shape[1], CAUSAL_WORDS, source.ids.shape[1])
        projected = vocabulary.gather(-1, indices)
        allowed = (source.positions[None, None] < visible[:, :, None]) & source.valid[:, None]
        prefix = F.pad(projected.masked_fill(~allowed[:, :, None], 0.0).cumsum(-1), (1, 0))
        shape = (h.shape[0], h.shape[1], CAUSAL_WORDS, source.ids.shape[1])
        start = source.pool_start[None, None].expand(shape)
        end = source.pool_end[None, None].expand(shape)
        scores = ((prefix.gather(-1, end) - prefix.gather(-1, start)) / source.pool_count[:, None].to(h.dtype)).sum(-2)
        age = (source.word_positions[None, None] - visible[:, :, None] * source.words_per_value).to(h.dtype) / 64
        scores = scores + self.recency[index](normalized) * age.clamp(-64, 0)
        scores = scores.masked_fill(~allowed, -torch.inf)
        null = torch.zeros(*h.shape[:2], 1, device=h.device, dtype=h.dtype)
        probability = torch.cat((scores, null), -1).softmax(-1)
        return (probability[..., :-1], probability[..., -1:])
EXTRA_PARAMETERS = 3072

@dataclass
class Query:
    logits: torch.Tensor
    selected: torch.Tensor
    confidence: torch.Tensor
    match: torch.Tensor
    presence: torch.Tensor
    positions: torch.Tensor
    eligible: torch.Tensor
    scope_valid: torch.Tensor

def conserve(probability, ids, bias):
    """Stable rescaling within non-EOS mass; no NULL/gen/EOS mass transfer."""
    ordinary = ids[:, None] != 2
    non_eos = probability.masked_fill(~ordinary, 0.0)
    extra = bias.expm1()
    mass = non_eos.sum(-1, keepdim=True)
    scale = 1 + (non_eos * extra).sum(-1, keepdim=True) / mass.clamp_min(torch.finfo(mass.dtype).tiny)
    adjusted = non_eos * (1 + extra) / scale
    return torch.where(ordinary, adjusted, probability)

class EntityReader(SpanReader):

    def __init__(self, seed=47, tokenizer_path=None):
        super().__init__(seed)
        word, boundary, capital, sentence, self.roles = metadata() if tokenizer_path is None else metadata(tokenizer_path)
        for name, value in [('word', word), ('boundary', boundary), ('capital', capital), ('sentence', sentence)]:
            self.register_buffer(name, value)
        self.selector = nn.Linear(4 * 256, 2, bias=False)
        self.entity_gate = nn.ModuleList([nn.Linear(256, 1, bias=False) for _ in range(2)])
        self.presence = nn.Parameter(torch.zeros(2, 256))
        with torch.no_grad():
            self.selector.weight.normal_(0, 0.02, generator=parameter_generator(seed, 'query_entity.selector'))
            for gate in self.entity_gate:
                gate.weight.zero_()
        self._inputs = None
        self._query = None
        self._embedding = None
        assert sum((p.numel() for p in self.selector.parameters())) + sum((p.numel() for p in self.entity_gate.parameters())) + self.presence.numel() == EXTRA_PARAMETERS

    def probabilities(self, h, source, visible, index):
        probability, null = super().probabilities(h, source, visible, index)
        if self._query is None:
            self._query = self.select_query(source, visible)
        bias = 8 * self.entity_gate[index](self.query_norm[index](h)).tanh() * self._query.match
        return (conserve(probability, source.ids, bias), null)

    def read(self, h, source, visible, index):
        result, probability, null = super().read(h, source, visible, index)
        return (result + self._query.presence[..., None] * self.presence[index], probability, null)

    @contextmanager
    def inputs(self, tokens, valid, query_indices, embedding):
        if self._inputs is not None:
            raise ValueError('Nested query reader context')
        object.__setattr__(self, '_inputs', (tokens, valid, query_indices))
        object.__setattr__(self, '_embedding', embedding)
        try:
            yield
        finally:
            object.__setattr__(self, '_inputs', None)
            object.__setattr__(self, '_query', None)
            object.__setattr__(self, '_embedding', None)

class Reader(EntityReader):

    def __init__(self, seed=47, tokenizer_path=None):
        super().__init__(seed, tokenizer_path)
        width = max(map(len, self.roles))
        self.register_buffer('role_table', torch.tensor([r + [-1] * (width - len(r)) for r in self.roles]))
        self.register_buffer('role_lengths', torch.tensor(list(map(len, self.roles))))
        if sum((p.numel() for p in self.parameters())) != 562688:
            raise ValueError('Local reader parameter budget differs')

    def features(self, values, valid, embedding, words_per_value=64):
        if values.dtype != torch.float32 or values.shape[-1] != 256:
            raise ValueError('256-wide FP32 V payload required')
        payload = unpack_values(values[..., :256], words_per_value)
        return self.literal_features(word_ids(payload), valid, embedding, words_per_value)

    def select_query(self, source, counts, *, detach_embedding=False):
        if self._inputs is None:
            raise ValueError('Query reader requires causal input context')
        tokens, valid, coordinates = self._inputs
        ids, available, positions, absolute = recent(source.ids, source.valid, tokens, valid, counts, coordinates, source.words_per_value)
        b, q, limit = ids.shape
        words = geometry(ids.reshape(-1, limit), available.reshape(-1, limit), self.word, self.boundary, self.sentence)
        body, start, scope_valid = scope(ids, available, self.role_table, self.role_lengths, len(self.roles[0]))
        eligible = words.starts.reshape(b, q, limit) & body
        eligible &= words.ends.reshape(b, q, limit) < limit
        embedding = self._embedding.detach() if detach_embedding else self._embedding
        table = F.rms_norm(embedding, (256,))
        projected = F.linear(table, self.selector.weight.reshape(8, 256)).T.reshape(2, 4, -1)
        logits = embedding.new_zeros(b, q, 2, limit)
        at = torch.arange(limit, device=ids.device)
        for lag in range(4):
            previous = (at - lag).clamp_min(0)
            score = projected[:, lag, ids[..., previous]].movedim(0, 2)
            permitted = (at >= lag)[None, None] & available[..., previous] & (previous[None, None] >= start[..., None])
            logits = logits + score.masked_fill(~permitted[:, :, None], 0.0)
        logits = logits + torch.stack((self.capital[ids].to(logits.dtype), torch.zeros_like(ids, dtype=logits.dtype)), 2)
        logits = logits.masked_fill(~eligible[:, :, None], -torch.inf)
        has_word = eligible.any(-1) & scope_valid
        safe = torch.where(has_word[:, :, None, None], logits, torch.zeros_like(logits))
        probability = safe.softmax(-1)
        selected = safe.argmax(-1)
        confidence = probability.gather(-1, selected[..., None])[..., 0] * has_word[:, :, None]
        old_words = geometry(source.ids, source.valid, self.word, self.boundary, self.sentence)
        keys = [v.reshape(b, q, limit).gather(-1, selected) for v in (words.lo, words.hi, words.length)]
        old_start = old_words.starts[:, None] & (old_words.ends[:, None] < counts[:, :, None] * source.words_per_value)
        current_begin = positions.gather(-1, start.clamp(0, limit - 1)[..., None])[..., 0]
        old_start &= torch.arange(source.ids.shape[1], device=ids.device)[None, None] < current_begin[..., None]
        hits = []
        for which in (0, 1):
            same = old_start
            for value, key in zip((old_words.lo, old_words.hi, old_words.length), keys):
                same = same & (value[:, None] == key[:, :, which, None])
            same &= has_word[..., None]
            groups = torch.zeros(b, q, source.ids.shape[1] + 1, device=ids.device, dtype=logits.dtype)
            sentence = old_words.sentence[:, None].expand(b, q, -1)
            groups.scatter_add_(-1, sentence, same.to(logits.dtype))
            hits.append(groups.clamp_max(1).gather(-1, sentence))
        allowed = source.valid[:, None] & (source.positions[None, None] < counts[:, :, None])
        match = hits[0] * confidence[:, :, 0, None] * (0.5 + 0.5 * hits[1] * confidence[:, :, 1, None])
        match = match.masked_fill(~allowed, 0.0)
        presence = match.amax(-1) if match.shape[-1] else confidence[..., 0] * 0
        return Query(logits, selected, confidence, match, presence, positions, eligible, scope_valid)
