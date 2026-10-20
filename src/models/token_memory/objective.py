"""Original assistant CE plus detached query/source auxiliary losses."""
from contextlib import contextmanager
from .reader import SourceFeatures, SpanReader, conserve
from .wordbits import unpack_values
from .schedule import rate
import torch
from torch.nn import functional as F

@contextmanager
def capture(model, source_terms):
    incoming = {}
    original = model.reader.read
    rr, cc = source_terms[:, :2].unbind(-1)

    def wrapped(h, source, counts, index):
        incoming[index] = (h[rr, cc].detach(), counts[rr, cc].detach())
        return original(h, source, counts, index)
    model.reader.read = wrapped
    try:
        yield incoming
    finally:
        model.reader.read = original

def validate(model, packet, terms):
    memory, mv, x, y, tv, visible = packet
    b, n = x.shape
    per = model.config.words_per_memory
    for kind, table in terms.items():
        if kind not in ('query', 'source') or table.ndim != 2 or table.shape[1] != 4 or (table.dtype != torch.long):
            raise ValueError('Invalid auxiliary table')
        if not len(table):
            continue
        rr, cc, a, d = table.unbind(-1)
        if bool(((rr < 0) | (rr >= b) | (cc < 0) | (cc >= n)).any()):
            raise ValueError('Auxiliary target outside packet')
        if bool((~tv[rr, cc] | (y[rr, cc] < 2)).any()):
            raise ValueError('Auxiliary target not an original assistant label')
        if len(torch.unique(rr * n + cc)) != len(rr):
            raise ValueError('Duplicate auxiliary target')
        absolute = visible[rr, cc // 64] * per + cc % 64
        if kind == 'query':
            if bool(((a >= absolute) | (d >= absolute) | (a < absolute - 128) | (d < absolute - 128)).any()):
                raise ValueError('Query slots outside causal128')
            if len(torch.unique(rr)) != len(rr):
                raise ValueError('Repeated query exposure in segment')
        else:
            ids = unpack_values(memory, per).reshape(b, -1)
            if bool(((a < 0) | (a >= ids.shape[1]) | (d < 0) | (d > 1)).any()):
                raise ValueError('Invalid critical source/field')
            if bool((a // per >= visible[rr, cc // 64]).any()) or bool((~mv[rr, a // per]).any()):
                raise ValueError('Future/padded critical source')
            if not torch.equal(ids[rr, a], y[rr, cc]):
                raise ValueError('Source word differs from original target')
            if len(torch.unique(rr * 2 + d)) != len(rr):
                raise ValueError('More than one critical target/field/segment')

def losses(model, packet, terms, query_coefficient=0.05, source_coefficient=0.02, check=True):
    if query_coefficient < 0 or source_coefficient < 0:
        raise ValueError('Negative auxiliary coefficient')
    if check:
        validate(model, packet, terms)
    qt, st = (terms['query'], terms['source'])
    qe, se = (query_coefficient > 0 and len(qt) > 0, source_coefficient > 0 and len(st) > 0)
    stats = dict(query_segments=len(qt), query_terms=2 * len(qt), source_segments=len(torch.unique(st[:, 0])), source_terms=len(st), critical_name=int((st[:, 3] == 0).sum()), critical_value=int((st[:, 3] == 1).sum()))
    if not qe and (not se):
        return (model.loss_sum(*packet), None, None, stats)
    memory, mv, x, y, tv, visible = packet
    per = model.config.words_per_memory
    with capture(model, st if se else st[:0]) as incoming:
        h, source, pointer, null, counts = model.hidden(memory, mv, x, tv, visible)
    rr, cc = (y != -100).nonzero(as_tuple=True)
    chosen = SourceFeatures(source.ids[rr], None, None, None, per, source.memory_positions)
    lp = model.reader.distribution(h[rr, cc][:, None], chosen, pointer[rr, cc][:, None], null[rr, cc][:, None], model.core.embedding.weight)
    ce = F.nll_loss(lp[:, 0], y[rr, cc], reduction='sum')
    requested = ([qt[:, 0]] if qe else []) + ([st[:, 0]] if se else [])
    rows = torch.unique(torch.cat(requested), sorted=True)
    inverse = torch.full((len(x),), -1, device=x.device, dtype=torch.long)
    inverse[rows] = torch.arange(len(rows), device=x.device)
    with torch.no_grad():
        aux_source = model.reader.literal_features(source.ids[rows].reshape(len(rows), source.memory_positions, per), mv[rows], model.core.embedding.weight.detach(), per)
    with model.reader.inputs(x[rows], tv[rows], torch.arange(x.shape[1], device=x.device), model.core.embedding.weight.detach()):
        query = model.reader.select_query(aux_source, counts[rows])
    qloss = sloss = None
    if qe:
        qr, qc, entity, topic = qt.unbind(-1)
        local = inverse[qr]
        absolute = counts[qr, qc] * per + qc % 64
        selected = torch.stack((entity, topic), -1) - absolute[:, None] + 128
        logits = query.logits[local, qc]
        if not bool(query.eligible[local, qc].gather(-1, selected).all()):
            raise ValueError('Gold query word not a complete causal candidate')
        qloss = F.cross_entropy(logits.reshape(-1, 128), selected.reshape(-1), reduction='sum') / 2
    if se:
        sr, sc, ss, field = st.unbind(-1)
        local = inverse[sr]
        with torch.no_grad():
            old_source = model.reader.literal_features(source.ids[sr].reshape(len(sr), source.memory_positions, per), mv[sr], model.core.embedding.weight.detach(), per)
        weights = torch.bincount(sr, minlength=len(x))[sr].to(h.dtype).reciprocal()
        layer_losses = []
        for index in (0, 1):
            before, available = incoming[index]
            with torch.no_grad():
                old_pointer, _ = SpanReader.probabilities(model.reader, before[:, None], old_source, available[:, None], index)
                normalized = model.reader.query_norm[index](before)
            bias = 8 * model.reader.entity_gate[index](normalized).tanh()[:, None] * query.match[local, sc][:, None]
            adjusted = conserve(old_pointer, old_source.ids, bias)[:, 0]
            probability = adjusted.gather(-1, ss[:, None])[:, 0]
            layer_losses.append((-probability.clamp_min(torch.finfo(probability.dtype).tiny).log() * weights).sum())
        sloss = sum(layer_losses) / 2
    return (ce, qloss, sloss, stats)

def update(model, optimizer, packets, step, total, target_count, query_count, source_count, lr=0.0003, query_coefficient=0.05, source_coefficient=0.02, do_step=True, check=True):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = rate(step, total, lr)
    values = dict(ce=0.0, query_loss=0.0, source_loss=0.0)
    totals = {}
    for packet, terms in packets:
        ce, query, source, stats = losses(model, packet, terms, query_coefficient, source_coefficient, check)
        objective = ce / target_count
        values['ce'] += float((ce / target_count).detach())
        for key, value, coefficient, count in (('query_loss', query, query_coefficient, query_count), ('source_loss', source, source_coefficient, source_count)):
            if value is not None:
                if count < 1:
                    raise ValueError('Missing effective auxiliary denominator')
                objective = objective + coefficient * value / count
                values[key] += float(value.detach()) / count
        objective.backward()
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
    if totals['query_segments'] != query_count or totals['source_segments'] != source_count:
        raise ValueError('Effective auxiliary exposure count differs')
    assert all((v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    if do_step:
        optimizer.step()
    return dict(**values, **totals, targets=target_count, lr=rate(step, total, lr), grad_norm=norm, objective=values['ce'] + query_coefficient * values['query_loss'] + source_coefficient * values['source_loss'], query_coefficient=query_coefficient, source_coefficient=source_coefficient)
