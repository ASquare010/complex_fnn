"""Independently verify released corpus files, masks, splits and memory labels."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np

from prepare_chat_corpus import ROOT, encode_dialogue, identity
from download_chat_sources import sha


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'dump/data/chat-memory-v2')
    args = parser.parse_args()
    from tokenizers import Tokenizer
    from models.token_memory.runtime import pin_cpu
    pin_cpu()
    path = args.data
    manifest = json.loads((path / 'manifest.json').read_text())
    serializer = encode_dialogue
    if manifest['format'] == 'chat-memory-v3':
        from chat_data.unified_format import encode_dialogue as serializer
    for name, digest in manifest['hashes'].items():
        assert sha(path / name) == digest, name
    tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
    counts, components, memory = Counter(), Counter(), defaultdict(Counter)
    modes, tool_roles = Counter(), Counter()
    groups, ids, answers = {}, set(), {}
    lengths, reply_lengths = [], []
    samples, checked = [], 0
    for split in ('train', 'dev', 'test'):
        tokens = np.memmap(path / f'{split}.tokens.bin', dtype='<u2', mode='r')
        mask = np.memmap(path / f'{split}.loss.bin', dtype='u1', mode='r')
        assert len(tokens) == len(mask)
        cursor = 0
        expected_segments = []
        source_sampled = set()
        with (path / f'{split}.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                if manifest['format'] == 'chat-memory-v3':
                    system = row['messages'][0]
                    assert system['role'] == 'system'
                    first = system['content'].splitlines()[0]
                    assert first in ('Reasoning: on.', 'Reasoning: off.')
                    mode = 'on' if first == 'Reasoning: on.' else 'off'
                    modes[mode] += 1
                    traces = [trace for msg in row['messages'] if msg['role']=='assistant'
                              for trace in re.findall(r'<think>(.*?)</think>',msg['content'],re.S) if trace.strip()]
                    assert bool(traces) == (mode == 'on'), row['id']
                    for msg in row['messages']:
                        if msg['role'] == 'tool':
                            tool_roles[split] += 1
                assert row['offset'] == cursor
                end = cursor + row['length']
                t, m = tokens[cursor:end], mask[cursor:end]
                assert len(t) == row['length'] and row['length'] <= 2560
                assert np.all((t >= 2) & (t < 4096)) and np.all(m <= 1)
                assert int(m.sum()) == row['targets']
                assert row['id'] not in ids
                ids.add(row['id'])
                assert groups.get(row['group'], split) == split
                groups[row['group']] = split
                counts[f'{split}/{row["source"]}/examples'] += 1
                counts[f'{split}/tokens'] += len(t)
                counts[f'{split}/targets'] += row['targets']
                lengths.append(len(t))
                components[row['origin'].get('component', row['source'])] += 1
                expected_mask = np.zeros(len(t), dtype='u1')
                for msg, span in zip(row['messages'], row['spans']):
                    if msg['role'] == 'assistant':
                        assert t[span['end'] - 1] == 2
                        expected_mask[span['body_start']:span['end']] = 1
                        reply_lengths.append(span['body_end'] - span['body_start'])
                        if len(msg['content']) >= 100:
                            h = identity(msg['content'])
                            assert answers.get(h, split) == split
                            answers[h] = split
                assert np.array_equal(m, expected_mask)
                for start in range(0, len(t), 512):
                    finish = min(start + 512, len(t))
                    n = int(m[start:finish].sum())
                    if n:
                        expected_segments.append(dict(id=row['id'], source=row['source'], offset=cursor,
                                                      start=start, end=finish, targets=n))
                if int(row['id'][:8], 16) % 500 == 0 or row['source'] not in source_sampled:
                    rt, rm, rs = serializer(row['messages'], tokenizer)
                    assert rt == t.tolist() and rm == m.tolist() and rs == row['spans']
                    checked += 1
                if row['source'] not in source_sampled:
                    samples.append(dict(split=split, source=row['source'], id=row['id'], messages=row['messages']))
                    source_sampled.add(row['source'])
                if 'memory' in row:
                    events = row['memory']
                    assert len(events['records']) == 3
                    assert events['fact_message_token_span'][1] <= (events['read_at_token'] // 512) * 512
                    assert events['read_at_token'] <= row['spans'][events['read']['before_message']]['body_start']
                    positive = events['read']['positive_record_ids']
                    assert bool(positive) != events['read']['no_match']
                    assert bool(events['read']['positive_chunk_indices']) == bool(positive)
                    for write, record in zip(events['writes'], events['records']):
                        assert write['record_id'] == record['id']
                        a, b = write['token_span']
                        decoded = tokenizer.decode(t[a:b].tolist())
                        assert record['name'] in decoded and record['value'] in decoded
                        assert b <= (events['read_at_token'] // 512) * 512
                    query = row['messages'][events['read']['query_message']]['content']
                    answer = row['messages'][events['read']['before_message']]['content']
                    for record in events['records']:
                        assert record['value'] not in query
                        if record['id'] in positive:
                            assert record['value'] in answer
                    category = 'missing' if events['read']['no_match'] else 'present'
                    memory[split][category] += 1
                    if category == 'missing':
                        assert all(r['name'] not in query for r in events['records'])
                cursor = end
        assert cursor == len(tokens)
        with (path / f'{split}.segments.jsonl').open() as f:
            actual_segments = [json.loads(line) for line in f]
        assert actual_segments == expected_segments
        assert sum(s['targets'] for s in actual_segments) == counts[f'{split}/targets']
        counts[f'{split}/segments'] = len(actual_segments)
        print(f'Audited {split}: {cursor:,} tokens, {len(actual_segments):,} training segments', flush=True)
    report = dict(state='passed', counts=counts, source_components=components, memory=memory,
                  reasoning_modes=modes, observed_tool_turns=tool_roles,
                  retokenized_dialogues=checked, unique_dialogues=len(ids), unique_split_groups=len(groups),
                  dialogue_token_quantiles=np.quantile(lengths, [0, .5, .9, .99, 1]).tolist(),
                  assistant_reply_token_quantiles=np.quantile(reply_lengths, [0, .5, .9, .99, 1]).tolist(),
                  checks=['all hashes', 'full token/mask spans', 'EOS', 'exact duplicate/group separation',
                          'long-answer split separation', 'every supervised token exactly once in segment index',
                          'memory causal boundary, missing labels and record chunk coverage', 'sampled retokenization'],
                  limitations=['No semantic near-duplicate guarantee or human quality certification.',
                               'Memory episodes are a controlled synthetic task, not arbitrary conversation or working DB.'])
    (path / 'audit.json').write_text(json.dumps(report, indent=2))
    (path / 'samples.json').write_text(json.dumps(samples, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
