"""Independently verify v4 prefix preservation and actual per-source counts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from download_chat_sources import ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT/'dump/data/unified-chat-v3')
    p.add_argument('--data', type=Path, default=ROOT/'dump/data/unified-chat-v4')
    args = p.parse_args()
    manifest = json.loads((args.data/'manifest.json').read_text())
    base = json.loads((args.base/'manifest.json').read_text())
    audit = json.loads((args.data/'audit.json').read_text())
    assert audit['state'] == 'passed'
    assert manifest['parent_manifest_sha256'] == sha(args.base/'manifest.json')
    prefixes = {}
    additions, actual, observed = Counter(), defaultdict(Counter), Counter()
    reasoning_tools = Counter()
    for split in ('train','dev','test'):
        for suffix in ('tokens.bin','loss.bin','jsonl','segments.jsonl'):
            filename = f'{split}.{suffix}'
            remaining = (args.base/filename).stat().st_size
            h = hashlib.sha256()
            with (args.data/filename).open('rb') as f:
                while remaining:
                    block = f.read(min(4*1024**2, remaining))
                    assert block, filename
                    h.update(block)
                    remaining -= len(block)
            assert h.hexdigest() == base['hashes'][filename], filename
            prefixes[filename] = h.hexdigest()
        with (args.data/f'{split}.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                stat = actual[f'{split}/{row["source"]}']
                stat.update(examples=1, tokens=row['length'], targets=row['targets'],
                            multi_turn=int(sum(m['role']=='assistant' for m in row['messages'])>1))
                if row['origin'].get('extension') == 'v4':
                    additions[row['source']] += 1
                observed[row['source']] += sum(m['role']=='tool' for m in row['messages'])
                if row['source'] == 'reasoning_tools':
                    calls = [json.loads(t)['name'] for m in row['messages'] if m['role']=='assistant'
                             for t in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', m['content'], re.S)]
                    reasoning_tools['final_answer_only_dialogues'] += bool(calls) and set(calls)=={'final_answer'}
                    reasoning_tools['dialogues_with_observations'] += any(m['role']=='tool' for m in row['messages'])
                    reasoning_tools['web_search_calls'] += calls.count('web_search')
        with (args.data/f'{split}.segments.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                actual[f'{split}/{row["source"]}']['segments'] += 1
    assert dict(actual) == manifest['counts']
    assert +additions == +Counter(manifest['extension_additions'])
    totals = Counter()
    for key, stat in actual.items():
        if key.startswith('train/'):
            totals.update(stat)
    assert dict(totals) == manifest['train_totals']
    report = dict(state='passed', base_prefixes=prefixes, additions=dict(additions),
                  observed_tool_turns_by_source=observed, train_totals=totals,
                  reasoning_tool_profile=reasoning_tools,
                  train_target_percent={k.split('/')[1]:100*v['targets']/totals['targets']
                                        for k,v in actual.items() if k.startswith('train/')})
    (args.data/'extension_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
