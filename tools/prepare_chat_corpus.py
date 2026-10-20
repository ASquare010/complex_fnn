"""Prepare chat-format, assistant-target data for both encoder-memory arms.

CPU preprocessing only. No encoder pass, optimizer, model creation or training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from models.token_memory.runtime import pin_cpu
from download_chat_sources import CACHE, sha
from chat_data.format import encode_dialogue

ROLES = {'system': 'System', 'user': 'User', 'assistant': 'Assistant'}
STORY_PROMPTS = ('Tell me a short, simple story.', 'Write a short story in simple English.',
                 'Please tell me a little story.', 'Tell a simple story with a beginning and an ending.')
QUESTIONS = {
    'train': ('Where does {name} keep the notebook?', 'Remind me where {name} stores the notebook.',
              'What is the notebook location for {name}?', 'Tell me where to find {name}\'s notebook.'),
    'dev': ('Which place did {name} choose for the notebook?', 'Recall the notebook storage place belonging to {name}.'),
    'test': ('I need {name}\'s notebook. Where should I look?', 'According to the earlier note, where is {name}\'s notebook?'),
}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).lower().split())


def identity(text):
    return hashlib.sha256(normalized(text).encode('utf-8')).hexdigest()


def split_for(group):
    bucket = int(group[:8], 16) % 100
    return 'test' if bucket == 0 else 'dev' if bucket == 1 else 'train'


def clean_messages(messages):
    """Keep full dialogues, or reject them. Never silently truncate a turn."""
    result = []
    expected = 'user'
    for item in messages:
        role, text = item.get('role'), item.get('content')
        if role not in ROLES or not isinstance(text, str) or not text.strip():
            raise ValueError('invalid_message')
        text = text.strip()
        if '\x00' in text or '<|endoftext|>' in text:
            raise ValueError('control_text')
        if role == 'system' and not result:
            pass
        elif role != expected:
            raise ValueError('role_order')
        else:
            expected = 'assistant' if role == 'user' else 'user'
        result.append(dict(role=role, content=text))
    if not result or result[-1]['role'] != 'assistant':
        raise ValueError('unfinished_dialogue')
    return result



def memory_episode(index, split):
    """Ground-truth-labelled synthetic episodes; not model-distilled conversations."""
    rng = random.Random(4700000 + index)
    pools = {
        'train': ['Amina', 'Bilal', 'Clara', 'Dara', 'Evan', 'Farah', 'Gina', 'Haris', 'Iris', 'Jamal', 'Kira', 'Luis'],
        'dev': ['Mona', 'Nabil', 'Orla', 'Pavel', 'Quinn', 'Remy'],
        'test': ['Rina', 'Sami', 'Tara', 'Uma', 'Vera', 'Wali'],
    }
    names = rng.sample(pools[split], 4)
    colors = ['blue', 'green', 'red', 'yellow', 'purple', 'white', 'orange', 'black']
    containers = ['box', 'basket', 'drawer', 'bag', 'cabinet', 'cupboard']
    rooms = ['kitchen', 'bedroom', 'office', 'hallway', 'workshop', 'study']
    values = [f'in the {rng.choice(colors)} {rng.choice(containers)} in the {rng.choice(rooms)}' for _ in names]
    # Exactly three records in both answerable and missing cases. The fourth
    # sampled name is absent this episode, but present in other episodes.
    records = [dict(id=f'{index}:{i}', name=n, value=v) for i, (n, v) in enumerate(zip(names[:3], values[:3]))]
    facts = 'Please remember these notes. ' + ' '.join(f'{r["name"]} keeps a notebook {r["value"]}.' for r in records)
    # Unrelated user context, long enough that the queried note leaves active tokens.
    weather = ['sunny', 'cloudy', 'windy', 'cool', 'warm', 'quiet']
    actions = ['water the plants', 'sweep the path', 'wash the cups', 'open the windows', 'fold the towels', 'watch the birds']
    filler = 'Here is unrelated background about a week at a guest house. '
    for day in range(32):
        filler += f'On day {day + 1}, it was {rng.choice(weather)}. We decided to {rng.choice(actions)} and then {rng.choice(actions)}. '
    missing = index % 5 == 0
    query = rng.randrange(3)
    name = names[3] if missing else names[query]
    question = rng.choice(QUESTIONS[split]).format(name=name)
    answer = "I don't have a saved location for that notebook." if missing else f'{name} keeps the notebook {values[query]}.'
    messages = [dict(role='user', content=facts), dict(role='assistant', content='I have noted those locations.'),
                dict(role='user', content=filler), dict(role='assistant', content='All right.'),
                dict(role='user', content=question), dict(role='assistant', content=answer)]
    events = dict(records=records, writes=[dict(message=0, record_id=r['id']) for r in records],
                  read=dict(before_message=5, query_message=4,
                            positive_record_ids=[] if missing else [records[query]['id']], no_match=missing),
                  no_read_before_messages=[1, 3], session_reset_before_message=4)
    return messages, events


class Writer:
    def __init__(self, path, tokenizer, max_tokens, protected, cleaner=clean_messages, serializer=encode_dialogue):
        self.path, self.tokenizer, self.max_tokens = path, tokenizer, max_tokens
        self.protected = protected
        self.cleaner, self.serializer = cleaner, serializer
        self.seen, self.answers, self.groups = set(), {}, {}
        self.rejects, self.stats = Counter(), defaultdict(Counter)
        self.handles = {}
        for split in ('train', 'dev', 'test'):
            self.handles[split] = {name: (path / f'{split}.{name}').open(mode)
                                   for name, mode in [('tokens.bin', 'wb'), ('loss.bin', 'wb'), ('jsonl', 'w'), ('segments.jsonl', 'w')]}

    def add(self, messages, source, origin, group=None, split=None, events=None):
        import numpy as np
        try:
            messages = self.cleaner(messages)
        except ValueError as error:
            self.rejects[str(error)] += 1
            return False
        user_texts = [m['content'] for m in messages if m['role'] == 'user']
        # A generic greeting is not a useful split identity for a whole dialogue.
        first = next((s for s in user_texts if len(s) >= 40), '\n'.join(user_texts))
        group = group or identity(first)
        split = split or split_for(group)
        if group in self.protected:
            self.rejects['prior_experiment_overlap'] += 1
            return False
        if group in self.groups and self.groups[group] != split:
            self.rejects['group_split_conflict'] += 1
            return False
        fingerprint = identity(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        # Long identical answers copied into different prompts cannot cross splits.
        answer_ids = [identity(m['content']) for m in messages if m['role'] == 'assistant' and len(m['content']) >= 100]
        if fingerprint in self.seen or any(a in self.answers and self.answers[a] != split for a in answer_ids):
            self.rejects['duplicate_or_cross_split_answer'] += 1
            return False
        tokens, mask, spans = self.serializer(messages, self.tokenizer)
        if source in ('chat', 'everyday') and any(s['role'] == 'assistant' and s['body_end'] - s['body_start'] > 384 for s in spans):
            self.rejects['long_assistant_reply'] += 1
            return False
        if len(tokens) > self.max_tokens or len(tokens) < 8:
            self.rejects['length'] += 1
            return False
        if not all(3 <= t < 4096 or t == 2 for t in tokens):
            self.rejects['special_or_unknown_token'] += 1
            return False
        if events:
            fact_message = events['writes'][0]['message']
            body_encoding = self.tokenizer.encode(' ' + messages[fact_message]['content'])
            header_length = spans[fact_message]['body_start']
            for record, write in zip(events['records'], events['writes']):
                phrase = f'{record["name"]} keeps a notebook {record["value"]}.'
                begin = (' ' + messages[fact_message]['content']).index(phrase)
                end = begin + len(phrase)
                indices = [j + header_length for j, (a, b) in enumerate(body_encoding.offsets) if a < end and b > begin]
                write['token_span'] = [min(indices), max(indices) + 1]
                write['chunk_indices'] = list(range(min(indices) // 64, max(indices) // 64 + 1))
            positive = set(events['read']['positive_record_ids'])
            events['read']['positive_chunk_indices'] = sorted({c for w in events['writes'] if w['record_id'] in positive for c in w['chunk_indices']})
            read = spans[events['read']['before_message']]
            fact_end = spans[fact_message]['body_end']
            boundary = (read['body_start'] // 512) * 512
            if fact_end > boundary:
                raise ValueError('Memory fact did not leave the active block')
            events['fact_message_token_span'] = [spans[fact_message]['body_start'], fact_end]
            events['read_at_token'] = read['body_start']
            events['distance_from_fact_end'] = read['body_start'] - fact_end
            query_span = spans[events['read']['query_message']]
            events['query_message_token_span'] = [query_span['body_start'], query_span['body_end']]
        handles = self.handles[split]
        offset = handles['tokens.bin'].tell() // 2
        np.asarray(tokens, dtype='<u2').tofile(handles['tokens.bin'])
        np.asarray(mask, dtype='u1').tofile(handles['loss.bin'])
        row = dict(id=fingerprint, group=group, source=source, origin=origin, offset=offset,
                   length=len(tokens), targets=sum(mask), messages=messages, spans=spans)
        if events:
            row['memory'] = events
        handles['jsonl'].write(json.dumps(row, ensure_ascii=True) + '\n')
        segments = 0
        for start in range(0, len(tokens), 512):
            end = min(start + 512, len(tokens))
            targets = sum(mask[start:end])
            if targets:
                handles['segments.jsonl'].write(json.dumps(dict(id=fingerprint, source=source,
                    offset=offset, start=start, end=end, targets=targets)) + '\n')
                segments += 1
        stat = self.stats[f'{split}/{source}']
        stat.update(examples=1, tokens=len(tokens), targets=sum(mask), segments=segments,
                    multi_turn=int(sum(m['role'] == 'assistant' for m in messages) > 1))
        self.seen.add(fingerprint)
        self.groups[group] = split
        for a in answer_ids:
            self.answers[a] = split
        return True

    def close(self):
        for handles in self.handles.values():
            for handle in handles.values():
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dump/data/chat-memory-v2')
    parser.add_argument('--chat-per-shard', type=int, default=50000)
    parser.add_argument('--stories', type=int, default=100000)
    parser.add_argument('--memory', type=int, default=40000)
    parser.add_argument('--max-tokens', type=int, default=2560)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a fresh output directory; existing datasets are immutable.')
    if min(args.chat_per_shard, args.stories, args.memory) < 1 or not 512 <= args.max_tokens <= 2560:
        raise SystemExit('Positive source limits and max-tokens 512..2560 are required.')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    pin_cpu()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    sources = json.loads((CACHE / 'sources.json').read_text())
    for source in sources:
        if sha(Path(source['path'])) != source['sha256']:
            raise ValueError('Source hash changed')
    tokenpath = ROOT / 'dump/data/token-memory/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenpath))
    protected = set()
    for split in ('train', 'dev', 'test'):
        path = ROOT / f'dump/data/mixed-chat-v1/{split}.json'
        if path.exists():
            for row in json.loads(path.read_text()):
                question = row['prompt'].split('User: ', 1)[1].rsplit('\nAssistant:', 1)[0]
                if len(question) >= 40:
                    protected.add(identity(question))
    args.output.mkdir(parents=True)
    writer = Writer(args.output, tokenizer, args.max_tokens, protected)
    scanned = Counter()
    try:
        # Everyday first: keep these whole conversations if repeated in Smol-SmolTalk.
        for source in sorted(sources, key=lambda s: (s['source'] != 'everyday', s['source'], s['remote'])):
            name, count = source['source'], 0
            cap = args.stories if name == 'stories' else args.chat_per_shard if name == 'chat' else math.inf
            for batch in pq.ParquetFile(source['path']).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    origin = dict(file=Path(source['path']).name, row=scanned[Path(source['path']).name])
                    scanned[Path(source['path']).name] += 1
                    if name == 'chat':
                        origin['component'] = row.get('source')
                        if row.get('source') not in {'smol-magpie-ultra-short', 'smol-contraints', 'smollm-rewrite-30k',
                                                     'smol-summarize-20k', 'smol-summarize-5k', 'everyday-conversations'}:
                            writer.rejects['component_outside_chat_scope'] += 1
                            continue
                    if name == 'stories':
                        story = row['text'].strip().removesuffix('<|endoftext|>').strip()
                        group = identity(story)
                        prompt = STORY_PROMPTS[int(group[:8], 16) % len(STORY_PROMPTS)]
                        messages = [dict(role='user', content=prompt), dict(role='assistant', content=story)]
                    else:
                        group = None
                        messages = row['messages']
                    count += writer.add(messages, name, origin, group=group)
                    if count and count % 10000 == 0:
                        print(f'Prepared {name}: {count}/{cap}', flush=True)
                    if count >= cap:
                        break
                if count >= cap:
                    break
            print(f'Finished {origin["file"]}: {count} complete dialogues', flush=True)
        for i in range(args.memory):
            group = identity(f'memory-episode-v2-{i}')
            split = split_for(group)
            messages, events = memory_episode(i, split)
            writer.add(messages, 'memory', dict(generator='notebook-location-v2', index=i),
                       group=group, split=split, events=events)
            if (i + 1) % 10000 == 0:
                print(f'Memory episodes: {i + 1}/{args.memory}', flush=True)
    finally:
        writer.close()
    import shutil
    shutil.copy2(tokenpath, args.output / 'tokenizer.json')
    hashes = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    train = Counter()
    for key, stats in writer.stats.items():
        if key.startswith('train/'):
            train.update(stats)
    manifest = dict(format='chat-memory-v2', state='prepared', sources=sources,
        tokenizer_sha256=sha(tokenpath), hashes=hashes, counts=dict(writer.stats), rejected=dict(writer.rejects), scanned=dict(scanned),
        max_tokens=args.max_tokens, active_tokens=512, old_vectors_max=32, token_dtype='<u2', loss_dtype='u1',
        template='Per-message newline + Role: header, leading-space body, assistant EOS2; see encode_dialogue',
        loss='Assistant body and EOS only; headers/user/system ignored; causal next-token shift in loader',
        splits='1% dev/1% test by first substantive user message (>=40 chars), otherwise all user messages; stories by content; memory by episode ID with held-out names/question templates',
        dedup='Exact normalized full dialogues and long cross-split answers; no semantic near-duplicate guarantee',
        previous_long_question_identities_excluded=len(protected),
        memory='Locally generated labelled episodes; DB actions in metadata, not textual tokens; reset marker only, no DB implemented',
        epochs=2, train_totals=dict(train), updates_two_epochs_batch32=2 * math.ceil(train['segments'] / 32),
        fixed_consumer_parameter_cap=23209216, encoder_parameters_extra=9426432,
        licenses={'stories':'CDLA-Sharing-1.0', 'chat':'Apache-2.0 dataset card; preserve upstream component notices',
                  'everyday':'Apache-2.0', 'memory':'locally generated'},
        notes=['Bounded subsets, not the entire upstream corpora.', 'Do not feed this dataset to the legacy token-memory-v1 trainer.',
               'Two epochs means each prepared training segment twice, without replacement within an epoch.',
               'New DB and no-DB trainers must consume exactly the same targets and schedule. No training performed.'])
    manifest['preparation_sources'] = {n: sha(ROOT / 'tools' / n) for n in ('prepare_chat_corpus.py', 'download_chat_sources.py')}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(train=dict(train), stats=dict(writer.stats), rejects=dict(writer.rejects),
                          updates_two_epochs_batch32=manifest['updates_two_epochs_batch32']), indent=2), flush=True)


if __name__ == '__main__':
    main()
