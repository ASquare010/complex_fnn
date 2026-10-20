"""Prepare one pinned, bounded chat corpus for matched encoder-memory experiments."""
import argparse
import ast
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil

from prepare_chat_corpus import ROOT, Writer, identity, split_for, memory_episode, STORY_PROMPTS
from download_chat_sources import sha, CACHE as BASE_CACHE
from download_unified_sources import SOURCES
from chat_data.unified_format import encode_dialogue, with_mode

# Accepted example ceilings, not repeat weights. The manifest reports actual
# assistant-target token shares; a long source must not silently dominate.
CAPS = dict(personas=25000, instructions=25000, science=25000, tools=10000,
            tool_selection=40000, reasoning=12000, reasoning_chat=2500,
            tables=10000, reasoning_tables=4000, broad_chat=75000, education=30000)
TAGS = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


def clean(messages):
    result = []
    previous = None
    for m in messages:
        role, text = m.get('role'), m.get('content')
        if role not in ('system', 'user', 'assistant', 'tool') or not isinstance(text, str) or not text.strip():
            raise ValueError('invalid_message')
        if '\x00' in text or '<|endoftext|>' in text or '\ufffd' in text:
            raise ValueError('control_text')
        if role == 'system':
            if result:
                raise ValueError('system_position')
        elif role == 'assistant':
            if previous not in ('user', 'tool'):
                raise ValueError('assistant_order')
        elif role == 'tool':
            if previous not in ('assistant', 'tool'):
                raise ValueError('tool_order')
        elif previous not in (None, 'system', 'assistant'):
            raise ValueError('user_order')
        result.append(dict(role=role, content=text.strip()))
        previous = role
    if previous != 'assistant':
        raise ValueError('unfinished_dialogue')
    return result


def parse_literal(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)  # Data literals only; never execute source code.


def adapt(row):
    messages = copy.deepcopy(row['messages'])
    kwargs = row.get('chat_template_kwargs') or {}
    thinking = bool(kwargs.get('enable_thinking', False))
    custom = kwargs.get('custom_instructions') or ''
    if kwargs.get('python_tools'):
        raise ValueError('unsupported_python_tool_schema')
    schemas = []
    for text in kwargs.get('xml_tools') or []:
        match = re.fullmatch(r'\s*<tools>\s*(.*?)\s*</tools>\s*', text, re.S)
        if not match:
            raise ValueError('tool_schema_wrapper')
        parsed = parse_literal(match[1])
        if not isinstance(parsed, list):
            raise ValueError('tool_schema_list')
        schemas.extend(parsed)
    functions = {s['function']['name']: s['function'] for s in schemas}
    pending = []
    traces = 0
    for m in messages:
        if m['role'] == 'assistant':
            text = m['content']
            matches = list(re.finditer(r'<think>(.*?)</think>', text, re.S))
            if text.count('<think>') != len(matches) or text.count('</think>') != len(matches):
                raise ValueError('incomplete_reasoning')
            for match in matches:
                if match[1].strip():
                    if not thinking or not text[match.end():].strip():
                        raise ValueError('reasoning_mode_or_missing_final')
                    traces += 1
                else:
                    text = text.replace(match[0], '').strip()
            calls = TAGS.findall(text)
            if text.count('<tool_call>') != len(calls) or text.count('</tool_call>') != len(calls):
                raise ValueError('incomplete_tool_call')
            pending = []
            def normalize_call(match):
                obj = parse_literal(match[1])
                if not isinstance(obj, dict) or obj.get('name') not in functions or not isinstance(obj.get('arguments'), dict):
                    raise ValueError('invalid_tool_call')
                function = functions[obj['name']]
                parameters = function.get('parameters') or {}
                required = parameters.get('required', [])
                if not set(required).issubset(obj['arguments']):
                    raise ValueError('missing_tool_argument')
                pending.append(obj['name'])
                return '<tool_call>\n' + json.dumps(obj, sort_keys=True, ensure_ascii=True) + '\n</tool_call>'
            m['content'] = TAGS.sub(normalize_call, text)
        elif m['role'] == 'tool':
            if not pending:
                raise ValueError('orphan_tool_result')
            results = re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>', m['content'], re.S)
            if not results:
                raise ValueError('invalid_tool_response')
            parsed = [parse_literal(s) for s in results]
            for result in parsed:
                if result.get('name') not in pending:
                    raise ValueError('tool_response_name')
                pending.remove(result['name'])
            m['content'] = '\n'.join('<tool_response>\n' + json.dumps(r, sort_keys=True, ensure_ascii=True) + '\n</tool_response>' for r in parsed)
    if thinking and not traces:
        raise ValueError('missing_reasoning')
    if schemas:
        custom += '\nAvailable tools: ' + json.dumps(schemas, sort_keys=True, ensure_ascii=True)
    if custom.strip():
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = custom.strip() + '\n' + messages[0]['content']
        else:
            messages.insert(0, dict(role='system', content=custom.strip()))
    return with_mode(messages, thinking), thinking


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-cache', type=Path, default=BASE_CACHE)
    p.add_argument('--tokenizer', type=Path, help='Defaults to the pinned tokenizer in --cache')
    p.add_argument('--cache', type=Path, default=ROOT/'dump/data/unified-chat-sources')
    p.add_argument('--output', type=Path, default=ROOT/'dump/data/unified-chat-v3')
    p.add_argument('--limit-per-source', type=int, help='Small pipeline smoke test only; changes recipe identity')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Dataset outputs are immutable; choose a fresh output')
    if args.limit_per_source is not None and args.limit_per_source < 1:
        raise ValueError('Limit must be positive')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from models.token_memory.runtime import pin_cpu
    pin_cpu()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    base_sources = json.loads((args.base_cache/'sources.json').read_text())
    from download_chat_sources import SOURCES as BASE_SOURCES
    for source in base_sources:
        repo, revision, files = BASE_SOURCES[source['source']]
        if source['repo'] != repo or source['revision'] != revision or source['remote'] not in files:
            raise ValueError('Unpinned base source')
        if sha(args.base_cache/Path(source['path']).name) != source['sha256']:
            raise ValueError('Base source changed')
    expected = {(name,remote) for name,(_,_,files) in BASE_SOURCES.items() for remote in files}
    if {(s['source'],s['remote']) for s in base_sources} != expected:
        raise ValueError('Missing or extra base sources')
    tokenizer_path = args.tokenizer or args.cache/'tokenizer.json'
    if sha(tokenizer_path) != 'f7797957af1aa8f77656599def7d31a44062bd800ad8179713a3f3af01b504e5':
        raise ValueError('Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    receipts = []
    for source in SOURCES:
        path = args.cache/(source['name']+'.parquet')
        record = json.loads(path.with_suffix('.json').read_text())
        if any(record[k] != v for k,v in source.items()) or sha(path) != record['sha256']:
            raise ValueError(f'Wrong source identity: {path}')
        receipts.append(record)
    if shutil.disk_usage(args.output.parent).free < 4*1024**3:
        raise ValueError('At least 4 GiB of free space is required for preparation')
    args.output.mkdir(parents=True)
    writer = Writer(args.output, tokenizer, 2560, set(), cleaner=clean, serializer=encode_dialogue)
    accepted, scanned, modes = Counter(), Counter(), Counter()
    try:
        # Build directly from pinned public files, never from previous runs or
        # machine-local exclusion lists. Same inputs produce the same splits.
        for source in sorted(base_sources, key=lambda s:(s['source']!='everyday',s['source'],s['remote'])):
            name = source['source']
            cap = dict(stories=50000, chat=110000, everyday=3000)[name]
            cap = min(cap, args.limit_per_source or cap)
            if accepted[name] >= cap:
                continue
            path = args.base_cache/Path(source['path']).name
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    index = scanned[path.name]
                    scanned[path.name] += 1
                    if name == 'stories':
                        story = row['text'].strip().removesuffix('<|endoftext|>').strip()
                        group = identity(story)
                        prompt = STORY_PROMPTS[int(group[:8],16)%len(STORY_PROMPTS)]
                        messages = [dict(role='user',content=prompt),dict(role='assistant',content=story)]
                    else:
                        messages, group = row['messages'], None
                    ok = writer.add(with_mode(messages, False),name,
                                    dict(file=path.name,row=index,component=row.get('source',name),reasoning=False),group=group)
                    accepted[name] += ok
                    modes['off'] += ok
                    if accepted[name] >= cap:
                        break
                if accepted[name] >= cap:
                    break
            print(f'Prepared {name}: {accepted[name]:,}/{cap:,}',flush=True)
        for index in range(min(40000,args.limit_per_source or 40000)):
            group = identity(f'memory-episode-v2-{index}')
            split = split_for(group)
            messages, events = memory_episode(index,split)
            for write in events['writes']:
                write['message'] += 1
            for key in ('before_message','query_message'):
                events['read'][key] += 1
            events['no_read_before_messages'] = [x+1 for x in events['no_read_before_messages']]
            events['session_reset_before_message'] += 1
            ok = writer.add(with_mode(messages,False),'memory',dict(generator='notebook-location-v2',index=index,reasoning=False),
                            group=group,split=split,events=events)
            accepted['memory'] += ok
            modes['off'] += ok
        print(f"Prepared memory: {accepted['memory']:,}",flush=True)
        for source in SOURCES:
            name = source['name']
            cap = min(CAPS[name], args.limit_per_source or CAPS[name])
            path = args.cache/(name+'.parquet')
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    index = scanned[name]
                    scanned[name] += 1
                    try:
                        if name == 'education':
                            text = row['text'].strip()
                            # Whole documents only. A real prefix continuation,
                            # not fabricated question answering or generic essay output.
                            paragraphs = text.split('\n', 1)
                            if len(paragraphs) != 2 or not all(s.strip() for s in paragraphs):
                                raise ValueError('education_no_paragraph_boundary')
                            messages = with_mode([dict(role='user', content='Continue this educational passage:\n'+paragraphs[0]),
                                                  dict(role='assistant', content=paragraphs[1])], False)
                            thinking = False
                            group = identity(text)
                        else:
                            messages, thinking = adapt(row)
                            users = [m['content'] for m in messages if m['role']=='user']
                            group = identity(next((x for x in users if len(x)>=40), '\n'.join(users)))
                        # Retain the original partition for shared prompts across
                        # think/no-think variants and sources.
                        split = writer.groups.get(group, split_for(group))
                        origin = dict(component=row.get('source', name), row=index, source_file=name+'.parquet', reasoning=thinking)
                        ok = writer.add(messages, name, origin, group=group, split=split)
                        accepted[name] += ok
                        modes['on' if thinking else 'off'] += ok
                    except (ValueError, KeyError, TypeError, SyntaxError) as error:
                        writer.rejects['adapter:'+str(error)[:90]] += 1
                    if accepted[name] >= cap:
                        break
                if accepted[name] >= cap:
                    break
            print(f'Prepared {name}: {accepted[name]:,}/{cap:,}; scanned {scanned[name]:,}', flush=True)
    finally:
        writer.close()
    shutil.copy2(tokenizer_path, args.output/'tokenizer.json')
    frozen = args.output/'preparation_source'
    frozen.mkdir()
    for path in [Path(__file__), ROOT/'tools/prepare_chat_corpus.py', ROOT/'tools/download_unified_sources.py',
                 ROOT/'tools/download_chat_sources.py', ROOT/'src/chat_data/unified_format.py']:
        shutil.copy2(path, frozen/path.name)
    cards = args.output/'source_cards'
    cards.mkdir()
    for card in args.cache.glob('*-README.md'):
        shutil.copy2(card, cards/card.name)
    totals = Counter()
    for key, stats in writer.stats.items():
        if key.startswith('train/'):
            totals.update(stats)
    manifest = dict(format='chat-memory-v3', state='prepared', base_sources=[{k:v for k,v in s.items() if k!='path'} for s in base_sources],
                    sources=receipts, counts=dict(writer.stats), train_totals=dict(totals), accepted=dict(accepted),
                    scanned=dict(scanned), rejected=dict(writer.rejects), reasoning_modes=dict(modes),
                    caps=CAPS, smoke_limit=args.limit_per_source, tokenizer_sha256=sha(tokenizer_path),
                    active_tokens=512, old_vectors_max=32, max_tokens=2560, epochs=2,
                    updates_two_epochs_batch32=2*math.ceil(totals['segments']/32),
                    template='unified_format.py; explicit Reasoning on/off, Tool role, assistant-only CE including calls/EOS',
                    dedup='Normalized exact dialogues, grouped prompts and cross-split long answers; not semantic deduplication',
                    benchmark_decontamination='Upstream SmolTalk2 filtering plus local splits; independent named benchmark decontamination NOT performed',
                    fixed_consumer_parameter_cap=23209216,
                    notes=['No optimizer or encoder pass.','Tool syntax/name/required-field checks are not execution validation.',
                           'DB labels remain metadata; no autonomous database behavior is claimed.',
                           'Whole examples exceeding 2560 tokens are rejected, including long reasoning.'])
    manifest['hashes'] = {str(p.relative_to(args.output)).replace(chr(92),'/'):sha(p) for p in args.output.rglob('*') if p.is_file()}
    from importlib.metadata import version
    manifest['dependency_versions'] = {name:version(name) for name in ('tokenizers','pyarrow','numpy')}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(accepted=accepted, train=totals, reasoning=modes,
                          updates=manifest['updates_two_epochs_batch32']), indent=2), flush=True)


if __name__ == '__main__':
    main()
