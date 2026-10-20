"""Append all eligible pinned tool/reasoning rows to immutable unified-chat-v3."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import re
import shutil

from prepare_unified_chat import ROOT, adapt, clean, parse_literal, TAGS
from prepare_chat_corpus import Writer, identity, split_for
from chat_data.unified_format import encode_dialogue
from download_unified_sources import SOURCES as ORIGINAL, sha
from download_chat_extensions import SOURCES as EXTRA

NAMES = ('tools', 'tool_selection', 'reasoning', 'reasoning_chat', 'reasoning_tables',
         'reasoning_multiturn', 'reasoning_tools')


def schema_list(text):
    """Accept JSON/list literals or concatenated JSON objects; never execute code."""
    try:
        value = parse_literal(text)
        return value if isinstance(value, list) else [value]
    except (ValueError, SyntaxError):
        result, remaining = [], text.strip()
        decoder = json.JSONDecoder()
        while remaining:
            obj, end = decoder.raw_decode(remaining)
            result.extend(obj if isinstance(obj, list) else [obj])
            remaining = remaining[end:].strip()
        return result


def adapt_extended(row):
    row = copy.deepcopy(row)
    kwargs = row.get('chat_template_kwargs') or {}
    row['chat_template_kwargs'] = kwargs
    schemas, instructions = [], []
    for text in kwargs.get('xml_tools') or []:
        # Preserve already-supported rows byte-for-byte for cross-version dedup.
        if re.fullmatch(r'\s*<tools>\s*(.*?)\s*</tools>\s*', text, re.S):
            match = re.fullmatch(r'\s*<tools>\s*(.*?)\s*</tools>\s*', text, re.S)
            schemas.extend(schema_list(match[1]))
        elif '<tools>' in text:
            blocks = list(re.finditer(r'<tools>(.*?)</tools>', text, re.S))
            actual = [m for m in blocks if m[1].strip()]
            if not actual:
                raise ValueError('missing_tool_schema')
            for match in actual:
                schemas.extend(schema_list(match[1]))
            # Empty illustrative tags remain part of the source instructions.
            for match in reversed(actual):
                text = text[:match.start()] + text[match.end():]
            instructions.append(text.strip())
        else:
            schemas.extend(schema_list(text))
    if schemas:
        names = [s['function']['name'] for s in schemas]
        if len(set(names)) != len(names):
            raise ValueError('duplicate_tool_name')
        kwargs['xml_tools'] = ['<tools>'+json.dumps(schemas, ensure_ascii=True)+'</tools>']
    if instructions:
        kwargs['custom_instructions'] = '\n'.join(filter(None, [kwargs.get('custom_instructions'), *instructions]))
    pending = []
    for message in row['messages']:
        if message['role'] == 'assistant':
            pending = [parse_literal(c)['name'] for c in TAGS.findall(message['content'])]
        elif message['role'] == 'tool' and '<tool_response>' not in message['content']:
            # A plain observation is attributable only to a single outstanding call.
            if len(pending) != 1:
                raise ValueError('ambiguous_tool_observation')
            message['content'] = '<tool_response>'+json.dumps(dict(name=pending.pop(), content=message['content']), ensure_ascii=True)+'</tool_response>'
    return adapt(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT/'dump/data/unified-chat-v3')
    p.add_argument('--cache', type=Path, default=ROOT/'dump/data/unified-chat-sources')
    p.add_argument('--output', type=Path, default=ROOT/'dump/data/unified-chat-v4')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new immutable output directory')
    from models.token_memory.runtime import pin_cpu
    pin_cpu()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    base = json.loads((args.base/'manifest.json').read_text())
    if base['format'] != 'chat-memory-v3' or base.get('smoke_limit') is not None:
        raise ValueError('A complete v3 base is required')
    for name, digest in base['hashes'].items():
        if sha(args.base/name) != digest:
            raise ValueError('Base file changed: '+name)
    sources = [s for s in ORIGINAL+EXTRA if s['name'] in NAMES]
    receipts = []
    for source in sources:
        path = args.cache/(source['name']+'.parquet')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if any(receipt.get(k) != v for k,v in source.items()) or sha(path) != receipt['sha256']:
            raise ValueError('Source identity mismatch: '+source['name'])
        receipts.append(receipt)
    base_bytes = sum((args.base/f'{s}.{n}').stat().st_size for s in ('train','dev','test')
                     for n in ('tokens.bin','loss.bin','jsonl','segments.jsonl'))
    if shutil.disk_usage(args.output.parent).free < base_bytes + 2*1024**3:
        raise ValueError('Insufficient space for copied base plus additions')
    args.output.mkdir(parents=True)
    writer = Writer(args.output, Tokenizer.from_file(str(args.base/'tokenizer.json')), 2560,
                    set(), cleaner=clean, serializer=encode_dialogue)
    additions, scanned, modes = Counter(), Counter(), Counter(base['reasoning_modes'])
    try:
        for split, handles in writer.handles.items():
            for name, handle in handles.items():
                mode = 'rb' if name.endswith('.bin') else 'r'
                with (args.base/f'{split}.{name}').open(mode) as source:
                    shutil.copyfileobj(source, handle)
                handle.flush()
            with (args.base/f'{split}.jsonl').open() as f:
                for line in f:
                    row = json.loads(line)
                    writer.seen.add(row['id'])
                    writer.groups[row['group']] = split
                    for m in row['messages']:
                        if m['role']=='assistant' and len(m['content'])>=100:
                            writer.answers[identity(m['content'])] = split
            print('Copied and indexed '+split, flush=True)
        for key, counts in base['counts'].items():
            writer.stats[key].update(counts)
        for source in sources:
            name = source['name']
            for batch in pq.ParquetFile(args.cache/(name+'.parquet')).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    index = scanned[name]
                    scanned[name] += 1
                    try:
                        messages, thinking = adapt_extended(row)
                        users = [m['content'] for m in messages if m['role']=='user']
                        group = identity(next((x for x in users if len(x)>=40), '\n'.join(users)))
                        ok = writer.add(messages, name, dict(component=row.get('source',name), row=index,
                            source_file=name+'.parquet', reasoning=thinking, extension='v4'), group=group,
                            split=writer.groups.get(group, split_for(group)))
                        additions[name] += ok
                        modes['on' if thinking else 'off'] += ok
                    except (ValueError, KeyError, TypeError, SyntaxError) as error:
                        writer.rejects[name+':adapter:'+str(error)[:90]] += 1
                if scanned[name] % 4096 == 0:
                    print(f'{name}: scanned {scanned[name]:,}; added {additions[name]:,}', flush=True)
            print(f'Complete {name}: added {additions[name]:,} / scanned {scanned[name]:,}', flush=True)
    finally:
        writer.close()
    shutil.copy2(args.base/'tokenizer.json', args.output/'tokenizer.json')
    shutil.copytree(args.base/'preparation_source', args.output/'base_preparation_source')
    shutil.copytree(args.base/'source_cards', args.output/'source_cards')
    frozen = args.output/'preparation_source'
    frozen.mkdir()
    for path in [Path(__file__), ROOT/'tools/download_chat_extensions.py', ROOT/'tools/prepare_unified_chat.py',
                 ROOT/'tools/prepare_chat_corpus.py', ROOT/'tools/download_unified_sources.py',
                 ROOT/'tools/download_chat_sources.py', ROOT/'src/chat_data/unified_format.py']:
        shutil.copy2(path, frozen/path.name)
    for source in EXTRA:
        shutil.copy2(args.cache/(source['name']+'-README.md'), args.output/'source_cards')
    totals = Counter()
    for key, counts in writer.stats.items():
        if key.startswith('train/'):
            totals.update(counts)
    manifest = copy.deepcopy(base)
    manifest.update(corpus_version='unified-chat-v4', parent_manifest_sha256=sha(args.base/'manifest.json'),
        extension_sources=receipts, extension_additions=dict(additions), extension_scanned=dict(scanned),
        extension_rejected=dict(writer.rejects), counts=dict(writer.stats), train_totals=dict(totals),
        reasoning_modes=dict(modes), updates_two_epochs_batch32=2*math.ceil(totals['segments']/32))
    # Historical base caps/rejections remain explicit rather than masquerading as v4 totals.
    for key in ('accepted','scanned','rejected','caps'):
        manifest['base_'+key] = manifest.pop(key)
    manifest['hashes'] = {str(p.relative_to(args.output)).replace(chr(92),'/'):sha(p)
                          for p in args.output.rglob('*') if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(additions=additions, train=totals, modes=modes), indent=2), flush=True)


if __name__ == '__main__':
    main()
