"""Verified local FineWeb paragraphs, retokenized with the frozen codec.

No joining unrelated paragraphs; no test-set access. Independent 64-token
encoding prevents future chunks leaking into teacher-forced input vectors.
"""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read_source(root, tokenizer, limits=(3000, 200), verify=True):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('complete') is False:
        raise ValueError('Dataset preparation is incomplete; refusing training')
    original = Tokenizer.from_file(str(root / 'tokenizer.json'))
    assert digest(root / 'tokenizer.json') == manifest['tokenizer_sha256']
    if not verify and original.to_str() != tokenizer.to_str():
        raise ValueError('Fast loading requires the dataset tokenizer to match the codec; use --verify-data for retokenization')
    result, documents, hashes = ({}, {}, {})
    for split, limit in zip(('train', 'valid'), limits):
        path = root / f'{split}.jsonl'
        hashes[split] = digest(path) if verify else manifest['split_sha256'][split]
        if verify:
            assert hashes[split] == manifest['split_sha256'][split]
        selected, doc_ids, seen = ([], set(), set())
        count = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                count += 1
                if count % 10000 == 0:
                    print(f"Loading {split}: {count}/{manifest['counts'][split]} rows", flush=True)
                if verify:
                    assert original.decode(row['ids'], skip_special_tokens=False) == row['text']
                doc_ids.add(row['document_id'])
                text_hash = hashlib.sha256(row['text'].encode()).hexdigest() if verify else row.get('paragraph_id', '')
                seen.add(text_hash)
                if len(selected) >= limit:
                    continue
                ids = tokenizer.encode(row['text']).ids if verify else row['ids']
                if manifest.get('window_format') == 'contiguous_document_tokens':
                    if len(ids) != manifest['window_tokens']:
                        raise ValueError('Document window length changed under codec tokenizer')
                usable = len(ids) // 64 * 64
                if usable >= 128:
                    selected.append({'tokens': torch.tensor(ids[:usable]).reshape(-1, 64), 'document_id': row['document_id'], 'text_sha256': text_hash})
        assert count == manifest['counts'][split]
        if len(selected) != limit:
            raise ValueError(f'{split}: requested {limit} eligible paragraphs, found {len(selected)}; prepare a larger dataset')
        result[split] = selected
        documents[split] = (doc_ids, seen)
    assert documents['train'][0].isdisjoint(documents['valid'][0])
    if verify:
        assert documents['train'][1].isdisjoint(documents['valid'][1])
    return (result, {'source': manifest, 'split_sha256': hashes, 'selected': {s: len(v) for s, v in result.items()}, 'verification': 'full' if verify else 'fast: manifest split hashes recorded, not reverified; text round-trip and text-overlap checks skipped', 'note': 'Encoder exposure possible. Development only.'})
