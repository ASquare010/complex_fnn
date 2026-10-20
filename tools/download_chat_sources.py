"""Fetch pinned public data only; no model loading or training."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'dump/data/chat-corpus-sources'
SOURCES = {
    'stories': ('roneneldan/TinyStories', 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
                ['data/train-00000-of-00004-2d5a1467fff1081b.parquet']),
    'chat': ('HuggingFaceTB/smol-smoltalk', 'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc',
             [f'data/train-{i:05d}-of-00004.parquet' for i in range(4)]),
    'everyday': ('HuggingFaceTB/everyday-conversations-llama3.1-2k',
                 '14f543216b9ba42b6b951dc5bd199460d193b162',
                 ['data/train_sft-00000-of-00001.parquet']),
}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    records = []
    for source, (repo, revision, files) in SOURCES.items():
        for i, remote in enumerate(files):
            path = CACHE / f'{source}-{i}.parquet'
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}'
            # The existing immutable shard has precisely the same pinned origin.
            old = ROOT / 'dump/data/mixed-chat-v1/source.parquet'
            if source == 'chat' and i == 0 and old.exists() and not path.exists():
                os.link(old, path)
            receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved['url'] != url or sha(path) != saved['sha256']:
                    raise ValueError(f'Cached source identity changed: {path}')
            elif not path.exists():
                print(f'Downloading {source} shard {i}', flush=True)
                size = 0
                with urllib.request.urlopen(url, timeout=120) as response, path.with_suffix('.part').open('wb') as f:
                    expected = response.headers.get('Content-Length')
                    while chunk := response.read(4 * 1024**2):
                        f.write(chunk)
                        size += len(chunk)
                        if size % (32 * 1024**2) == 0:
                            print(f'{source}-{i}: {size // 1024**2} MiB', flush=True)
                    if expected is not None and size != int(expected):
                        raise ValueError('Incomplete source download')
                path.with_suffix('.part').replace(path)
            record = dict(source=source, repo=repo, revision=revision, remote=remote,
                          url=url, path=str(path), bytes=path.stat().st_size, sha256=sha(path))
            receipt.write_text(json.dumps(record, indent=2))
            records.append(record)
            print(f'Verified {path.name}', flush=True)
    (CACHE / 'sources.json').write_text(json.dumps(records, indent=2))
    print('Sources ready. No encoding or training performed.', flush=True)


if __name__ == '__main__':
    main()
