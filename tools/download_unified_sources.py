"""Download immutable public source shards for unified chat data; never trains."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
from download_chat_sources import ROOT, sha

REV = 'fc6cc2103c066455aade5d7fbb346039ae36ca5e'
COMPONENTS = {
    'personas': 'smoltalk_smollm3_systemchats_30k_no_think',
    'instructions': 'tulu_3_sft_personas_instruction_following_no_think',
    'science': 'Mixture_of_Thoughts_science_no_think',
    'tools': 'hermes_function_calling_v1_no_think',
    'tool_selection': 'xlam_traces_no_think',
    'reasoning': 'smoltalk_systemchats_Qwen3_32B_think',
    'reasoning_chat': 'smoltalk_everyday_convs_reasoning_Qwen3_32B_think',
    'tables': 'table_gpt_no_think',
    'reasoning_tables': 'table_gpt_Qwen3_32B_think',
}
SOURCES = [dict(name=k, repo='HuggingFaceTB/smoltalk2', revision=REV,
                file=f'SFT/{v}-00000-of-00001.parquet') for k,v in COMPONENTS.items()]
SOURCES += [dict(name='broad_chat', repo='HuggingFaceTB/smoltalk2', revision=REV,
                 file='SFT/OpenHermes_2.5_no_think-00000-of-00002.parquet'),
            dict(name='education', repo='HuggingFaceFW/fineweb-edu',
                 revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
                 file='sample/10BT/000_00000.parquet')]


def fetch(url, path):
    part = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as r, part.open('wb') as f:
        expected = r.headers.get('Content-Length')
        total = 0
        while block := r.read(4 * 1024**2):
            f.write(block)
            total += len(block)
        if expected and total != int(expected):
            raise ValueError('Incomplete download')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'dump/data/unified-chat-sources')
    p.add_argument('--sources', nargs='+', choices=[s['name'] for s in SOURCES])
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    tokenizer = args.cache/'tokenizer.json'
    if not tokenizer.exists():
        fetch('https://huggingface.co/awaisamin010/latent-text-compressor-64/resolve/fced6bd27cb3934896b4178f4792543832d9ab57/tokenizer.json', tokenizer)
    if sha(tokenizer) != 'f7797957af1aa8f77656599def7d31a44062bd800ad8179713a3f3af01b504e5':
        raise ValueError('Frozen tokenizer identity mismatch')
    records = []
    for source in SOURCES:
        if args.sources and source['name'] not in args.sources:
            continue
        path = args.cache / (source['name'] + '.parquet')
        receipt = path.with_suffix('.json')
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
        if path.exists():
            if not receipt.exists():
                raise ValueError(f'Unverified existing file: {path}')
            old = json.loads(receipt.read_text())
            if old['url'] != url or sha(path) != old['sha256']:
                raise ValueError(f'Source changed: {path}')
        else:
            if shutil.disk_usage(args.cache).free < 5 * 1024**3:
                raise ValueError('Keep at least 5 GiB free before starting another source download')
            print(f"Downloading {source['name']}", flush=True)
            fetch(url, path)
        record = dict(source, url=url, bytes=path.stat().st_size, sha256=sha(path))
        receipt.write_text(json.dumps(record, indent=2))
        records.append(record)
        card = args.cache / (source['name'] + '-README.md')
        if not card.exists():
            fetch(f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/README.md", card)
        print(f"Verified {source['name']}: {record['bytes']/1024**2:.1f} MiB", flush=True)
    print('Public source shards ready. No training performed.', flush=True)


if __name__ == '__main__':
    main()
