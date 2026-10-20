"""Download pinned public reasoning/tool additions; no model or training changes."""
import argparse
import json
import shutil
from pathlib import Path
from download_unified_sources import ROOT, REV, fetch, sha

SOURCES = [dict(name=name, repo='HuggingFaceTB/smoltalk2', revision=REV,
                file=f'SFT/{component}-00000-of-00001.parquet', expected_bytes=size)
           for name, component, size in [
               ('reasoning_multiturn', 'multi_turn_reasoning_if_think', 178159863),
               ('reasoning_tools', 'smolagents_toolcalling_traces_think', 81765978)]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'dump/data/unified-chat-sources')
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        path = args.cache/(source['name']+'.parquet')
        receipt = path.with_suffix('.json')
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
        if path.exists():
            old = json.loads(receipt.read_text())
            if any(old.get(k) != v for k,v in source.items()) or sha(path) != old['sha256']:
                raise ValueError('Existing source identity mismatch')
        else:
            if shutil.disk_usage(args.cache).free < source['expected_bytes'] + 4*1024**3:
                raise ValueError('Insufficient space: reserve 4 GiB for preparation')
            print('Downloading '+source['name'], flush=True)
            fetch(url, path)
        if path.stat().st_size != source['expected_bytes']:
            raise ValueError('Source byte count mismatch')
        receipt.write_text(json.dumps(dict(source, url=url, sha256=sha(path), bytes=path.stat().st_size), indent=2))
        card = args.cache/(source['name']+'-README.md')
        if not card.exists():
            fetch(f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/README.md", card)
        print('Verified '+source['name'], flush=True)


if __name__ == '__main__':
    main()
