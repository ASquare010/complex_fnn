"""Read saved validation only; never loads models or launches training."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'dump/raw-transformer-2560-20k')
    parser.add_argument('--hierarchical',type=Path,default=ROOT/'dump/hierarchical-memory-64-20k')
    args=parser.parse_args()
    arms={'token_memory':[ROOT/'dump'/n for n in ('token-memory-512-fresh','token-memory-512-micro8','token-memory-512-micro16')],
          'raw_transformer':[args.baseline],'hierarchical_memory':[args.hierarchical]}
    scores={};recipes={}
    for arm,runs in arms.items():
        scores[arm]={}
        for run in runs:
            if (run/'validation.json').exists():
                for row in json.loads((run/'validation.json').read_text()):scores[arm][row['step']]=row
            if (run/'recipe.json').exists():recipes[arm]=json.loads((run/'recipe.json').read_text())
        latest=max(scores[arm],default=None)
        print(f'{arm}: '+('no validation yet' if latest is None else f'latest validation {latest}'))
    reference=recipes.get('token_memory',{})
    for arm,recipe in recipes.items():
        for key in ('data_manifest_sha256','seed','batch_size','steps','lr','valid_documents'):
            if recipe.get(key)!=reference.get(key):raise SystemExit(f'Not matched: {arm} differs in {key}')
    shared=set.intersection(*(set(v) for v in scores.values()))
    print('\nStep | token memory CE | raw baseline CE | hierarchical CE')
    for step in sorted(shared):
        print(str(step)+' | '+' | '.join(f'{scores[arm][step]["ce"]:.5f}' for arm in arms))
    if not shared:print('No validation step is available for all three yet.')
    print('Same tokenizer/targets; different architectures and compute budgets. Lower CE is better.')


if __name__=='__main__':main()
