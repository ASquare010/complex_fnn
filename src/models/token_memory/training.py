"""Fresh native256 memory training on the original two-epoch comparison recipe."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from models.token_memory import Model, PARAMETER_BUDGET
from models.token_memory.dataset import Data, CORPUS, CACHE
from models.token_memory.labels import Labels
from models.token_memory.objective import update
from models.token_memory.reader import SourceFeatures
from models.token_memory.runtime import ROOT, ENCODER, ENCODER_SHA, digest, load, save, write_json, pin_cpu, worker_lock
from models.token_memory.schedule import cursor, rewind_logs

FORMAT = 'token-memory-native256-lowbit-original-v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', help='New folder name under dump; never overwrites an existing run')
    p.add_argument('--resume', type=Path, help='Resume this architecture in its existing run folder')
    p.add_argument('--data', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--epochs', type=int, choices=[2], default=2)
    p.add_argument('--batch-size', type=int, choices=[32], default=32)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--stop-after', type=int, help='NEW updates, without changing the two-epoch schedule')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--inspect', action='store_true', help='Show architecture/recipe; no data or training')
    p.add_argument('--check-data', action='store_true', help='Verify corpus/cache/labels; no GPU or updates')
    return p


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT/'src/models/components.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def check_sources(previous, current):
    """Architecture migrations require a fresh run; exact resumes require equality."""
    normalize = lambda values: {k.replace('\\', '/'): v for k, v in values.items()}
    if normalize(previous) != normalize(current):
        raise ValueError('Sources changed; use the saved source archive')


def recipe(data, cache):
    return dict(format=FORMAT, data=str(Path(data).resolve()), cache=str(Path(cache).resolve()),
                epochs=2, batch_size=32, seed=47, reader_parameters=PARAMETER_BUDGET,
                encoder_separate=True, encoder_sha256=ENCODER_SHA, precision='FP32',
                raw_context=64, words_per_memory=64, width=256, layers=8,
                read_layers=[2, 5], ffn_hidden_total=38062, initialization='fresh seed47',
                lr=3e-4, minimum_lr=3e-5, warmup=200, weight_decay=.01, clip=1.,
                query_coefficient=.05, source_coefficient=.02, layout='length_first')


def checked_counts(args):
    if args.microbatch < 1 or 32 % args.microbatch:
        raise ValueError('Microbatch must be a positive divisor of effective batch 32')
    if min(args.eval_every, args.save_every, args.log_every) < 1:
        raise ValueError('Progress, evaluation and saving intervals must be positive')
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError('Stop-after must be positive')


def assets(data, labels):
    paths = [ENCODER, data.corpus/'manifest.json', data.corpus/'tokenizer.json']
    for split in ('train', 'dev'):
        paths += [data.corpus/f'{split}.{s}' for s in ('tokens.bin', 'loss.bin', 'segments.jsonl')]
        paths += [data.cache_dir/f'{split}.{s}' for s in ('json', 'f32')]
    paths += [labels.directory/'manifest.json'] + [labels.directory/n for n in labels.manifest['files']]
    result = {}
    for p in paths:
        # The large cache hash has already been checked by Data.
        result[str(p.resolve())] = (json.loads(p.with_suffix('.json').read_text())['sha256']
                                    if p.suffix == '.f32' else digest(p))
    if result[str(ENCODER.resolve())] != ENCODER_SHA:
        raise ValueError('Frozen encoder changed')
    return result


def physical_order(dataset, indices):
    rows = dataset.rows[indices]
    order = np.lexsort(((rows[:, 2]-1)//64, rows[:, 2]-rows[:, 1]))
    return [int(indices[i]) for i in order]


def validation_text(step, score):
    return (f"VALID {step:,} | CE {score['ce']:.6f} | accuracy {100*score['token_accuracy']:.3f}%"
            f" | correct {score['correct']:,}/{score['targets']:,}")


@torch.inference_mode()
def evaluate(model, data, folder, step, micro, device):
    model.eval()
    docs = defaultdict(lambda: [0, 0, 0.])
    groups = defaultdict(lambda: [0, 0, 0.])
    ds = data.datasets['dev']
    print(f'VALIDATING {step:,}: {len(ds):,} original development segments', flush=True)
    try:
        for at in range(0, len(ds), micro):
            indices = list(range(at, min(at+micro, len(ds))))
            memory, mv, x, y, tv, visible = data.batch('dev', indices, 'compressed64', device)
            h, source, pointer, null, _ = model.hidden(memory, mv, x, tv, visible)
            rr, cc = (y != -100).nonzero(as_tuple=True)
            chosen = SourceFeatures(source.ids[rr], None, None, None, 64, source.memory_positions)
            lp = model.reader.distribution(h[rr, cc][:, None], chosen, pointer[rr, cc][:, None],
                                           null[rr, cc][:, None], model.core.embedding.weight)[:, 0]
            loss = F.nll_loss(lp, y[rr, cc], reduction='none')
            correct = lp.argmax(-1) == y[rr, cc]
            for row, index in enumerate(indices):
                mask = rr == row
                values = [int(correct[mask].sum()), int(mask.sum()), float(loss[mask].sum())]
                for target in (docs[str(int(ds.rows[index, 0]))], groups[ds.sources[int(ds.rows[index, 4])]]):
                    for j, value in enumerate(values):
                        target[j] += value
        def metric(values):
            hits, count, ce = map(sum, zip(*values))
            return dict(correct=hits, targets=count, token_accuracy=hits/count, ce=ce/count)
        result = metric(docs.values())
        if result['targets'] != 1859388:
            raise ValueError('Original development label count differs')
        result['sources'] = {k: metric([v]) for k, v in groups.items()}
        write_json(folder/f'dev-documents-{step}.json', dict(docs))
        with (folder/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step, **result))+'\n')
        print(validation_text(step, result), flush=True)
        return result
    finally:
        model.train()


def execute(args):
    checked_counts(args)
    torch.set_num_threads(1)
    if args.inspect:
        model = Model()
        print(f'Reader: {sum(p.numel() for p in model.parameters()):,} parameters | native256 | 8 layers | no memory projection')
        print('Memory: FP32 low-bit payload | 64 old tokens per global position | live raw window 64')
        print('Fresh seed47 | 2 original epochs | effective batch 32 | LR 3e-4 to 3e-5 | query .05 + source .02')
        return
    parent = load(args.resume) if args.resume else None
    if parent and parent.get('format') != FORMAT:
        raise ValueError('This is a fresh native256 architecture; old checkpoints require their archived source')
    if args.resume:
        folder = args.resume.resolve()
        if folder.is_file():
            folder = folder.parent
        if args.name:
            raise ValueError('Resume uses the existing folder; omit --name')
    else:
        if not args.name or Path(args.name).name != args.name or args.name in ('.', '..') or ':' in args.name:
            raise ValueError('Provide a new plain --name under dump')
        folder = ROOT/'dump'/args.name
        if folder.exists() and not args.check_data:
            raise ValueError('Output already exists; choose a new name or use --resume')
    previous = parent['recipe'] if parent else {}
    config = recipe(args.data or previous.get('data', CORPUS), args.cache_dir or previous.get('cache', CACHE))
    snapshot = source_hashes()
    if parent and (previous != config or parent['sources'] != snapshot):
        raise ValueError('Recipe or code changed; use the saved source archive for exact resume')
    print('Verifying original train/dev corpus, frozen FP32 cache and training-only auxiliary labels...', flush=True)
    data = Data(config['data'], config['cache'])
    labels = Labels()
    inputs = assets(data, labels)
    if parent and inputs != parent['inputs']:
        raise ValueError('Training inputs changed')
    if args.check_data:
        print('DATA VERIFIED | 780,186 train segments | 48,762 updates | 361,357,104 assistant labels over two epochs')
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the CUDA Python environment')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(47)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    step = 0
    if parent:
        if parent['cursor'] != cursor(parent['step'], data.per_epoch) or not 0 <= parent['step'] <= data.total:
            raise ValueError('Invalid saved training cursor')
        if digest(folder/'orders.npy') != parent['orders_sha256']:
            raise ValueError('Saved sampler orders changed')
        model.load_state_dict(parent['model'])
        optimizer.load_state_dict(parent['optimizer'])
        step = parent['step']
        torch.set_rng_state(parent['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(parent['cuda_rng'])
        rewind_logs(folder, step)
        del parent
        orders = np.load(folder/'orders.npy', mmap_mode='r')
    else:
        folder.mkdir(parents=True)
        orders = np.array([data.order(epoch) for epoch in range(2)], dtype=np.int64)
        np.save(folder/'orders.npy', orders)
        write_json(folder/'recipe.json', dict(**config, core=asdict(model.core_config)))
        with zipfile.ZipFile(folder/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in snapshot:
                archive.write(ROOT/path, path)
    orders_sha = digest(folder/'orders.npy')
    begin_step = step
    stopped = [False]
    handlers = {s: signal.signal(s, lambda *_: stopped.__setitem__(0, True)) for s in (signal.SIGINT, signal.SIGTERM)}
    last_saved_step = step if args.resume else -1
    def checkpoint():
        nonlocal last_saved_step
        if snapshot != source_hashes():
            raise ValueError('Code changed while training; refusing inconsistent checkpoint')
        if last_saved_step == step:
            return
        last = folder/'last.pt'
        if last.exists() and digest(last) == last.with_suffix('.sha256').read_text().strip():
            shutil.copy2(last, folder/'previous.pt')
            shutil.copy2(last.with_suffix('.sha256'), folder/'previous.sha256')
        save(last, dict(format=FORMAT, recipe=config, sources=snapshot, inputs=inputs, orders_sha256=orders_sha,
                        step=step, cursor=cursor(step, data.per_epoch), model=model.state_dict(),
                        optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))
        last_saved_step = step
    print(f'TRAINING | {PARAMETER_BUDGET:,} reader parameters | seed47 | {step:,}/{data.total:,} updates | microbatch {args.microbatch}', flush=True)
    print(f'Full development evaluation every {args.eval_every:,} updates and at completion. Output: {folder}', flush=True)
    begin = time.perf_counter()
    try:
        if step == 0:
            checkpoint()
        while step < data.total and not stopped[0] and (args.stop_after is None or step-begin_step < args.stop_after):
            epoch, batch = divmod(step, data.per_epoch)
            indices = orders[epoch, batch*32:(batch+1)*32].tolist()
            actual = physical_order(data.datasets['train'], indices)
            counts = labels.counts(indices)
            targets = int(data.datasets['train'].rows[indices, 3].sum())
            packets = ((data.batch('train', actual[a:a+args.microbatch], 'compressed64', device),
                        labels.batch(actual[a:a+args.microbatch], device)) for a in range(0, len(actual), args.microbatch))
            metric = update(model, optimizer, packets, step+1, data.total, targets,
                            counts['query']['segments'], counts['source']['segments'], check=False)
            if metric['query_terms'] != 2*counts['query']['terms'] or metric['source_terms'] != counts['source']['terms']:
                raise ValueError('Auxiliary exposure count differs')
            step += 1
            entry = dict(step=step, epoch=epoch, batch_index=batch, sample_indices=indices,
                         micro_order=actual, microbatch=args.microbatch, cursor=cursor(step, data.per_epoch), **metric)
            with (folder/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(entry)+'\n')
            if step % args.log_every == 0 or step == begin_step+1 or step == data.total:
                speed = (step-begin_step)/(time.perf_counter()-begin)
                eta = (data.total-step)/max(speed, 1e-9)/60
                print(f'TRAIN {step:,}/{data.total:,} | CE {metric["ce"]:.5f} | {speed:.2f} updates/s | ETA {eta:.1f}m', flush=True)
                write_json(folder/'status.json', dict(state='training', pid=os.getpid(), total=data.total, **entry))
            bound = args.stop_after is not None and step-begin_step >= args.stop_after
            if step % args.save_every == 0 or step % args.eval_every == 0 or bound or stopped[0] or step == data.total:
                checkpoint()
            if step % args.eval_every == 0 or step == data.total:
                evaluate(model, data, folder, step, args.microbatch, device)
            if bound:
                break
        checkpoint()
        state = 'complete' if step == data.total else 'paused'
        write_json(folder/'status.json', dict(state=state, step=step, total=data.total))
        print(f'{state.upper()}: {folder} | {step:,}/{data.total:,} updates', flush=True)
    except BaseException as error:
        write_json(folder/'status.json', dict(state='failed', step=step, error=repr(error)))
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    args = parser().parse_args()
    pin_cpu()
    with worker_lock():
        execute(args)


if __name__ == '__main__':
    main()
