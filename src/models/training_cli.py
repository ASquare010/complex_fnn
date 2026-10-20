"""Explicit fresh/resume training, readable progress and immutable run folders."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT.parent / 'latent-text-compressor/artifacts/data/comparison-v1/base'

def parser_for(model=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if model is None:
        p.add_argument('model', choices=['branch', 'compressor', 'status'])
    else:
        p.set_defaults(model=model)
    p.add_argument('--name', help='Run label saved in checkpoints and dump/<name>; refuses existing folders')
    p.add_argument('--resume', type=Path, help='Checkpoint or run folder; writes a NEW continuation folder')
    p.add_argument('--run', type=Path, help='Run folder for status')
    p.add_argument('--status', action='store_true', help='Read saved status with --run; no training')
    budget = p.add_mutually_exclusive_group()
    budget.add_argument('--steps', type=int, help='TOTAL optimizer steps, including restored steps (fresh default 1000)')
    budget.add_argument('--additional-steps', type=int, help='Explicit additional updates after --resume')
    p.add_argument('--microbatch', type=int, help='Paragraph windows per forward/backward pass (fresh default 16)')
    p.add_argument('--accumulation', type=int, help='Microbatches per optimizer update (fresh default 2)')
    p.add_argument('--batch-size', type=int, help='Effective paragraph-window batch; must divide by microbatch')
    p.add_argument('--context', type=int, help='Latent: vector positions; other models: tokens')
    p.add_argument('--width', type=int)
    p.add_argument('--heads', type=int)
    p.add_argument('--layers', type=int, help='Transformer layers (encoder layers for compressor)')
    p.add_argument('--decoder-layers', type=int, help='Compressor only')
    p.add_argument('--seed', type=int)
    p.add_argument('--lr', type=float, help='Fresh peak LR; on extension starts a new explicit schedule')
    p.add_argument('--min-lr', type=float)
    p.add_argument('--warmup', type=int)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--train-rows', type=int)
    p.add_argument('--verify-data', action='store_true', help='Opt in to full dataset hash, text round-trip and retokenization checks')
    p.add_argument('--sampling', choices=['replacement', 'shuffle'], help='shuffle visits each window once per pass')
    p.add_argument('--valid-rows', type=int)
    p.add_argument('--data', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--cpu', type=int, help='Logical CPU; default highest allowed core')
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--max-vram-gib', type=float, default=10)
    p.add_argument('--stop-after', type=int, help='Pause after this many updates in this invocation')
    p.add_argument('--inspect', action='store_true', help='Print resolved recipe without preparing data/training')
    return p

def pin_cpu(index=None):
    if os.name == 'nt':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process = k.GetCurrentProcess()
        allowed, system = (ctypes.c_size_t(), ctypes.c_size_t())
        if not k.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), 'Cannot inspect CPU affinity')
        cpus = [i for i in range(allowed.value.bit_length()) if allowed.value & 1 << i]
        selected = max(cpus) if index is None else index
        if selected not in cpus or not k.SetProcessAffinityMask(process, 1 << selected):
            raise ValueError(f'Unavailable CPU {selected}; allowed {cpus}')
    else:
        cpus = os.sched_getaffinity(0)
        selected = max(cpus) if index is None else index
        os.sched_setaffinity(0, {selected})
    return selected

def checkpoint_path(path):
    if path.is_dir():
        path = path / 'last.pt'
    if not path.is_file():
        raise ValueError(f'Checkpoint missing: {path}. Select an explicit milestone if needed.')
    return path.resolve()

def load_parent(path):
    import torch
    from models.training_data import digest
    path = checkpoint_path(path)
    expected = None
    if path.with_suffix('.sha256').is_file():
        expected = path.with_suffix('.sha256').read_text().strip()
    elif (path.parent / 'result.json').is_file() and path.stem in ('last', 'best'):
        expected = json.loads((path.parent / 'result.json').read_text()).get(path.stem + '_sha256')
    if expected is None or digest(path) != expected:
        raise ValueError('Missing or mismatched checkpoint checksum; refusing unsafe resume')
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('Checkpoint CRC failed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'recipe' not in state or 'step' not in state:
        raise ValueError('Unsupported historical checkpoint format; use its original trainer')
    return (path, state)

def schedule_lr(schedule, step):
    offset = step - schedule['start']
    span = schedule['end'] - schedule['start']
    progress = (offset - 1) / max(span - 1, 1)
    warm = min(offset / schedule['warmup'], 1) if schedule['warmup'] else 1
    return warm * (schedule['minimum'] + (schedule['peak'] - schedule['minimum']) * (1 + math.cos(math.pi * progress)) / 2)

def recipe_for(args, parent=None):
    from models.training_tasks import config_for
    defaults = dict(microbatch=16, accumulation=2, seed=47, train_rows=3000, valid_rows=200, weight_decay=0.01)
    if parent:
        r = dict(parent['recipe'])
        if r['model'] != args.model:
            raise ValueError('Checkpoint belongs to another model')
        for k in defaults:
            if getattr(args, k) is not None and getattr(args, k) != r[k]:
                raise ValueError(f'Resume preserves {k}')
        for k, ck in [('width', 'width'), ('heads', 'heads'), ('layers', 'layers' if args.model == 'branch' else 'encoder_layers'), ('context', 'context' if args.model == 'branch' else 'max_tokens'), ('decoder_layers', 'decoder_layers')]:
            if getattr(args, k) is not None and getattr(args, k) != r['config'].get(ck):
                raise ValueError('Resume architecture mismatch')
        if args.data is not None and str(args.data.resolve()) != r['data']:
            raise ValueError('Resume preserves data')
        if args.sampling is not None and args.sampling != r.get('sampling', 'replacement'):
            raise ValueError('Resume preserves sampling')
        end = parent['step'] + args.additional_steps if args.additional_steps is not None else args.steps if args.steps is not None else r['steps']
        if end <= parent['step']:
            raise ValueError('Increase total steps or use --additional-steps')
        if end != r['steps']:
            if end < r['steps']:
                raise ValueError('Cannot shorten saved schedule')
            peak = args.lr if args.lr is not None else schedule_lr(r['schedule'], parent['step'])
            r['schedule'] = dict(start=parent['step'], end=end, peak=peak, minimum=args.min_lr if args.min_lr is not None else peak, warmup=args.warmup or 0)
        elif any((getattr(args, k) is not None for k in ('lr', 'min_lr', 'warmup'))):
            raise ValueError('Resume preserves schedule')
        r.update(steps=end, start_step=parent['step'])
    else:
        if args.additional_steps is not None:
            raise ValueError('--additional-steps requires resume')
        r = dict(model=args.model, config=config_for(args.model, args), **defaults)
        for k in defaults:
            if getattr(args, k) is not None:
                r[k] = getattr(args, k)
        r.update(data=str((args.data or DEFAULT_DATA).resolve()), steps=args.steps or 1000, start_step=0, sampling=args.sampling or 'replacement')
        if args.steps == 0:
            raise ValueError('Steps must be positive')
        r['schedule'] = dict(start=0, end=r['steps'], peak=args.lr if args.lr is not None else 0.0003, minimum=args.min_lr if args.min_lr is not None else 3e-05, warmup=args.warmup if args.warmup is not None else 50)
    if args.decoder_layers is not None and args.model != 'compressor':
        raise ValueError('Decoder layers applies to compressor')
    if args.batch_size is not None:
        if args.batch_size < 1 or args.batch_size % r['microbatch']:
            raise ValueError('Invalid effective batch')
        acc = args.batch_size // r['microbatch']
        if (parent or args.accumulation is not None) and acc != r['accumulation']:
            raise ValueError('Batch conflicts with recipe')
        r['accumulation'] = acc
    for k in ('steps', 'microbatch', 'accumulation', 'train_rows', 'valid_rows'):
        if r[k] < 1:
            raise ValueError('Counts must be positive')
    if min(args.log_every, args.eval_every, args.save_every) < 1:
        raise ValueError('Intervals must be positive')
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError('stop-after must be positive')
    c = r['config']
    lr = r['schedule']
    if c['width'] < 1 or c['heads'] < 1 or c['width'] % c['heads'] or c['width'] // c['heads'] % 2:
        raise ValueError('Invalid heads/width')
    if c.get('layers', c.get('encoder_layers', 0)) < 1 or c.get('context', c.get('max_tokens', 0)) < 1:
        raise ValueError('Invalid layers/context')
    if not (0 < lr['minimum'] <= lr['peak'] and math.isfinite(lr['peak']) and (lr['warmup'] >= 0)):
        raise ValueError('Invalid LR schedule')
    if not math.isfinite(r['weight_decay']) or r['weight_decay'] < 0:
        raise ValueError('Invalid weight decay')
    if not math.isfinite(args.max_vram_gib) or args.max_vram_gib <= 0:
        raise ValueError('Invalid VRAM limit')
    r.update(cli_version=1, loss='cross_entropy_only', precision='fp32')
    return r

@contextmanager
def worker_lock(root):
    path = root / 'dump/.training-cli.lock'
    path.parent.mkdir(exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0)
        if path.stat().st_size == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError('Another CLI training worker is active') from e
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)

def architecture_hashes():
    from models.training_data import digest
    names = ['models/components.py', 'settings.py', 'models/branch_sigmoid/transformer.py', 'models/position_compressor/ffn.py', 'models/position_compressor/transformer.py', 'models/position_compressor/residual.py', 'models/position_compressor/codec.py']
    return {n: digest(ROOT / 'src' / n) for n in names}

def describe(r, task, args):
    count = sum((p.numel() for p in task.model.parameters()))
    context = r['config'].get('context', r['config'].get('max_tokens'))
    unit = 'tokens'
    print(f"\nRun: {r['run_name']}", flush=True)
    print(f"Model: {r['model']} | trainable: {count:,} parameters | FP32 | {args.device}", flush=True)
    print(f"Architecture: width={r['config']['width']}, heads={r['config']['heads']}, layers={r['config'].get('layers', r['config'].get('encoder_layers'))}, context={context} {unit}")
    print(f"Batch: microbatch {r['microbatch']} x accumulation {r['accumulation']} = {r['microbatch'] * r['accumulation']} paragraph windows/update (variable token counts)")
    print(f"Loss: {r['loss']} | updates {r['start_step']} -> {r['steps']} ({r['steps'] - r['start_step']} remaining) | seed {r['seed']}")
    s = r['schedule']
    print(f"LR: {s['peak']:g} -> {s['minimum']:g}, warmup={s['warmup']}, schedule steps {s['start']} -> {s['end']}")
    print(f"Data: {r['train_rows']} train / {r['valid_rows']} development paragraphs; evaluate every {args.eval_every}, save every {args.save_every}")
    print('Ctrl+C requests a checkpoint after the current optimizer update. No automatic retries.\n', flush=True)
    return count

def execute(args, recipe, out, parent_path=None, parent=None):
    import shutil
    import torch
    from tokenizers import Tokenizer
    from models.training_data import digest, read_source
    from models.token_memory.runtime import save, write_json
    from models.training_tasks import Task
    recipe = dict(recipe)
    recipe['run_name'] = out.name
    metric = 'ce'
    if parent:
        recipe['parent_run_name'] = parent['recipe'].get('run_name', parent_path.parent.name)
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        reason = 'This Python has CPU-only PyTorch.' if torch.version.cuda is None else 'PyTorch includes CUDA, but no CUDA device is available.'
        raise RuntimeError(f'{reason} Active Python: {sys.executable}. Use an existing CUDA-enabled Python executable, or explicitly select --device cpu. No training started.')
    torch.set_num_threads(1)
    torch.manual_seed(recipe['seed'])
    task = Task(recipe, args.device)
    count = describe(recipe, task, args)
    if args.inspect:
        if parent:
            task.model.load_state_dict(parent['model'], strict=True)
            print(f'Resume checkpoint: {parent_path}\nPrepared data will be verified before training.')
        print('Inspection only: no data preparation or optimizer updates.')
        return
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', {'state': 'preparing', 'run_name': out.name, 'pid': os.getpid()})
    with zipfile.ZipFile(out / 'source.zip', 'w') as archive:
        for path in (ROOT / 'src').rglob('*.py'):
            archive.write(path, str(path.relative_to(ROOT)))
    hashes = architecture_hashes()
    implementation = {name: digest(ROOT / 'src' / name) for name in ('models/training_cli.py', 'models/training_tasks.py', 'models/training_data.py', 'models/token_memory/runtime.py')}
    previous_handlers = {}
    stopping = [False]

    def request_stop(*_):
        if not stopping[0]:
            print('\nPause requested; finishing the current safe boundary...', flush=True)
        stopping[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    try:
        if parent:
            if 'implementation' in parent and parent['implementation'] != implementation:
                raise ValueError('Trainer implementation changed; use the run source snapshot to resume')
            if 'architecture' in parent:
                expected = parent['architecture']
            else:
                original = json.loads((parent_path.parent / 'source_hashes.json').read_text())
                expected = {n: original[n] for n in hashes}
            if expected != hashes:
                raise ValueError('Architecture/source mismatch; use the saved sources for resume')
            prepared_path = parent_path.parent / 'prepared.pt'
            if digest(prepared_path) != recipe['prepared_sha256']:
                raise ValueError('Prepared data checksum mismatch')
            prepared = torch.load(prepared_path, weights_only=True, map_location='cpu')
            shutil.copy2(prepared_path, out / 'prepared.pt')
            identity = json.loads((parent_path.parent / 'data_identity.json').read_text())
            print('Verified saved prepared data; no re-encoding.', flush=True)
        else:
            print('Checking full data hashes, decoded text and split separation...' if args.verify_data else 'Fast loading: reuse saved token IDs; full text/data-hash verification skipped.', flush=True)
            task.tokenizer = Tokenizer.from_file(str(Path(recipe['data']) / 'tokenizer.json'))
            task.tokenizer.encode_special_tokens = True
            rows, identity = read_source(recipe['data'], task.tokenizer, (recipe['train_rows'], recipe['valid_rows']), verify=args.verify_data)
            print(f"Loaded {len(rows['train'])} train and {len(rows['valid'])} development windows.", flush=True)
            print('Preparing token windows...', flush=True)
            prepared = rows
            save(out / 'prepared.pt', prepared)
            recipe['prepared_sha256'] = digest(out / 'prepared.pt')
        write_json(out / 'data_identity.json', identity)
        examples = {s: task.examples(v) for s, v in prepared.items()}
        print(f"Windows: {len(examples['train'])} train / {len(examples['valid'])} development.", flush=True)
        recipe['trainable_parameters'] = count
        recipe['max_paragraph_chunks'] = max((len(r['tokens']) for r in prepared['train']))
        optimizer = torch.optim.AdamW(task.model.parameters(), lr=recipe['schedule']['peak'], weight_decay=recipe['weight_decay'])
        rng = random.Random(recipe['seed'])
        step = recipe['start_step']
        if parent:
            task.model.load_state_dict(parent['model'], strict=True)
            optimizer.load_state_dict(parent['optimizer'])
            rng.setstate(parent['python_rng'])
            torch.set_rng_state(parent['torch_rng'].cpu())
            if args.device == 'cuda' and parent['cuda_rng']:
                torch.cuda.set_rng_state_all([v.cpu() for v in parent['cuda_rng']])
            recipe['parent_checkpoint_sha256'] = digest(parent_path)
            recipe['parent_checkpoint'] = str(parent_path)
        write_json(out / 'recipe.json', recipe)
        history = []
        best = float('inf')

        def snapshot():
            return dict(run_name=out.name, model=task.model.state_dict(), optimizer=optimizer.state_dict(), recipe=recipe, step=step, python_rng=rng.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [], architecture=hashes, implementation=implementation)

        def checkpoint(filename):
            save(out / filename, snapshot())
            (out / filename).with_suffix('.sha256').write_text(digest(out / filename))

        def validate():
            nonlocal best
            print(f'Validating at step {step}...', flush=True)
            score = task.evaluate(examples['valid'], recipe['microbatch'])
            history.append({'step': step, **score})
            write_json(out / 'validation.json', history)
            detail = f" | token accuracy {score['token_accuracy']:.2%} | exact {score['exact']}/{score['sequences']}" if metric == 'ce' else ' | vector error; no token accuracy measured here'
            print(f'VALID {step}: {metric.upper()} {score[metric]:.6g}' + detail, flush=True)
            if score[metric] < best:
                best = score[metric]
                checkpoint('best.pt')
            return score
        checkpoint('initial.pt')
        checkpoint('last.pt')
        validate()
        begin = time.monotonic()
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with (out / 'training.jsonl').open('x', encoding='utf-8') as log:
            shuffled_epoch, shuffled_order = (None, None)
            while step < recipe['steps'] and (not stopping[0]):
                batch_size = recipe['microbatch'] * recipe['accumulation']
                if recipe.get('sampling', 'replacement') == 'shuffle':
                    samples = []
                    for draw in range(step * batch_size, (step + 1) * batch_size):
                        epoch, offset = divmod(draw, len(examples['train']))
                        if epoch != shuffled_epoch:
                            shuffled_order = list(range(len(examples['train'])))
                            random.Random(recipe['seed'] + epoch).shuffle(shuffled_order)
                            shuffled_epoch = epoch
                        samples.append(examples['train'][shuffled_order[offset]])
                else:
                    samples = [examples['train'][rng.randrange(len(examples['train']))] for _ in range(batch_size)]
                total_tokens = task.count(samples)
                lr = schedule_lr(recipe['schedule'], step + 1)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                value = 0.0
                for i in range(0, len(samples), recipe['microbatch']):
                    loss, stats = task.loss(samples[i:i + recipe['microbatch']])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss; stopped without retry')
                    weighted = loss * (stats['tokens'] / total_tokens)
                    weighted.backward()
                    value += float(weighted.detach())
                norm = torch.nn.utils.clip_grad_norm_(task.model.parameters(), 1.0, error_if_nonfinite=True)
                memory = torch.cuda.max_memory_reserved() / 1024 ** 3 if args.device == 'cuda' else 0
                if memory > args.max_vram_gib:
                    raise RuntimeError('VRAM guard exceeded; no automatic retry')
                optimizer.step()
                step += 1
                row = dict(step=step, lr=lr, targets=total_tokens, grad_norm=float(norm), **{metric: value})
                log.write(json.dumps(row) + '\n')
                log.flush()
                if step % args.log_every == 0 or step == recipe['start_step'] + 1:
                    elapsed = time.monotonic() - begin
                    rate = (step - recipe['start_step']) / max(elapsed, 1e-06)
                    eta = (recipe['steps'] - step) / rate
                    print(f"[{out.name}] TRAIN {step:>6}/{recipe['steps']} ({step / recipe['steps']:5.1%}) | {metric.upper()} {value:10.6g} | LR {lr:.3g} | {rate:.1f} updates/s | ETA {eta / 60:.1f}m | reserved {memory:.2f} GiB", flush=True)
                    os.fsync(log.fileno())
                    write_json(out / 'status.json', dict(state='training', run_name=out.name, pid=os.getpid(), total=recipe['steps'], **row))
                if step % args.save_every == 0:
                    checkpoint(f'step-{step}.pt')
                    checkpoint('last.pt')
                if step % args.eval_every == 0:
                    validate()
                if args.stop_after and step - recipe['start_step'] >= args.stop_after:
                    stopping[0] = True
        checkpoint('last.pt')
        finished = step == recipe['steps']
        if history[-1]['step'] != step:
            validate()
        restored = torch.load(out / 'last.pt', weights_only=True, map_location=args.device)
        task.model.load_state_dict(restored['model'])
        replay = task.evaluate(examples['valid'], recipe['microbatch'])
        if replay != {k: v for k, v in history[-1].items() if k != 'step'}:
            raise ValueError('Saved checkpoint validation replay mismatch')
        result = dict(state='complete' if finished else 'paused', run_name=out.name, step=step, total=recipe['steps'], final=replay, last_sha256=digest(out / 'last.pt'), best_sha256=digest(out / 'best.pt'), final_reload_replay=True, elapsed_seconds=time.monotonic() - begin)
        write_json(out / 'result.json', result)
        write_json(out / 'status.json', result)
        module_folder = {'branch': 'branch_sigmoid', 'compressor': 'position_compressor'}[args.model]
        print(f'''\n{result['state'].upper()}: step {step}/{recipe['steps']}. Validation {metric.upper()} {replay[metric]:.6g}.\nCheckpoint: {out / 'last.pt'}\nResume: python src/models/{module_folder}/training.py --resume "{out}"''' + (' --additional-steps <N>' if finished else ''), flush=True)
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', run_name=out.name, error=repr(error)))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

def main(root=None, model=None):
    args = parser_for(model).parse_args()
    os.chdir(root or ROOT)
    if args.model == 'status' or args.status:
        if not args.run:
            raise SystemExit('status requires --run <folder>')
        for name in ('recipe.json', 'status.json'):
            print(f'{name}:\n{(args.run / name).read_text()}')
        return
    try:
        cpu = pin_cpu(args.cpu)
        print(f'CPU affinity: {cpu} | Python: {sys.executable}', flush=True)
        parent_path, parent = load_parent(args.resume) if args.resume else (None, None)
        recipe = recipe_for(args, parent)
        name = args.name or f'{args.model}-' + ('resume-' if parent else '') + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('--name must be a single folder name')
        out = (ROOT / 'dump' / name).resolve()
        if out.exists():
            raise ValueError(f'Output already exists: {out}; choose a new name')
        print(f"Mode: {('RESUME' if parent else 'FRESH')} | Output: {out}", flush=True)
        if args.inspect:
            execute(args, recipe, out, parent_path, parent)
        else:
            with worker_lock(ROOT):
                execute(args, recipe, out, parent_path, parent)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f'Training refused/stopped: {error}') from error
if __name__ == '__main__':
    main()
