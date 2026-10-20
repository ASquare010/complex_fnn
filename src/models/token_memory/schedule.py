import math,json,os,time
from pathlib import Path
def rate(step, total, peak):
    warmup = min(200, total)
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)

def cursor(step, per_epoch):
    epoch, batch_index = divmod(step, per_epoch)
    return dict(epoch=epoch, batch_index=batch_index)

def rewind_logs(out, step):
    """Keep evidence of uncheckpointed work; replay from the durable checkpoint."""
    for filename in ('training.jsonl', 'validation.jsonl'):
        path = out / filename
        if not path.exists():
            continue
        keep, discarded = ([], [])
        original = path.read_text()
        for line in original.splitlines(keepends=True):
            try:
                record = json.loads(line)
                if record['step'] <= step:
                    keep.append(line.rstrip('\r\n') + '\n')
                else:
                    discarded.append(line)
            except (ValueError, KeyError):
                discarded.append(line)
        if discarded:
            stamp = time.time_ns()
            (out / f'{filename}.discarded-{stamp}').write_text(''.join(discarded))
            print(f'Preserved {len(discarded)} uncheckpointed/malformed {filename} rows; resuming step {step}.', flush=True)
        if ''.join(keep) != original:
            path.write_text(''.join(keep))
