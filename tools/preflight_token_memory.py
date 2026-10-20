"""One bounded, zero-update capacity check for the canonical native256 model."""
from pathlib import Path
import sys
import time
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from models.token_memory import Model
from models.token_memory.dataset import Data
from models.token_memory.labels import Labels
from models.token_memory.objective import update
from models.token_memory.runtime import ROOT, worker_lock, write_json, pin_cpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if args.microbatch < 1:
        raise ValueError('Positive microbatch required')
    pin_cpu()
    torch.set_num_threads(1)
    with worker_lock():
        data, labels = Data(verify=False), Labels(verify=False)
        rows = data.datasets['train'].rows
        indices = np.lexsort(((rows[:, 2]-1)//64, rows[:, 2]-rows[:, 1]))[-args.microbatch:].tolist()
        packet = data.batch('train', indices, 'compressed64', args.device)
        terms = labels.batch(indices, args.device)
        counts = labels.counts(indices)
        model = Model().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        version = {k: v._version for k, v in model.state_dict(keep_vars=True).items()}
        if args.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        begin = time.perf_counter()
        score = update(model, optimizer, [(packet, terms)], 1, 48762,
                       int((packet[3] != -100).sum()), counts['query']['segments'],
                       counts['source']['segments'], do_step=False, check=True)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-begin
        assert not optimizer.state
        assert version == {k: v._version for k, v in model.state_dict(keep_vars=True).items()}
        result = dict(passed=True, production_optimizer_updates=0, parameters=sum(p.numel() for p in model.parameters()),
                      device=str(args.device), microbatch=args.microbatch, original_indices=indices,
                      raw_positions=packet[2].shape[1], memory_positions=packet[0].shape[1],
                      forward_backward_seconds=seconds, ce=score['ce'], finite_all_gradients=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
        folder = ROOT/'dump/token-memory-cleanup-20261020'
        folder.mkdir(exist_ok=True)
        write_json(folder/f'preflight-{args.device}-micro{args.microbatch}.json', result)
        print(f'PREFLIGHT PASS | {args.microbatch} examples | {packet[2].shape[1]} raw positions | {packet[0].shape[1]} memory positions | {seconds:.3f}s forward/backward | zero optimizer updates')
        if args.device == 'cuda':
            print(f'Peak GPU allocation: {result["peak_allocated_bytes"]/1024**3:.3f} GiB')


if __name__ == '__main__':
    main()
