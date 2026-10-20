"""Exclusive worker lock for the raw baseline."""
from contextlib import contextmanager
import os
from models.token_memory.runtime import ROOT


@contextmanager
def worker_lock(arm):
    if arm!='raw_transformer':raise ValueError('Unknown comparison arm')
    path=ROOT/'dump'/f'.{arm}.lock';path.parent.mkdir(exist_ok=True)
    with path.open('a+b') as f:
        if path.stat().st_size==0:f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError(f'Another {arm} worker is active') from e
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)
