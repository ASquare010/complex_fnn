"""Small shared process/checkpoint helpers; no model selection or objectives."""
import hashlib,json,os,zipfile
from pathlib import Path
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[3]
ENCODER=ROOT/'dump/pretrained-latent64-fced6bd/encoder.pt'
ENCODER_SHA='96f3b38735355cec0f0e8693232f3fd3f80bc411fd4e06b8218449cc35949bd7'

def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def pin_cpu():
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        p=k.GetCurrentProcess();a=ctypes.c_size_t();s=ctypes.c_size_t()
        if not k.GetProcessAffinityMask(p,ctypes.byref(a),ctypes.byref(s)): raise OSError('Cannot read CPU affinity')
        selected=a.value.bit_length()-1
        if not k.SetProcessAffinityMask(p,1<<selected): raise OSError('Cannot set CPU affinity')
    else:
        selected=max(os.sched_getaffinity(0));os.sched_setaffinity(0,{selected})
    return selected

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save(path,state):
    import torch
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f: torch.save(state,f);f.flush();os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None: raise ValueError('Checkpoint CRC failed')
    os.replace(tmp,path);path.with_suffix('.sha256').write_text(digest(path))

def load(path):
    import torch
    path=Path(path)
    if path.is_dir(): path=path/'last.pt'
    if digest(path)!=path.with_suffix('.sha256').read_text().strip(): raise ValueError('Checkpoint hash mismatch')
    return torch.load(path,weights_only=True,map_location='cpu')

@contextmanager
def worker_lock():
    path=ROOT/'dump/.training-cli.lock';path.parent.mkdir(exist_ok=True)
    with path.open('a+b') as f:
        if path.stat().st_size==0: f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e: raise RuntimeError('Another training worker is active; stop it before starting this run') from e
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)
