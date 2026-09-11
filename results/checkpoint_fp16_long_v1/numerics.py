"""NumPy FP64 comparison, native gradient as relative-error denominator."""

import math

import numpy as np


def compare(candidate, native):
    assert candidate.keys() == native.keys()
    ee = aa = bb = ab = 0.0
    per = {}
    for name in native:
        u, v = [d[name].numpy().astype(np.float64).ravel() for d in (candidate, native)]
        assert u.shape == v.shape and np.isfinite(u).all() and np.isfinite(v).all()
        delta = u - v
        e = float(delta @ delta)
        b = float(v @ v)
        ee += e
        aa += float(u @ u)
        bb += b
        ab += float(u @ v)
        per[name] = math.sqrt(e) / max(math.sqrt(b), 1e-8)
    error = math.sqrt(ee) / max(math.sqrt(bb), 1e-12)
    cosine = ab / math.sqrt(aa * bb)
    return dict(
        global_relative_l2=error,
        max_tensor_relative_l2=max(per.values()),
        cosine=cosine,
        per_tensor=per,
        passed=error <= 0.002 and max(per.values()) <= 0.02 and cosine >= 0.99999,
    )
