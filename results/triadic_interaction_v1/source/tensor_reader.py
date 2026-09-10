"""Read this experiment's dense CPU tensors without importing Torch.

Uses the documented torch.save ZIP container. This deliberately limited reader
rejects every pickle global except observed tensor constructors/storage types.
It is for audited local evidence, not a general checkpoint or model loader.
BF16 payloads are decoded exactly into FP32 for numerical comparison.
"""

import io
import pickle
import zipfile
from collections import OrderedDict

import numpy as np


def rebuild(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    raw, dtype = storage
    encoded = np.dtype("<u2" if dtype == "bf16" else dtype)
    value = np.ndarray(
        tuple(size),
        dtype=encoded,
        buffer=raw,
        offset=offset * encoded.itemsize,
        strides=tuple(s * encoded.itemsize for s in stride),
    )
    if dtype == "bf16":
        return (value.astype(np.uint32) << 16).view(np.float32)
    return value


class DenseReader(pickle.Unpickler):
    def __init__(self, archive, prefix):
        super().__init__(io.BytesIO(archive.read(prefix + "data.pkl")))
        self.archive, self.prefix, self.storages = archive, prefix, {}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        types = {
            "FloatStorage": "<f4",
            "DoubleStorage": "<f8",
            "LongStorage": "<i8",
            "BFloat16Storage": "bf16",
        }
        if module == "torch" and name in types:
            return types[name]
        raise ValueError(f"Unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, pid):
        kind, dtype, key, location, elements = pid
        assert kind == "storage" and location == "cpu"
        if key not in self.storages:
            raw = self.archive.read(self.prefix + "data/" + str(key))
            itemsize = 2 if dtype == "bf16" else np.dtype(dtype).itemsize
            assert len(raw) == elements * itemsize
            self.storages[key] = (raw, dtype)
        assert self.storages[key][1] == dtype
        return self.storages[key]


def load(path):
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        assert len(matches) == 1
        prefix = matches[0][: -len("data.pkl")]
        assert archive.read(prefix + "byteorder") == b"little"
        return DenseReader(archive, prefix).load()
