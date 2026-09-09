"""Durable artifact writes with immediate semantic readback; not a storage diagnosis."""

import hashlib
import json
import os
import zipfile
from pathlib import Path

import torch


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value, *, exclusive=True):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb" if exclusive else "wb") as f:
        assert f.write(payload) == len(payload)
        f.flush()
        os.fsync(f.fileno())
    assert Path(path).read_bytes() == payload
    assert read(path) == value


def equal_payload(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_payload(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_payload(a, b)
    else:
        assert actual == expected


def record_observation(root, label, kind, data, tensors=None):
    result = {"status": "PASS", "kind": kind, "optimizer_updates": 0, **data}
    if tensors is not None:
        path = root / (label + ".pt")
        with path.open("xb") as f:
            torch.save(tensors, f)
            f.flush()
            os.fsync(f.fileno())
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
        restored = torch.load(path, map_location="cpu", weights_only=True)
        equal_payload(restored, tensors)
        result.update(tensor_file=path.name, tensor_sha256=sha(path),
                      tensor_payload_readback_exact=True)
    write_json(root / (label + ".json"), result)
